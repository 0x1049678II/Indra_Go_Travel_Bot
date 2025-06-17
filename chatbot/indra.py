"""
Author: Cameron Murphy
Date: June 4th 2025

Indra chatbot with features:
- Fuzzy string matching for location recognition
- Conversation context tracking and state management
- Confidence scoring system for intent classification
- Concurrent API calls using asyncio
- Regex patterns and natural language processing
"""

import difflib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple

import structlog
# ChatterBot imports (using ShoneGK's maintained fork)
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer

from chatbot.analytics import TravelAnalytics
from chatbot.exceptions import (
    IndraException, ChatterPyException, ChatterPyTrainingException,
    LocationNotFoundException
)
from chatbot.services.async_weather_service import AsyncWeatherService
from chatbot.services.news_service import NewsService, NewsCategory
from chatbot.utils import async_retry, performance_monitor, safe_gather
from chatbot.validators import EnglandLocationValidator
from config.locations import EnglandLocations
from config.settings import Config

logger = structlog.get_logger()


class IntentType(Enum):
    """Enumeration of supported intent types with confidence thresholds."""
    GREETING = "greeting"
    FAREWELL = "farewell"
    WEATHER_CURRENT = "weather_current"
    WEATHER_FORECAST = "weather_forecast"
    WEATHER_COMPARISON = "weather_comparison"
    ACTIVITIES = "activities"
    NEWS = "news"
    LOCATION_INFO = "location_info"
    AFFIRMATIVE = "affirmative"
    NEGATIVE = "negative"
    GENERAL = "general"


@dataclass
class ConversationContext:
    """Tracks conversation state and context for follow-ups."""
    last_location: Optional[str] = None
    last_intent: Optional[IntentType] = None
    last_response_type: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    session_start: datetime = field(default_factory=datetime.now)
    turn_count: int = 0
    
    def add_turn(self, user_input: str, intent: IntentType, location: Optional[str], 
                 confidence: float, bot_response: str):
        """Add a conversation turn to the history."""
        self.turn_count += 1
        self.conversation_history.append({
            'turn': self.turn_count,
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'intent': intent.value,
            'location': location,
            'confidence': confidence,
            'bot_response': bot_response[:200] + "..." if len(bot_response) > 200 else bot_response
        })
        
        # Update current context
        if location:
            self.last_location = location
        self.last_intent = intent
        
        # Keep only last 10 turns for memory efficiency
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]


@dataclass
class IntentClassificationResult:
    """Result of intent classification with confidence scoring."""
    intent: IntentType
    confidence: float
    location: Optional[str]
    patterns_matched: List[str]
    fuzzy_matches: List[Tuple[str, float]]


def _calculate_pattern_confidence(pattern: str, match: re.Match, text: str) -> float:
    """Calculate confidence score for a pattern match."""
    base_confidence = 0.7

    # Boost confidence for more specific patterns
    if len(pattern) > 50:  # Complex patterns
        base_confidence += 0.15
    elif len(pattern) > 30:  # Medium complexity
        base_confidence += 0.10

    # Boost confidence for exact matches
    if match.group().strip() == text.strip():
        base_confidence += 0.20

    # Boost confidence for longer matches
    match_coverage = len(match.group()) / len(text)
    if match_coverage > 0.7:
        base_confidence += 0.10
    elif match_coverage > 0.5:
        base_confidence += 0.05

    return min(base_confidence, 1.0)


def _get_contextual_intent(last_intent: IntentType, response_type: IntentType) -> IntentType:
    """Determine contextual intent based on previous interaction."""
    if response_type == IntentType.AFFIRMATIVE:
        if last_intent == IntentType.WEATHER_CURRENT:
            return IntentType.WEATHER_FORECAST
        elif last_intent == IntentType.WEATHER_FORECAST:
            return IntentType.ACTIVITIES
        elif last_intent == IntentType.ACTIVITIES:
            return IntentType.NEWS
        else:
            return last_intent
    return IntentType.GENERAL


def _extract_location_fuzzy(text: str) -> Tuple[Optional[str], List[Tuple[str, float]]]:
    """
    Extract location from text using fuzzy string matching.

    Args:
        text: Input text to search for locations

    Returns:
        Tuple of (best_location, fuzzy_matches_list)
    """
    fuzzy_matches = []
    best_location = None
    best_score = 0.0

    # Extract potential location words (all words, not just capitalized)
    words = text.split()
    potential_locations = []

    # Check individual words and combinations
    clean_pattern = r'[^\w\s]'
    for i in range(len(words)):
        # Single words
        word = re.sub(clean_pattern, '', words[i]).strip()
        if len(word) > 2:
            potential_locations.append(word)

        # Two-word combinations for compound locations
        if i < len(words) - 1:
            word1 = re.sub(clean_pattern, '', words[i])
            word2 = re.sub(clean_pattern, '', words[i+1])
            combo = f"{word1} {word2}".strip()
            potential_locations.append(combo)

    # Remove duplicates and filter by length
    potential_locations = list(set([loc for loc in potential_locations if len(loc) > 2]))

    for potential in potential_locations:
        for valid_location in Config.VALID_LOCATIONS:
            # Calculate similarity using difflib
            similarity = difflib.SequenceMatcher(None, potential.lower(), valid_location.lower()).ratio()

            if similarity >= 0.8:  # 80% similarity threshold
                fuzzy_matches.append((valid_location, similarity))
                if similarity > best_score:
                    best_score = similarity
                    best_location = valid_location

    # Sort fuzzy matches by score
    fuzzy_matches.sort(key=lambda x: x[1], reverse=True)

    return best_location, fuzzy_matches


class IntentClassifier:
    """Intent classification with fuzzy matching and confidence scoring."""
    
    def __init__(self):
        #  Intent patterns regex
        self.intent_patterns = {
            IntentType.WEATHER_CURRENT: [
                r'\b(weather|temperature|conditions?|climate)\b.*\b(today|now|current|currently|right\s+now)\b',
                r'\b(what\s*\'?s|how\s*\'?s|tell\s+me)\b.*\b(weather|temperature|conditions?)\b(?!.*\b(forecast|tomorrow|week|days?)\b)',
                r'\b(current|today\s*\'?s)\b.*\b(weather|temperature|conditions?)\b',
                r'^\s*(weather|temperature|conditions?)\s*(?:in|at|for)?\s*\w+\s*$',
                r'\b(is\s+it|will\s+it\s+be)\b.*\b(sunny|rainy|cold|warm|hot)\b.*\b(today|now)\b'
            ],
            IntentType.WEATHER_FORECAST: [
                r'\b(forecast|prediction|outlook)\b',
                r'\b(weather|temperature|conditions?)\b.*\b(tomorrow|week|days?|future|next|coming)\b',
                r'\b(5\s*[\-\s]*day|five\s*[\-\s]*day|weekly)\b.*\b(forecast|weather|prediction)\b',
                r'\b(next|coming|this)\b.*\b(week|days?)\b.*\b(weather|temperature|forecast)\b',
                r'\b(what\s+will|how\s+will)\b.*\b(weather|temperature)\b.*\b(be|look)\b'
            ],
            IntentType.WEATHER_COMPARISON: [
                r'\b(compare|comparison|versus|vs\.?|between)\b.*\b(weather|temperature|conditions?)\b',
                r'\b(better|worse|warmer|colder|sunnier|rainier)\b.*\b(weather|temperature)\b',
                r'\b(which|where)\b.*\b(warmer|colder|sunnier|rainier|better|worse)\b',
                r'\b(difference|differences)\b.*\b(weather|temperature|climate)\b'
            ],
            IntentType.ACTIVITIES: [
                r'\b(activities|things\s+to\s+do|attractions|visit|see|explore|do)\b',
                r'\b(what\s+(can|should|could)\s+I|what\s+to)\b.*\b(do|visit|see|explore)\b',
                r'\b(recommendations|suggestions|ideas|advice)\b.*\b(activities|things|places|attractions)\b',
                r'\b(fun|interesting|exciting)\b.*\b(activities|things|places)\b',
                r'\b(tourist|sightseeing|tourism)\b.*\b(activities|attractions|sites)\b'
            ],
            IntentType.NEWS: [
                r'\b(news|events|happening|latest|updates|current\s+events)\b',
                r'\b(what\s*\'?s\s+(happening|going\s+on|new))\b',
                r'\b(events|festivals|shows|exhibitions|concerts)\b',
                r'\b(local|recent|breaking)\b.*\b(news|events)\b'
            ],
            IntentType.LOCATION_INFO: [
                r'\b(tell\s+me\s+about|information\s+about|about|details\s+about)\b',
                r'\b(describe|explain|overview)\b',
                r'\b(what\s+is|where\s+is|how\s+is)\b',
                r'^\s*\w+\s*$',  # Single word (likely location name)
                r'\b(show\s+me|give\s+me)\b.*\b(info|information|details)\b'
            ],
            IntentType.AFFIRMATIVE: [
                r'^\s*(yes|yeah|yep|sure|ok|okay|alright|absolutely|definitely|please|go\s+ahead)\s*$',
                r'^\s*(yes\s+please|that\s+sounds\s+good|sounds\s+great|perfect)\s*$',
                r'^\s*(i\s+would\s+like|i\s+\'?d\s+like|that\s+would\s+be\s+great)\s*.*$'
            ],
            IntentType.NEGATIVE: [
                r'^\s*(no|nope|nah|not\s+really|no\s+thanks?|not\s+interested)\s*$',
                r'^\s*(maybe\s+later|not\s+now|skip\s+that)\s*$'
            ]
        }
        
        # Greeting patterns
        self.greeting_patterns = [
            r'\b(hello|hi|hey|good\s+(morning|afternoon|evening)|greetings|salutations)\b',
            r'^\s*(hi|hello|hey)\s*(?:there|indra)?\s*$'
        ]
        
        # Farewell patterns
        self.farewell_patterns = [
            r'\b(goodbye|bye|farewell|see\s+you|thanks?|thank\s+you|cheers)\b',
            r'\b(have\s+a\s+good|take\s+care|until\s+next\s+time)\b',
            r'^\s*(bye|goodbye|thanks?|cheers)\s*$'
        ]
    
    def classify_intent(self, text: str, context: ConversationContext) -> IntentClassificationResult:
        """
        Classify user intent with confidence scoring and context awareness.
        
        Args:
            text: User input text
            context: Current conversation context
            
        Returns:
            IntentClassificationResult with intent, confidence, and location
        """
        text_lower = text.lower().strip()
        patterns_matched = []
        max_confidence = 0.0
        best_intent = IntentType.GENERAL
        
        # Check for greetings first
        for pattern in self.greeting_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                patterns_matched.append(pattern)
                return IntentClassificationResult(
                    intent=IntentType.GREETING,
                    confidence=0.95,
                    location=None,
                    patterns_matched=patterns_matched,
                    fuzzy_matches=[]
                )
        
        # Check for farewells
        for pattern in self.farewell_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                patterns_matched.append(pattern)
                return IntentClassificationResult(
                    intent=IntentType.FAREWELL,
                    confidence=0.95,
                    location=None,
                    patterns_matched=patterns_matched,
                    fuzzy_matches=[]
                )
        
        # Context-aware affirmative/negative detection
        if context.last_intent and context.turn_count > 0:
            for pattern in self.intent_patterns[IntentType.AFFIRMATIVE]:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    patterns_matched.append(pattern)
                    # Use context to determine the actual intent
                    contextual_intent = _get_contextual_intent(context.last_intent, IntentType.AFFIRMATIVE)
                    return IntentClassificationResult(
                        intent=contextual_intent,
                        confidence=0.90,
                        location=context.last_location,
                        patterns_matched=patterns_matched,
                        fuzzy_matches=[]
                    )
            
            for pattern in self.intent_patterns[IntentType.NEGATIVE]:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    patterns_matched.append(pattern)
                    return IntentClassificationResult(
                        intent=IntentType.NEGATIVE,
                        confidence=0.90,
                        location=None,
                        patterns_matched=patterns_matched,
                        fuzzy_matches=[]
                    )
        
        # Main intent classification with confidence scoring
        for intent, patterns in self.intent_patterns.items():
            intent_confidence = 0.0
            intent_patterns_matched = []
            
            for pattern in patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    intent_patterns_matched.append(pattern)
                    # Calculate confidence based on pattern specificity and match quality
                    pattern_confidence = _calculate_pattern_confidence(pattern, match, text_lower)
                    intent_confidence = max(intent_confidence, pattern_confidence)
            
            if intent_confidence > max_confidence:
                max_confidence = intent_confidence
                best_intent = intent
                patterns_matched = intent_patterns_matched
        
        # Extract location with fuzzy matching
        location, fuzzy_matches = _extract_location_fuzzy(text)
        
        # Boost confidence if location is found and intent makes sense
        if location and best_intent in [IntentType.WEATHER_CURRENT, IntentType.WEATHER_FORECAST, 
                                       IntentType.ACTIVITIES, IntentType.NEWS, IntentType.LOCATION_INFO]:
            max_confidence = min(max_confidence + 0.15, 1.0)
        
        # Use context to boost confidence for location-less queries
        if not location and context.last_location and best_intent != IntentType.GENERAL:
            location = context.last_location
            max_confidence = min(max_confidence + 0.10, 1.0)
        
        return IntentClassificationResult(
            intent=best_intent,
            confidence=max_confidence,
            location=location,
            patterns_matched=patterns_matched,
            fuzzy_matches=fuzzy_matches
        )


def _parse_training_data() -> List[str]:
    """Parse training data from conversations.txt file."""
    try:
        with open(Config.CHATTERPY_TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse conversation pairs
        conversations = []
        lines = content.split('\n')
        current_user = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('User: '):
                current_user = line[6:]  # Remove 'User: '
            elif line.startswith('Bot: ') and current_user:
                bot_response = line[5:]  # Remove 'Bot: '
                conversations.extend([current_user, bot_response])
                current_user = None

        return conversations

    except Exception as e:
        logger.error("Failed to parse training data", error=str(e))
        return []


def _analyze_temperature_trend(temperatures: List[float]) -> str:
    """Analyze temperature trend over the forecast period."""
    if len(temperatures) < 3:
        return "Insufficient data for trend analysis"

    # Simple trend analysis
    increasing = sum(1 for i in range(1, len(temperatures)) if temperatures[i] > temperatures[i-1])
    decreasing = sum(1 for i in range(1, len(temperatures)) if temperatures[i] < temperatures[i-1])

    if increasing > decreasing:
        return "Temperatures trending warmer"
    elif decreasing > increasing:
        return "Temperatures trending cooler"
    else:
        return "Temperatures remaining stable"


class IndraBot:
    """
    Indra - The England travel assistant chatbot.
    
    Named after the Hindu god of weather and storms (seemed fitting for a travel bot 
    that deals with English weather), Indra provides (somewhat)intelligent travel assistance
    for the England destinations with features including fuzzy location matching,
    conversation context tracking, and concurrent API operations.
    
    The bot attempts to go beyond simple question-answer patterns by maintaining conversation
    context, making suggestions, and fetching multiple data sources
    concurrently for faster responses. It's designed to handle the unpredictable
    nature of both English weather and human conversation with equal grace.
    
    Features:
        - Fuzzy string matching for location recognition (handles typos)
        - Conversation context tracking and state management
        - Confidence scoring system for intent classification
        - Concurrent API calls using asyncio for optimal performance
        - Multiple API integration (OpenWeather + NewsAPI)
        - Caching and rate limiting
        - Natural language processing with regex patterns
        - ChatterBot fallback for unclassified intents
    
    Attributes:
        validator: Location validation system with fuzzy matching
        analytics: Travel analytics and weather trend analysis
        weather_service: Async weather service for concurrent operations
        news_service: News and events service with category filtering
        intent_classifier: Intent classification with confidence scoring
        conversation_context: Context tracking for follow-ups
        chatbot: ChatterBot instance for fallback responses
        
    Note:
        Uses async context manager pattern for proper resource lifecycle management.
        Call using: async with IndraBot() as bot:
    """
    
    def __init__(self):
        """
        Initialize the Indra chatbot with all necessary services.
        
        Sets up the core components including location validation, weather/news services,
        intent classification, and conversation context tracking. Uses ChatterBot
        (ShoneGK's maintained fork) for fallback responses when our intent classification
        doesn't quite hit the mark.
        
        The initialization is deliberately lightweight - heavy async operations
        are deferred to the async context manager to avoid blocking the main thread.
        """
        self.validator = None
        self.analytics = None
        self.config = Config()
        self.locations = EnglandLocations()
        self.weather_service = None
        self.news_service = None
        self._is_initialized = False
        
        # Conversation features
        self.intent_classifier = IntentClassifier()
        self.conversation_context = ConversationContext()
        
        # Initialize ChatterBot (using ShoneGK's maintained fork)
        self.chatbot = ChatBot(
            'Indra',
            storage_adapter='chatterbot.storage.SQLStorageAdapter',
            database_uri=Config.CHATTERPY_DATABASE_URI,
            logic_adapters=[
                {
                    'import_path': 'chatterbot.logic.BestMatch',
                    'default_response': 'I\'m sorry, I didn\'t understand that. Could you ask about weather, activities, or news for England destinations?',
                    'maximum_similarity_threshold': 0.90
                },
                {
                    'import_path': 'chatterbot.logic.MathematicalEvaluation'
                }
            ],
            preprocessors=[
                'chatterbot.preprocessors.clean_whitespace',
                'chatterbot.preprocessors.unescape_html',
                'chatterbot.preprocessors.convert_to_ascii'
            ]
        )
        
        # Load location data
        self._load_location_data()
    
    async def __aenter__(self):
        """
        Async context manager entry
        
        Initializes all async services if not already done.
        
        Returns:
            Self for use in async with statement
        """
        if not self._is_initialized:
            await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit
        
        Properly closes all async sessions and cleans up resources. We don't want to
        leave any HTTP sessions open.
        
        Args:
            exc_type: Exception type if context exited due to exception
            exc_val: Exception value if context exited due to exception  
            exc_tb: Exception traceback if context exited due to exception
        """
        if self.weather_service and hasattr(self.weather_service, 'session'):
            if self.weather_service.session and not self.weather_service.session.closed:
                await self.weather_service.session.close()
        logger.info("Indra chatbot session cleanup completed")
    
    def _load_location_data(self):
        """
        Load location data from JSON file with graceful fallbacks.
        
        Attempts to load location data from the training directory. If the file
        doesn't exist, creates a sensible fallback structure so the bot doesn't
        completely lose its mind.
        
        The location data helps with location recognition and mapping, though the
        main source of truth remains the config files.
        """
        try:
            location_file = os.path.join(Config.TRAINING_DIR, 'england_locations.json')
            if os.path.exists(location_file):
                with open(location_file, 'r') as f:
                    self.location_data = json.load(f)
                logger.info("Location data loaded successfully")
            else:
                logger.warning("Location data file not found", file=location_file)
                self.location_data = {'valid_locations': [], 'location_mapping': {}}
        except Exception as e:
            logger.error("Failed to load location data", error=str(e))
            self.location_data = {'valid_locations': [], 'location_mapping': {}}
    
    async def initialize(self):
        """
        Initialize the bot and all its async services.
        
        We do this asynchronously because some services might need to validate API
        keys or establish connections, and we don't want to block the main thread
        while that happens.
        
        Raises:
            ChatterPyException: If critical services fail to initialize
        """
        try:
            logger.info("Initializing Indra chatbot... (this might take a moment)")
            
            # Initialize weather service - the crown jewel of English conversation
            self.weather_service = AsyncWeatherService()
            
            # Initialize news service - because weather isn't everything
            self.news_service = NewsService()
            
            # Initialize analytics service - for those who like their travel data crunched
            self.analytics = TravelAnalytics()
            
            # Initialize location validator - handles fuzzy matching like a champion
            self.validator = EnglandLocationValidator()
            
            # Train the chatbot if needed - teaching it to be more conversational
            await self._train_if_needed()
            
            self._is_initialized = True
            logger.info("Indra chatbot initialized successfully - ready for conversation!")
            
        except Exception as e:
            logger.error("Failed to initialize Indra chatbot", error=str(e))
            raise ChatterPyException(f"Initialization failed: {str(e)}")
    
    async def _train_if_needed(self):
        """Train the chatbot with conversation data."""
        try:
            # Check if training is needed (database empty or old)
            if self._needs_training():
                logger.info("Training ChatterPy with conversation data...")
                await self._train_chatbot()
                logger.info("ChatterPy training completed")
            else:
                logger.info("ChatterPy already trained, skipping training")
                
        except Exception as e:
            logger.error("ChatterBot training failed", error=str(e))
            raise ChatterPyTrainingException(str(e), Config.CHATTERPY_TRAINING_DATA_PATH)
    
    def _needs_training(self) -> bool:
        """Check if the chatbot needs training."""
        try:
            # Simple check: if database is empty or very small
            statement_count = len(list(self.chatbot.storage.filter()))
            return statement_count < 10
        except Exception:
            return True
    
    async def _train_chatbot(self):
        """Train the chatbot with conversation data."""
        try:
            # Train with custom conversation data
            if os.path.exists(Config.CHATTERPY_TRAINING_DATA_PATH):
                trainer = ListTrainer(self.chatbot)
                
                # Load and parse training data
                training_data = _parse_training_data()
                if training_data:
                    trainer.train(training_data)
                    logger.info("Trained with custom conversation data", 
                               conversations=len(training_data))
            
            # Train with English corpus
            corpus_trainer = ChatterBotCorpusTrainer(self.chatbot)
            corpus_trainer.train("chatterbot.corpus.english.greetings")
            corpus_trainer.train("chatterbot.corpus.english.conversations")
            
            logger.info("ChatterBot training completed successfully")
            
        except Exception as e:
            logger.error("ChatterBot training failed", error=str(e))
            raise ChatterPyTrainingException(str(e), Config.CHATTERPY_TRAINING_DATA_PATH)

    @performance_monitor(log_performance=True)
    @async_retry(max_retries=2, delay=0.5)
    async def get_response_async(self, user_input: str) -> str:
        """
        Get response with fuzzy matching, context awareness, and concurrent API calls.
        
        Args:
            user_input: User's message
            
        Returns:
            bot response based on context and classification
        """
        try:
            # Classify intent
            classification_result = self.intent_classifier.classify_intent(
                user_input, self.conversation_context
            )
            
            intent = classification_result.intent
            location = classification_result.location
            confidence = classification_result.confidence
            
            logger.info("Intent classification",
                       intent=intent.value, 
                       location=location, 
                       confidence=confidence,
                       patterns_matched=len(classification_result.patterns_matched),
                       fuzzy_matches=len(classification_result.fuzzy_matches))
            
            # Handle low confidence with clarification
            if confidence < 0.6:
                response = await self._handle_low_confidence(classification_result)
            else:
                # Route to appropriate handler
                response = await self._route_intent(intent, location, user_input)
            
            # Update conversation context
            self.conversation_context.add_turn(
                user_input, intent, location, confidence, response
            )
            
            return response
                
        except Exception as e:
            logger.error("Error generating response", error=str(e), input=user_input)
            return await self._handle_error(e)
    
    async def _handle_low_confidence(self, classification_result: IntentClassificationResult) -> str:
        """Handle low confidence classifications with clarification."""
        if classification_result.fuzzy_matches:
            best_match = classification_result.fuzzy_matches[0]
            return f"Did you mean {best_match[0]}? I can provide weather, activities, or information about {best_match[0]}. Or please rephrase your question."
        
        if self.conversation_context.last_location:
            return f"I'm not sure what you'd like to know. Are you asking about weather, activities, or news for {self.conversation_context.last_location}?"
        
        return "I'm not sure I understand. Could you ask about weather forecasts, activities, or information for England destinations like Cambridge, Oxford, or the Cotswolds?"
    
    async def _route_intent(self, intent: IntentType, location: Optional[str],
                            user_input: str) -> str:
        """Route intent to appropriate handler with concurrent operations where beneficial."""
        
        if intent == IntentType.GREETING:
            return await self._handle_greeting()
        elif intent == IntentType.FAREWELL:
            return await self._handle_farewell()
        elif intent == IntentType.WEATHER_CURRENT and location:
            return await self._handle_current_weather(location)
        elif intent == IntentType.WEATHER_FORECAST and location:
            return await self._handle_weather_forecast(location)
        elif intent == IntentType.WEATHER_COMPARISON:
            return await self._handle_weather_comparison(user_input)
        elif intent == IntentType.ACTIVITIES and location:
            return await self._handle_activities_async(location)
        elif intent == IntentType.NEWS and location:
            return await self._handle_news_enhanced(location)
        elif intent == IntentType.LOCATION_INFO and location:
            # Use concurrent API calls for comprehensive response
            return await self._handle_location_info_concurrent(location)
        else:
            # Fall back to ChatterBot for general conversation
            return await self._get_chatterbot_response(user_input)
    
    async def _handle_greeting(self) -> str:
        """Handle greeting messages with context awareness."""
        if self.conversation_context.turn_count > 0:
            greetings = [
                "Hello again! How can I help you explore England today?",
                "Hi there! Ready for more England travel insights?",
                "Good to see you back! What destination interests you now?"
            ]
        else:
            greetings = [
                "Hello! I'm Indra, your AI travel assistant for exploring England. I can provide weather forecasts, activity recommendations, and local news for England's top destinations. I understand location names even with typos! What would you like to know?",
                "Hi there! I'm here to help you explore England. I can check weather, suggest activities, and share news for places like Cambridge, Oxford, the Cotswolds, and more. I'm also great at understanding what you mean even if you misspell a location! Where would you like to explore?",
                "Welcome! I'm Indra, specializing in England travel with smart conversation features. Whether you're planning to visit historic sites, university towns, or natural areas, I can help with weather forecasts, activity suggestions, and keep track of our conversation context. What destination interests you?"
            ]
        
        import random
        return random.choice(greetings)
    
    async def _handle_farewell(self) -> str:
        """Handle farewell messages with conversation summary."""
        farewells = [
            f"Goodbye! Thank you for {self.conversation_context.turn_count} great questions about England travel. I'm here whenever you need more weather updates or travel advice.",
            "Safe travels! I hope you have an amazing time exploring England's beautiful destinations. Our conversation context will be ready when you return!",
            "Farewell! It's been wonderful helping you plan your England adventures. Remember, I'm always here for travel assistance!"
        ]
        
        import random
        return random.choice(farewells)
    
    async def _handle_current_weather(self, location: str) -> str:
        """Handle current weather requests with async weather service."""
        try:
            async with self.weather_service as weather:
                weather_data = await weather.get_current_weather_async(location)
            
            current = weather_data['current']
            temp = current['temperature']
            desc = current['description']
            feels_like = current['feels_like']
            humidity = current['humidity']
            wind_speed = current['wind_speed']
            
            # Detailed response with activity recommendations
            response = f"Current weather in {location}: {temp}°C with {desc}. "
            response += f"Feels like {feels_like}°C, humidity {humidity}%, wind {wind_speed} m/s."
            
            # Activity suggestions based on weather
            if current['main'] == 'Clear' and temp > 15:
                response += " Perfect weather for outdoor activities and sightseeing!"
            elif current['main'] == 'Rain':
                response += " Great weather for indoor attractions and museums."
            elif temp < 10:
                response += " Dress warmly and consider indoor activities."
            elif current['main'] == 'Clouds' and temp > 12:
                response += " Good weather for walking tours and covered attractions."
            
            # Context-aware follow-up
            response += f" Would you like the 5-day forecast, activity recommendations, or local news for {location}?"
            
            return response
                
        except Exception as e:
            logger.error("Error getting current weather", location=location, error=str(e))
            return f"I'm having trouble getting the current weather for {location}. Please try again in a moment, or try a different location."
    
    async def _handle_weather_forecast(self, location: str) -> str:
        """Handle weather forecast requests with trend analysis."""
        try:
            async with self.weather_service as weather_svc:
                forecast_data = await weather_svc.get_forecast(location, days=5)
            
            response = f"Here's the 5-day forecast for {location}:\n\n"
            
            # Forecast with trend analysis
            temperatures = []
            conditions = []
            
            for day in forecast_data['days'][:5]:
                day_name = day['day_name']
                temp_min = day['temperature_min']
                temp_max = day['temperature_max']
                condition = day['dominant_condition']
                
                temperatures.append((temp_min + temp_max) / 2)
                conditions.append(condition)
                
                response += f"• {day_name}: {temp_min}-{temp_max}°C, {condition}\n"
            
            # Add trend analysis
            if len(temperatures) >= 3:
                trend = _analyze_temperature_trend(temperatures)
                response += f"\nTrend: {trend}\n"
            
            # Context-aware suggestions
            response += f"Would you like activity recommendations for any specific day, or shall I suggest the best days for outdoor activities in {location}?"
            
            return response
                
        except Exception as e:
            logger.error("Error getting weather forecast", location=location, error=str(e))
            return f"I'm having trouble getting the forecast for {location}. Please try again in a moment."

    async def _handle_weather_comparison(self, user_input: str) -> str:
        """Handle weather comparison requests parsing."""
        # Location extraction for comparisons
        locations = []
        
        # Use fuzzy matching to find all locations in the input
        for valid_location in Config.VALID_LOCATIONS:
            potential_matches = re.findall(rf'\b\w*{re.escape(valid_location.lower())}\w*\b', user_input.lower())
            for match in potential_matches:
                similarity = difflib.SequenceMatcher(None, match, valid_location.lower()).ratio()
                if similarity >= 0.8 and valid_location not in locations:
                    locations.append(valid_location)
        
        # Also check for partial matches
        for valid_location in Config.VALID_LOCATIONS:
            if valid_location.lower() in user_input.lower() and valid_location not in locations:
                locations.append(valid_location)
        
        if len(locations) < 2:
            return "I need at least two locations to compare weather. Please mention two England destinations like 'Compare weather between Oxford and Cambridge'."
        
        try:
            # Get weather data for each location concurrently
            tasks = [self._get_weather_data_async(location) for location in locations[:3]]
            results = await safe_gather(*tasks, return_exceptions=True)
            
            # Process results
            weather_data = {}
            for location, result in zip(locations[:3], results):
                if not isinstance(result, Exception):
                    weather_data[location] = result
            
            if not weather_data:
                return "I couldn't get weather data for comparison right now. Please try again."
            
            response = f"Weather comparison for {', '.join(weather_data.keys())}:\n\n"
            
            temperatures = {}
            for location, data in weather_data.items():
                temp = data['current']['temperature']
                desc = data['current']['description']
                temperatures[location] = temp
                response += f"• {location}: {temp}°C, {desc}\n"
            
            # Comparison analysis
            if len(temperatures) >= 2:
                warmest = max(temperatures, key=temperatures.get)
                coolest = min(temperatures, key=temperatures.get)
                temp_diff = temperatures[warmest] - temperatures[coolest]
                
                response += f"\n{warmest} is warmest at {temperatures[warmest]}°C"
                response += f"\n{coolest} is coolest at {temperatures[coolest]}°C"
                response += f"\nTemperature difference: {temp_diff:.1f}°C"
                
                # Activity recommendations based on comparison
                if temp_diff > 5:
                    response += f"\nFor outdoor activities, {warmest} would be more comfortable today."
            
            return response
                
        except Exception as e:
            logger.error("Error comparing weather", locations=locations, error=str(e))
            return "I'm having trouble comparing weather data right now. Please try again."
    
    @async_retry(max_retries=3, delay=1.0)
    async def _get_weather_data_async(self, location: str) -> Dict[str, Any]:
        """Asynchronously get weather data for a location."""
        async with self.weather_service as weather:
            return await weather.get_current_weather_async(location)
    
    @async_retry(max_retries=2, delay=0.5)
    async def _get_news_data_async(self, location: str) -> Dict[str, Any]:
        """Asynchronously get news data for a location."""
        try:
            return await self.news_service.get_location_news(location, [NewsCategory.TRAVEL, NewsCategory.EVENTS])
        except Exception as e:
            logger.error("Failed to get news data", location=location, error=str(e))
            return {'total_articles': 0, 'articles': {}}
    
    async def _handle_activities_async(self, location: str) -> str:
        """Handle activity recommendations with weather-based intelligence."""
        try:
            # Get current weather for recommendations
            async with self.weather_service as weather:
                weather_data = await weather.get_current_weather_async(location)
            
            current = weather_data['current']
            condition = current['main']
            temp = current['temperature']
            
            # Get location-specific information
            location_info = self.locations.get_location(location)
            if not location_info:
                return f"I don't have detailed activity information for {location} right now. Try asking about weather or other England destinations."
            
            response = f"Smart activity recommendations for {location} "
            response += f"(current conditions: {temp}°C, {current['description']}):\n\n"
            
            # Weather-aware activity categorization
            if condition == 'Rain' or temp < 8:
                # Indoor activities
                indoor_activities = [attr for attr in location_info.key_attractions 
                                   if any(keyword in attr.lower() for keyword in 
                                         ['museum', 'cathedral', 'college', 'gallery', 'theatre', 'market', 'shopping'])]
                if indoor_activities:
                    response += "🏛️ Perfect indoor activities for current weather:\n"
                    for activity in indoor_activities[:4]:
                        response += f"• {activity}\n"
                    response += "\n"
            elif condition == 'Clear' and temp > 15:
                # Outdoor activities
                outdoor_activities = [attr for attr in location_info.key_attractions 
                                    if any(keyword in attr.lower() for keyword in 
                                          ['walk', 'garden', 'park', 'bridge', 'castle', 'trail', 'outdoor'])]
                if outdoor_activities:
                    response += "☀️ Excellent outdoor activities for today's weather:\n"
                    for activity in outdoor_activities[:4]:
                        response += f"• {activity}\n"
                    response += "\n"
            
            # Always show top general attractions
            response += "🌟 Top attractions (weather-flexible):\n"
            for attraction in location_info.key_attractions[:3]:
                response += f"• {attraction}\n"
            
            # Follow-up based on context
            if self.conversation_context.turn_count > 0:
                response += f"\nBased on our conversation, would you like the 5-day forecast to plan ahead, or local news for {location}?"
            else:
                response += f"\nWould you like weather forecasts to plan your visit, or information about getting to {location}?"
            
            return response
                
        except Exception as e:
            logger.error("Error getting activities", location=location, error=str(e))
            return f"I'm having trouble getting activity recommendations for {location}. Please try again."
    
    async def _handle_news_enhanced(self, location: str) -> str:
        """Handle news requests with categorization and relevance scoring."""
        try:
            # Determine relevant news categories based on context
            categories = [NewsCategory.TRAVEL, NewsCategory.EVENTS, NewsCategory.LOCAL]
            
            # Add weather alerts if recent weather query
            if self.conversation_context.last_intent == IntentType.WEATHER_CURRENT:
                categories.append(NewsCategory.WEATHER_ALERTS)
            
            # Get news for the location
            news_data = await self.news_service.get_location_news(location, categories)
            
            if not news_data.get('total_articles'):
                return f"I couldn't find recent news for {location}. Would you like weather information or activity recommendations instead?"
            
            response = f"Latest news and updates for {location}:\n\n"
            
            # Present top articles by category
            articles_shown = 0
            for category, articles in news_data.get('articles', {}).items():
                if articles and articles_shown < 5:
                    response += f"**{category.title()}:**\n"
                    for article in articles[:2]:  # Top 2 per category
                        response += f"• {article['title']}\n"
                        response += f"  {article['source']} - {article['published_at'][:10]}\n"
                        articles_shown += 1
                    response += "\n"
            
            # Add metadata
            response += f"Found {news_data['total_articles']} relevant articles across {len(news_data['categories_searched'])} categories.\n"
            
            # Context-aware follow-up
            if self.conversation_context.turn_count > 0:
                response += f"\nWould you like more details on any specific topic, or shall I check the weather forecast for {location}?"
            else:
                response += f"\nI can also provide weather forecasts and activity recommendations for {location}."
            
            return response
            
        except Exception as e:
            logger.error("Error getting news", location=location, error=str(e))
            return f"I'm having trouble accessing news for {location} right now. Would you like weather information or activity recommendations instead?"
    
    async def _handle_location_info_concurrent(self, location: str) -> str:
        """Handle location information with concurrent API calls for comprehensive data."""
        location_info = self.locations.get_location(location)
        if not location_info:
            return f"I don't have detailed information about {location} right now. I can help with weather forecasts and activity recommendations for England's top destinations."
        
        try:
            # Concurrent API calls for weather and news
            tasks = [
                self._get_weather_data_async(location),
                self._get_news_data_async(location),
            ]
            
            # Execute concurrent calls with safe gathering
            results = await safe_gather(*tasks, return_exceptions=True)
            weather_result = results[0] if not isinstance(results[0], Exception) else None
            news_result = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else None
            
            # Build comprehensive response
            response = f"{location_info.name}\n"
            response += f"{location_info.description}\n\n"
            
            # Add current weather if available
            if weather_result:
                current = weather_result['current']
                response += f"Current Weather: {current['temperature']}°C, {current['description']}\n\n"
            
            # Add latest news if available
            if news_result and news_result.get('total_articles', 0) > 0:
                response += "Recent News:\n"
                articles_shown = 0
                for category, articles in news_result.get('articles', {}).items():
                    if articles and articles_shown < 2:
                        for article in articles[:1]:  # Top 1 per category
                            response += f"• {article['title']}\n"
                            articles_shown += 1
                response += "\n"
            
            response += f"Key attractions:\n"
            for attraction in location_info.key_attractions[:4]:
                response += f"• {attraction}\n"
            
            response += f"\nBest seasons: {', '.join(location_info.best_seasons)}\n"
            response += f"Recommended stay: {location_info.recommended_duration}\n"
            
            # Follow-up
            response += f"\nI can provide detailed weather forecasts, activity recommendations, or help you compare {location} with other destinations. What interests you most?"
            
            return response
            
        except Exception as e:
            logger.error("Error getting concurrent location info", location=location, error=str(e))
            # Fallback to basic info
            response = f"{location_info.name} - {location_info.description}\n\n"
            response += f"Key attractions:\n"
            for attraction in location_info.key_attractions[:3]:
                response += f"• {attraction}\n"
            response += f"\nWould you like current weather conditions or activity recommendations for {location}?"
            return response
    
    async def _get_chatterbot_response(self, user_input: str) -> str:
        """Get response from ChatterBot as fallback with error handling."""
        try:
            # Prepare input for ChatterBot
            chatbot_input = user_input.strip()
            
            # Add context if available
            if self.conversation_context.last_location:
                chatbot_input = f"{chatbot_input} (about {self.conversation_context.last_location})"
            
            # Get ChatterBot response with proper error handling
            chatbot_response = self.chatbot.get_response(chatbot_input)
            
            # Handle different response types
            if hasattr(chatbot_response, 'text'):
                response_text = str(chatbot_response.text)
            elif hasattr(chatbot_response, '__str__'):
                response_text = str(chatbot_response)
            else:
                response_text = "I'm not sure how to respond to that."
            
            # Clean and validate response
            response_text = response_text.strip()
            
            # Enhance weak responses
            if (not response_text or 
                len(response_text) < 10 or 
                response_text.lower() in ['i dont know', 'i do not know', 'unknown'] or
                "I don't understand" in response_text):
                
                response_text = self._generate_contextual_fallback(user_input)
            
            return response_text
            
        except Exception as e:
            logger.error("ChatterBot response failed", error=str(e), input=user_input)
            return self._generate_contextual_fallback(user_input)
    
    def _generate_contextual_fallback(self, user_input: str) -> str:
        """Generate contextual fallback response."""
        input_lower = user_input.lower().strip()
        
        # Handle common conversational responses
        if input_lower in ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay']:
            if self.conversation_context.last_location:
                return f"Great! What would you like to know about {self.conversation_context.last_location}? I can provide weather forecasts, activity recommendations, or local information."
            else:
                return "Great! What England destination would you like to explore? I can help with weather, activities, and travel information for places like Cambridge, Oxford, or the Cotswolds."
        
        elif input_lower in ['no', 'nope', 'not really', 'nothing']:
            return "No problem! Feel free to ask about weather, activities, or information for any England destination whenever you're ready."
        
        elif 'help' in input_lower:
            return "I'm Indra, your England travel assistant! I can help with weather forecasts, activity recommendations, local news, and travel information for destinations like Cambridge, Oxford, Bath, York, and more. Just ask!"
        
        else:
            # General fallback with context
            if self.conversation_context.last_location:
                return f"I'm not sure about that. Would you like weather information, activities, or news for {self.conversation_context.last_location}? Or ask about a different England destination."
            else:
                return "I specialize in England travel information. Try asking about weather, activities, or attractions for destinations like Cambridge, Oxford, Bath, or York!"
    
    async def _handle_error(self, error: Exception) -> str:
        """Handle errors gracefully with context awareness."""
        if isinstance(error, IndraException):
            return f"I encountered an issue: {error.message}. Please try rephrasing your question or ask about a different England destination."
        else:
            context_help = ""
            if self.conversation_context.last_location:
                context_help = f" I was last helping with {self.conversation_context.last_location} - would you like to continue with that?"
            
            return f"I'm having some technical difficulties.{context_help} Please try asking about weather, activities, or information for England destinations."
    
    # Public API methods for Flask app
    async def get_weather_forecast(self, location: str) -> Dict[str, Any]:
        """Get weather forecast (used by Flask app)."""
        if not self.weather_service:
            raise IndraException("Weather service not initialized")
        
        async with self.weather_service as weather_svc:
            return await weather_svc.get_forecast(location, days=5)
    
    async def get_activity_recommendations(self, location: str) -> Dict[str, Any]:
        """Get activity recommendations with weather context (used by Flask app)."""
        location_info = self.locations.get_location(location)
        if not location_info:
            raise LocationNotFoundException(location)
        
        # Get current weather for context
        try:
            async with self.weather_service as weather:
                weather_data = await weather.get_current_weather_async(location)
                weather_condition = weather_data['current']['main']
                temperature = weather_data['current']['temperature']
        except Exception:
            weather_condition = 'Unknown'
            temperature = None
        
        return {
            'location': location,
            'weather_condition': weather_condition,
            'temperature': temperature,
            'key_attractions': location_info.key_attractions,
            'weather_appropriate_activities': self.locations.get_weather_appropriate_activities(location, weather_condition),
            'best_seasons': location_info.best_seasons,
            'recommended_duration': location_info.recommended_duration,
            'travel_tips': location_info.travel_tips,
            'confidence_score': 0.95,  # High confidence for structured data
            'context_aware': True
        }
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics for monitoring."""
        return {
            'turn_count': self.conversation_context.turn_count,
            'session_duration': (datetime.now() - self.conversation_context.session_start).total_seconds(),
            'last_location': self.conversation_context.last_location,
            'last_intent': self.conversation_context.last_intent.value if self.conversation_context.last_intent else None,
            'conversation_history_length': len(self.conversation_context.conversation_history),
            'user_preferences': self.conversation_context.user_preferences
        }