"""
Author: Cameron Murphy (Student ID: 1049678, GitHub: 0x1049678II)
Date: June 10th 2025

Validation system for England locations and travel data.

This validation system is the gatekeeper for location recognition in the Indra chatbot.
It handles the reality of user input - typos, abbreviations, alternate spellings etc.

The validator uses fuzzy string matching with difflib to handle typos gracefully,
provides suggestions when exact matches fail, and validates coordinates
to ensure they actually fall within England's borders.

Features:
    - Fuzzy string matching with configurable thresholds
    - Location extraction from natural language
    - Coordinate validation for England boundaries
    - Suggestion system for near-misses
    - Travel query analysis and intent classification
    - Common alias and misspelling recognition

The system is designed to be helpful rather than pedantic - it tries to understand
what users mean even when they don't spell it perfectly.
"""

import re
import difflib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import structlog

from config.locations import EnglandLocations
from config.settings import Config

logger = structlog.get_logger()


class ValidationType(Enum):
    """Types of validation available."""
    LOCATION_NAME = "location_name"
    COORDINATES = "coordinates"
    TRAVEL_DATES = "travel_dates"
    ACTIVITY_TYPE = "activity_type"
    WEATHER_PARAMS = "weather_params"


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    suggestions: List[str]
    corrected_value: Optional[Any] = None
    confidence: float = 0.0


@dataclass
class LocationValidationResult:
    """Location validation result."""
    original_input: str
    is_valid: bool
    validated_location: Optional[str]
    fuzzy_matches: List[Tuple[str, float]]
    validation_issues: List[ValidationResult]
    coordinates: Optional[Dict[str, float]]
    location_info: Optional[Dict[str, Any]]


class EnglandLocationValidator:
    """
    The location validation specialist - handles user location input.
    
    This validator deals with typos (cambrige -> Cambridge) and abbreviations (camb -> Cambridge).
    
    The validator uses a multi-stage approach:
    1. Exact matching (for the lucky cases)
    2. Alias checking (for common abbreviations)
    3. Fuzzy matching (for the typos)
    4. Suggestions (for the hopeless cases)
    
    It also validates coordinates to ensure they fall within England's borders.
    
    Features:
        - Multi-stage location matching with fallbacks
        - Fuzzy string matching with configurable thresholds
        - Location alias and common misspelling recognition
        - Coordinate validation for England boundaries
        - Travel query analysis and intent classification
        - Suggestion system
        
    The validator is designed to be forgiving - it tries to understand what
    users mean rather than rejecting input for minor spelling mistakes.
    
    Attributes:
        locations: England locations configuration and data
        valid_locations: List of supported location names
        fuzzy_threshold: Minimum similarity score for fuzzy matches
        high_confidence_threshold: Threshold for high-confidence matches
        location_aliases: Dictionary of common aliases and misspellings
        compiled_patterns: Pre-compiled regex patterns for extraction
    """
    
    def __init__(self):
        """
        Initialize the location validator with fuzzy matching capabilities.
        
        Sets up location data, fuzzy matching thresholds, and pre-compiles
        regex patterns for efficient location extraction. Also builds the
        alias dictionary for common misspellings and abbreviations.
        """
        self.locations = EnglandLocations()
        self.valid_locations = Config.VALID_LOCATIONS
        self.fuzzy_threshold = 0.6  # Minimum similarity for fuzzy matching
        self.high_confidence_threshold = 0.8
        
        # Pre-compile common location patterns
        self._compile_location_patterns()
        
        # Location aliases and common misspellings
        self.location_aliases = self._build_location_aliases()
        
        logger.info("England location validator initialized", 
                   valid_locations=len(self.valid_locations))
    
    def _compile_location_patterns(self):
        """Pre-compile regex patterns for location extraction."""
        # Common location prefixes/suffixes
        self.location_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Capitalized words
            r'\b(the\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # "The" + location
            r'\b([a-z]+(?:\s+[a-z]+)*)\b'  # Any word combination
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) 
                                for pattern in self.location_patterns]
    
    def _build_location_aliases(self) -> Dict[str, str]:
        """Build dictionary of location aliases and common misspellings."""
        aliases = {
            # Cambridge aliases and misspellings
            'camb': 'Cambridge',
            'cambrige': 'Cambridge',
            'cambridg': 'Cambridge',
            'cambdridge': 'Cambridge',
            
            # Oxford aliases and misspellings
            'ox': 'Oxford',
            'oxfrd': 'Oxford',
            'oxfor': 'Oxford',
            'oxfrod': 'Oxford',
            
            # Bristol aliases and misspellings
            'bristoll': 'Bristol',
            'bristl': 'Bristol',
            'bris': 'Bristol',
            
            # Norwich aliases and misspellings
            'norwich': 'Norwich',
            'norwhich': 'Norwich',
            'norwitsch': 'Norwich',
            'norwitch': 'Norwich',
            
            # Birmingham aliases and misspellings
            'bham': 'Birmingham',
            'birm': 'Birmingham',
            'birmingham': 'Birmingham',
            'birmingam': 'Birmingham',
            'brimmingham': 'Birmingham',
            
            # Cumbria aliases and misspellings
            'cumbria': 'Cumbria',
            'cumbira': 'Cumbria',
            'lake district': 'Cumbria',
            'the lake district': 'Cumbria',
            'lakes': 'Cumbria',
            'lakeland': 'Cumbria',
            
            # The Cotswolds aliases and misspellings
            'the cotswolds': 'The Cotswolds',
            'cotswolds': 'The Cotswolds',
            'cotswalds': 'The Cotswolds',
            'cotswold': 'The Cotswolds',
            'the cotswold': 'The Cotswolds',
            
            # Stonehenge aliases and misspellings
            'stonehenge': 'Stonehenge',
            'stone henge': 'Stonehenge',
            'stonehege': 'Stonehenge',
            'stoenhenge': 'Stonehenge',
            
            # Corfe Castle aliases and misspellings
            'corfe castle': 'Corfe Castle',
            'corfe': 'Corfe Castle',
            'corf castle': 'Corfe Castle',
            'corffe castle': 'Corfe Castle',
            
            # Watergate Bay aliases and misspellings
            'watergate bay': 'Watergate Bay',
            'watergate': 'Watergate Bay',
            'water gate bay': 'Watergate Bay',
            'watergatebaay': 'Watergate Bay'
        }
        
        return aliases
    
    def validate_location(self, user_input: str) -> LocationValidationResult:
        """
        Location validation with fuzzy matching and suggestions.
        
        Args:
            user_input: Raw user input containing location
            
        Returns:
            Complete validation result with suggestions
        """
        original_input = user_input.strip()
        validation_issues = []
        
        # Step 1: Direct exact match
        exact_match = self._check_exact_match(original_input)
        if exact_match:
            return self._create_success_result(original_input, exact_match)
        
        # Step 2: Check aliases and common misspellings
        alias_match = self._check_aliases(original_input)
        if alias_match:
            issue = ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message=f"Location recognized from common variation",
                suggestions=[],
                corrected_value=alias_match,
                confidence=0.9
            )
            validation_issues.append(issue)
            return self._create_success_result(original_input, alias_match, validation_issues)
        
        # Step 3: Extract potential locations from text
        extracted_locations = self._extract_location_candidates(original_input)
        
        # Step 4: Fuzzy matching on extracted candidates
        fuzzy_matches = []
        best_match = None
        best_confidence = 0.0
        
        for candidate in extracted_locations:
            matches = self._perform_fuzzy_matching(candidate)
            fuzzy_matches.extend(matches)
            
            if matches and matches[0][1] > best_confidence:
                best_match = matches[0][0]
                best_confidence = matches[0][1]
        
        # Remove duplicates and sort
        fuzzy_matches = list(set(fuzzy_matches))
        fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
        
        # Step 5: Determine result based on confidence
        if best_confidence >= self.high_confidence_threshold:
            issue = ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message=f"Location matched with {best_confidence:.0%} confidence",
                suggestions=[match[0] for match in fuzzy_matches[:3]],
                corrected_value=best_match,
                confidence=best_confidence
            )
            validation_issues.append(issue)
            return self._create_success_result(original_input, best_match, validation_issues)
        
        elif best_confidence >= self.fuzzy_threshold:
            issue = ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"Possible location match with {best_confidence:.0%} confidence",
                suggestions=[match[0] for match in fuzzy_matches[:5]],
                corrected_value=best_match,
                confidence=best_confidence
            )
            validation_issues.append(issue)
        else:
            issue = ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="No matching England location found",
                suggestions=self._get_suggestion_fallbacks(),
                confidence=0.0
            )
            validation_issues.append(issue)
        
        return LocationValidationResult(
            original_input=original_input,
            is_valid=False,
            validated_location=None,
            fuzzy_matches=fuzzy_matches[:10],  # Top 10 matches
            validation_issues=validation_issues,
            coordinates=None,
            location_info=None
        )
    
    def _check_exact_match(self, input_text: str) -> Optional[str]:
        """Check for exact match in valid locations."""
        # Case-insensitive exact match
        for location in self.valid_locations:
            if input_text.lower() == location.lower():
                return location
        return None
    
    def _check_aliases(self, input_text: str) -> Optional[str]:
        """Check location aliases and common misspellings."""
        input_lower = input_text.lower().strip()
        return self.location_aliases.get(input_lower)
    
    def _extract_location_candidates(self, text: str) -> List[str]:
        """Extract potential location names from text."""
        candidates = set()
        
        # Simple word extraction for known locations
        words = text.split()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) > 2:
                candidates.add(clean_word)
        
        # Check for exact location name matches in text
        for location in self.valid_locations:
            if location.lower() in text.lower():
                candidates.add(location)
        
        # Apply regex patterns
        for pattern in self.compiled_patterns:
            matches = pattern.findall(text)
            candidates.update([match.strip() for match in matches if len(match.strip()) > 2])
        
        # Add the full text as a candidate
        candidates.add(text.strip())
        
        # Split on common separators and add parts
        separators = [',', '/', '|', '&', 'and', 'or', 'in', 'near', 'around']
        for sep in separators:
            parts = text.split(sep)
            for part in parts:
                clean_part = part.strip()
                if len(clean_part) > 2:
                    candidates.add(clean_part)
        
        return list(candidates)
    
    def _perform_fuzzy_matching(self, candidate: str) -> List[Tuple[str, float]]:
        """Perform fuzzy matching against valid locations."""
        matches = []
        
        for location in self.valid_locations:
            # Calculate similarity ratio
            ratio = difflib.SequenceMatcher(None, candidate.lower(), location.lower()).ratio()
            
            # Return all matches above 0.3 for testing and suggestion purposes
            # The validation threshold is applied separately
            if ratio >= 0.3:
                matches.append((location, ratio))
        
        # Sort by similarity score
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def _get_suggestion_fallbacks(self) -> List[str]:
        """Get fallback suggestions when no good matches found."""
        # Return most popular/recognizable locations from our supported set
        popular_locations = [
            'Cambridge', 'Oxford', 'Bristol', 'Birmingham',
            'The Cotswolds', 'Norwich', 'Cumbria', 'Stonehenge'
        ]
        return [loc for loc in popular_locations if loc in self.valid_locations]
    
    def _create_success_result(self, original_input: str, validated_location: str,
                              issues: List[ValidationResult] = None) -> LocationValidationResult:
        """Create a successful validation result."""
        if issues is None:
            issues = []
        
        # Get location info and coordinates
        location_info = self.locations.get_location(validated_location)
        coordinates = self.locations.get_coordinates(validated_location)
        
        return LocationValidationResult(
            original_input=original_input,
            is_valid=True,
            validated_location=validated_location,
            fuzzy_matches=[(validated_location, 1.0)],
            validation_issues=issues,
            coordinates=coordinates,
            location_info=location_info.__dict__ if location_info else None
        )
    
    def validate_coordinates(self, lat: float, lon: float) -> ValidationResult:
        """
        Validate coordinates for England bounds.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Validation result for coordinates
        """
        # England approximate bounds
        england_bounds = {
            'lat_min': 49.9,    # Southern tip (Scilly Isles)
            'lat_max': 55.8,    # Northern border with Scotland
            'lon_min': -6.4,    # Western tip (Cornwall)
            'lon_max': 1.8      # Eastern tip (Norfolk)
        }
        
        issues = []
        
        # Check latitude
        if not (england_bounds['lat_min'] <= lat <= england_bounds['lat_max']):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Latitude {lat} is outside England bounds ({england_bounds['lat_min']}-{england_bounds['lat_max']})",
                suggestions=["Check latitude value", "Ensure coordinates are for England"]
            )
        
        # Check longitude
        if not (england_bounds['lon_min'] <= lon <= england_bounds['lon_max']):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Longitude {lon} is outside England bounds ({england_bounds['lon_min']}-{england_bounds['lon_max']})",
                suggestions=["Check longitude value", "Ensure coordinates are for England"]
            )
        
        # Check if coordinates are too precise (suspicious)
        if len(str(lat).split('.')[-1]) > 6 or len(str(lon).split('.')[-1]) > 6:
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message="Coordinates are very precise - ensure accuracy",
                suggestions=["Verify coordinate precision is needed"],
                confidence=0.8
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Coordinates are valid for England",
            suggestions=[],
            confidence=1.0
        )
    
    def suggest_similar_locations(self, location: str, max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """
        Get similar location suggestions based on fuzzy matching.
        
        Args:
            location: Input location
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of (location, similarity_score) tuples
        """
        suggestions = []
        
        for valid_location in self.valid_locations:
            similarity = difflib.SequenceMatcher(None, location.lower(), valid_location.lower()).ratio()
            if similarity > 0.3:  # Lower threshold for suggestions
                suggestions.append((valid_location, similarity))
        
        # Sort by similarity and return top suggestions
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:max_suggestions]
    
    def validate_travel_query(self, query: str) -> Dict[str, Any]:
        """
        Validate and extract information from a complete travel query.
        
        Args:
            query: Complete user query
            
        Returns:
            Validation summary with extracted information
        """
        # Extract location
        location_result = self.validate_location(query)
        
        # Extract other travel-related information
        activity_keywords = self._extract_activity_keywords(query)
        time_indicators = self._extract_time_indicators(query)
        weather_keywords = self._extract_weather_keywords(query)
        
        # Determine query type
        query_type = self._determine_query_type(query, activity_keywords, weather_keywords)
        
        return {
            'location_validation': location_result,
            'query_type': query_type,
            'activity_keywords': activity_keywords,
            'time_indicators': time_indicators,
            'weather_keywords': weather_keywords,
            'validation_summary': {
                'location_valid': location_result.is_valid,
                'has_activities': len(activity_keywords) > 0,
                'has_time_context': len(time_indicators) > 0,
                'has_weather_context': len(weather_keywords) > 0
            }
        }
    
    def _extract_activity_keywords(self, query: str) -> List[str]:
        """Extract activity-related keywords from query."""
        activity_keywords = [
            'visit', 'see', 'tour', 'explore', 'walk', 'hiking', 'cycling',
            'museum', 'gallery', 'castle', 'church', 'park', 'garden',
            'shopping', 'restaurant', 'pub', 'theater', 'attraction',
            'sightseeing', 'photography', 'punting', 'boat', 'festival'
        ]
        
        found_keywords = []
        query_lower = query.lower()
        
        for keyword in activity_keywords:
            if keyword in query_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _extract_time_indicators(self, query: str) -> List[str]:
        """Extract time-related indicators from query."""
        time_patterns = [
            r'\b(today|tomorrow|yesterday)\b',
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(spring|summer|autumn|fall|winter)\b',
            r'\b(this\s+week|next\s+week|this\s+month|next\s+month)\b',
            r'\b(morning|afternoon|evening|night)\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'  # Date patterns
        ]
        
        found_indicators = []
        query_lower = query.lower()
        
        for pattern in time_patterns:
            matches = re.findall(pattern, query_lower)
            found_indicators.extend(matches)
        
        return found_indicators
    
    def _extract_weather_keywords(self, query: str) -> List[str]:
        """Extract weather-related keywords from query."""
        weather_keywords = [
            'weather', 'temperature', 'rain', 'sunny', 'cloudy', 'wind',
            'storm', 'snow', 'forecast', 'cold', 'warm', 'hot', 'cool',
            'humid', 'dry', 'wet', 'clear', 'overcast', 'drizzle'
        ]
        
        found_keywords = []
        query_lower = query.lower()
        
        for keyword in weather_keywords:
            if keyword in query_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _determine_query_type(self, query: str, activity_keywords: List[str], 
                             weather_keywords: List[str]) -> str:
        """Determine the primary type of the query."""
        query_lower = query.lower()
        
        # Weather query indicators
        if any(word in query_lower for word in ['weather', 'temperature', 'forecast', 'rain', 'sunny']):
            return 'weather'
        
        # News/events query indicators
        if any(word in query_lower for word in ['news', 'events', 'happening', 'festival', 'what\'s on']):
            return 'news'
        
        # Activity query indicators
        if activity_keywords or any(word in query_lower for word in ['do', 'activities', 'attractions', 'visit']):
            return 'activities'
        
        # General information query
        if any(word in query_lower for word in ['about', 'information', 'tell me', 'describe']):
            return 'location_info'
        
        # Default to general
        return 'general'
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get statistics about the validation system."""
        return {
            'total_valid_locations': len(self.valid_locations),
            'location_aliases': len(self.location_aliases),
            'fuzzy_threshold': self.fuzzy_threshold,
            'high_confidence_threshold': self.high_confidence_threshold,
            'validation_patterns': len(self.compiled_patterns),
            'supported_location_types': [
                'Cities', 'Towns', 'Historic sites', 'Natural areas', 
                'Universities', 'Tourist destinations'
            ]
        }