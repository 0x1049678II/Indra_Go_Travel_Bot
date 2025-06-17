"""
Author: Cameron Murphy (Student ID: 1049678, GitHub: 0x1049678II)
Date: June 10th 2025

News service for England travel integration using NewsAPI.

This service fetches travel-relevant news for the English destinations because sometimes
the weather forecast just isn't enough - you need to know if there's a festival in
Cambridge or if the trains are running on time.

The service provides async methods for concurrent news fetching across multiple
categories, relevance scoring to filter out articles about celebrity
gossip when you're looking for travel information, and smart caching to avoid
hitting NewsAPI rate limits while still providing fresh information.
#NOTE - The news API works just fine - however it pulls in the strangest information
#NOTE - but it's so funny I thought I'd leave it.

Features:
    - Concurrent category fetching
    - Relevance scoring to filter noise from signal
    - Caching with different TTLs for different content types
    - Location-specific news filtering #TODO Need stricter filtering for higher accuracy
    - Event and festival detection
    - Travel advisory integration
"""

import asyncio
import hashlib
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple

import structlog
from cachetools import TTLCache
from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException

from chatbot.exceptions import (
    NewsAPIConnectionException,
    LocationNotFoundException,
    RateLimitExceededException
)
from config.locations import EnglandLocations
from config.settings import Config

logger = structlog.get_logger()


class NewsCategory(Enum):
    """News categories relevant to England travel."""
    TRAVEL = "travel"
    EVENTS = "events AND festivals"
    LOCAL = "local news"
    TOURISM = "tourism"
    CULTURE = "culture AND heritage"
    WEATHER_ALERTS = "weather warnings"
    TRANSPORT = "transport AND rail"
    GENERAL = "general"


class NewsRelevanceScorer:
    """Scores news articles for relevance to travel and location."""
    
    TRAVEL_KEYWORDS = [
        'travel', 'tourism', 'tourist', 'visitor', 'attraction', 'festival',
        'event', 'museum', 'gallery', 'castle', 'garden', 'heritage', 'historic',
        'sightseeing', 'explore', 'visit', 'destination', 'holiday', 'trip'
    ]
    
    NEGATIVE_KEYWORDS = [
        'crime', 'accident', 'injury', 'death', 'violence', 'politics',
        'brexit', 'covid', 'pandemic', 'lockdown', 'restriction'
    ]
    
    @staticmethod
    def calculate_relevance_score(article: Dict[str, Any], location: str) -> float:
        """
        Calculate relevance score for a news article.
        
        Args:
            article: News article dictionary from NewsAPI
            location: Target location for relevance
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        score = 0.0
        
        # Check title and description
        title = (article.get('title') or '').lower()
        description = (article.get('description') or '').lower()
        content = (article.get('content') or '').lower()
        combined_text = f"{title} {description} {content}"
        
        # Location relevance
        if location.lower() in combined_text:
            score += 0.3
        
        # Travel keyword relevance
        travel_matches = sum(1 for keyword in NewsRelevanceScorer.TRAVEL_KEYWORDS 
                           if keyword in combined_text)
        score += min(travel_matches * 0.1, 0.4)
        
        # Negative keyword penalty
        negative_matches = sum(1 for keyword in NewsRelevanceScorer.NEGATIVE_KEYWORDS 
                             if keyword in combined_text)
        score -= min(negative_matches * 0.15, 0.5)
        
        # Recency bonus (articles from last 7 days get bonus)
        published_at = article.get('publishedAt')
        if published_at:
            try:
                pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                days_old = (datetime.now(pub_date.tzinfo) - pub_date).days
                if days_old <= 7:
                    score += 0.2 * (1 - days_old / 7)
            except Exception:
                pass
        
        # Source quality bonus
        source = article.get('source', {}).get('name', '')
        quality_sources = ['BBC', 'The Guardian', 'The Telegraph', 'The Times', 
                          'Visit Britain', 'Time Out', 'Lonely Planet']
        if any(src in source for src in quality_sources):
            score += 0.1
        
        return max(0.0, min(1.0, score))


class NewsService:
    """
    News service for England travel - your window into what's happening locally(kind of).
    
    This service fetches and filters news relevant to England travel destinations,
    because sometimes you need to know more than just the weather. Is there a
    music festival in Bristol? Are the trains to Cambridge delayed again?
    This service finds out.
    
    The service uses NewsAPI to fetch articles and applies filtering
    to separate travel-relevant news from the general chaos of British media.
    It fetches news across multiple categories concurrently and caches results
    based on content type (breaking transport news needs faster refresh than
    cultural heritage articles).
    
    Features:
        - Concurrent multicategory news fetching
        - relevance scoring (attempts to filter out celebrity gossip)
        - Dynamic caching with category-specific TTLs
        - Location-specific filtering and event detection
        - Rate limiting to stay within NewsAPI quotas
        - Travel advisory and transport update integration
        
    The service gracefully handles API outages and provides sensible fallbacks
    because the news might be down but your travel planning shouldn't stop.
    
    Attributes:
        client: NewsAPI client for fetching articles
        locations: England locations configuration
        relevance_scorer: Scores articles for travel relevance
        news_caches: Category-specific TTL caches
        cache_ttls: TTL settings per news category
        
    Cache Strategy:
        - Weather alerts: 15 minutes (urgent)
        - Transport news: 15 minutes (time-sensitive)
        - Events: 30 minutes (moderately time-sensitive)
        - Local news: 30 minutes (moderately time-sensitive)
        - Travel/Tourism: 1 hour (stable)
        - Culture: 2 hours (very stable)
    """
    
    def __init__(self):
        """
        Initialize the news service with category-specific caching.
        
        Sets up NewsAPI client, relevance scoring, and multi-tier caching system.
        Different news categories get different cache TTLs because transport delays
        need faster updates than cultural heritage articles.
        """
        self.api_key = Config.NEWS_API_KEY if hasattr(Config, 'NEWS_API_KEY') else None
        if not self.api_key:
            logger.warning("News API key not configured")
            self.client = None
        else:
            self.client = NewsApiClient(api_key=self.api_key)
        
        self.locations = EnglandLocations()
        self.relevance_scorer = NewsRelevanceScorer()
        
        # Caching with different TTLs per category
        self.cache_ttls = {
            NewsCategory.TRAVEL: 3600,      # 1 hour
            NewsCategory.EVENTS: 1800,      # 30 minutes
            NewsCategory.LOCAL: 1800,       # 30 minutes
            NewsCategory.TOURISM: 3600,     # 1 hour
            NewsCategory.CULTURE: 7200,     # 2 hours
            NewsCategory.WEATHER_ALERTS: 900, # 15 minutes
            NewsCategory.TRANSPORT: 900,    # 15 minutes
            NewsCategory.GENERAL: 3600      # 1 hour
        }
        
        # Separate caches per category
        self.news_caches = {
            category: TTLCache(maxsize=50, ttl=ttl)
            for category, ttl in self.cache_ttls.items()
        }
        
        # Rate limiting
        self.api_calls_today = 0
        self.last_reset_date = datetime.now().date()
        self.rate_limit = 500  # NewsAPI free tier limit
        
        logger.info("News service initialized", 
                   api_configured=bool(self.api_key),
                   cache_categories=len(self.news_caches))
    
    def _check_rate_limit(self):
        """Check and update rate limiting."""
        current_date = datetime.now().date()
        
        # Reset daily counter
        if current_date > self.last_reset_date:
            self.api_calls_today = 0
            self.last_reset_date = current_date
        
        if self.api_calls_today >= self.rate_limit:
            raise RateLimitExceededException(
                'NewsAPI', 
                self.rate_limit,
                f"Resets at midnight UTC"
            )
        
        self.api_calls_today += 1
    
    def _get_cache_key(self, location: str, category: NewsCategory, 
                      query: Optional[str] = None) -> str:
        """Generate cache key for news data."""
        key_parts = [location, category.value]
        if query:
            key_parts.append(query)
        
        key_string = "_".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get_location_news(self, location: str, 
                               categories: Optional[List[NewsCategory]] = None) -> Dict[str, Any]:
        """
        Get news for a specific England location with intelligent filtering.
        
        Args:
            location: England destination name
            categories: Optional list of news categories to fetch
            
        Returns:
            Dictionary containing categorized and scored news articles
        """
        if not self.client:
            raise NewsAPIConnectionException(response_text="API key not configured")
        
        # Validate location
        if location not in Config.VALID_LOCATIONS:
            raise LocationNotFoundException(location, Config.VALID_LOCATIONS)
        
        # Default categories if not specified
        if not categories:
            categories = [NewsCategory.TRAVEL, NewsCategory.EVENTS, NewsCategory.LOCAL]
        
        # Check cache first
        cache_results = self._check_cache_multi(location, categories)
        if cache_results and all(cache_results.values()):
            logger.info("All news retrieved from cache", 
                       location=location, categories=[c.value for c in categories])
            return self._format_news_response(location, cache_results)
        
        # Fetch news concurrently for different categories
        tasks = []
        for category in categories:
            if category not in cache_results or not cache_results[category]:
                tasks.append(self._fetch_news_for_category(location, category))
        
        if tasks:
            # Execute concurrent fetches
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            fresh_results = {}
            for category, result in zip([c for c in categories if c not in cache_results], results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to fetch {category.value} news", 
                               location=location, error=str(result))
                    fresh_results[category] = []
                else:
                    fresh_results[category] = result
            
            # Merge with cached results
            all_results = {**cache_results, **fresh_results}
        else:
            all_results = cache_results
        
        return self._format_news_response(location, all_results)
    
    def _check_cache_multi(self, location: str, 
                          categories: List[NewsCategory]) -> Dict[NewsCategory, List[Dict]]:
        """Check cache for multiple categories."""
        results = {}
        for category in categories:
            cache_key = self._get_cache_key(location, category)
            cache = self.news_caches[category]
            if cache_key in cache:
                results[category] = cache[cache_key]
        return results
    
    async def _fetch_news_for_category(self, location: str, 
                                     category: NewsCategory) -> List[Dict[str, Any]]:
        """
        Fetch news for a specific category and location.
        
        Args:
            location: England destination
            category: News category
            
        Returns:
            List of relevant news articles
        """
        self._check_rate_limit()
        
        # Build search query
        location_info = self.locations.get_location(location)
        query_parts = [location]
        
        # Add region/county if available
        if location_info and hasattr(location_info, 'region'):
            query_parts.append(location_info.region)
        
        # Add category-specific terms
        query_parts.append(category.value)
        
        if category == NewsCategory.TRAVEL:
            query_parts.extend(['tourism', 'attractions', 'things to do'])
        elif category == NewsCategory.EVENTS:
            query_parts.extend(['festival', 'exhibition', 'show'])
        elif category == NewsCategory.WEATHER_ALERTS:
            query_parts.extend(['weather warning', 'flood', 'storm'])
        
        query = ' OR '.join(f'"{part}"' for part in query_parts)
        
        try:
            # Run in executor since newsapi-python is synchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.get_everything(
                    q=query,
                    language='en',
                    sort_by='relevancy',
                    from_param=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                    to=datetime.now().strftime('%Y-%m-%d'),
                    page_size=50
                )
            )
            
            if response['status'] == 'ok':
                articles = response.get('articles', [])
                
                # Score and filter articles
                scored_articles = []
                for article in articles:
                    score = self.relevance_scorer.calculate_relevance_score(article, location)
                    if score >= 0.3:  # Minimum relevance threshold
                        article['relevance_score'] = score
                        article['category'] = category.value
                        scored_articles.append(article)
                
                # Sort by relevance score
                scored_articles.sort(key=lambda x: x['relevance_score'], reverse=True)
                
                # Cache the results
                cache_key = self._get_cache_key(location, category)
                self.news_caches[category][cache_key] = scored_articles[:10]  # Top 10
                
                logger.info(f"Fetched {category.value} news", 
                           location=location, 
                           total_articles=len(articles),
                           relevant_articles=len(scored_articles))
                
                return scored_articles[:10]
            else:
                raise NewsAPIConnectionException("NewsAPI", None, 
                                               response.get('message', 'Unknown error'))
                
        except NewsAPIException as e:
            if 'rateLimited' in str(e):
                raise RateLimitExceededException('NewsAPI', self.rate_limit)
            raise NewsAPIConnectionException("NewsAPI", None, str(e))
        except Exception as e:
            logger.error(f"Error fetching {category.value} news", 
                        location=location, error=str(e))
            raise NewsAPIConnectionException("NewsAPI", None, str(e))
    
    def _format_news_response(self, location: str, 
                            categorized_news: Dict[NewsCategory, List[Dict]]) -> Dict[str, Any]:
        """Format news response with metadata and summaries."""
        total_articles = sum(len(articles) for articles in categorized_news.values())
        
        # Create category summaries
        category_summaries = {}
        formatted_articles = {}
        
        for category, articles in categorized_news.items():
            if articles:
                category_summaries[category.value] = {
                    'count': len(articles),
                    'top_score': articles[0]['relevance_score'] if articles else 0,
                    'average_score': sum(a['relevance_score'] for a in articles) / len(articles)
                }
                
                # Format articles for response
                formatted_articles[category.value] = [
                    self._format_article(article) for article in articles[:5]  # Top 5 per category
                ]
        
        return {
            'location': location,
            'timestamp': datetime.now().isoformat(),
            'total_articles': total_articles,
            'categories_searched': [cat.value for cat in categorized_news.keys()],
            'category_summaries': category_summaries,
            'articles': formatted_articles,
            'metadata': {
                'cache_status': 'mixed',  # Some cached, some fresh
                'api_calls_remaining': self.rate_limit - self.api_calls_today,
                'last_updated': datetime.now().isoformat()
            }
        }
    
    def _format_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single article for response."""
        return {
            'title': article.get('title', 'No title'),
            'description': article.get('description', '')[:200] + '...' 
                          if len(article.get('description', '')) > 200 else article.get('description', ''),
            'url': article.get('url'),
            'source': article.get('source', {}).get('name', 'Unknown'),
            'published_at': article.get('publishedAt'),
            'relevance_score': article.get('relevance_score', 0),
            'category': article.get('category', 'general'),
            'image_url': article.get('urlToImage')
        }
    
    async def get_travel_updates(self, locations: List[str]) -> Dict[str, Any]:
        """
        Get travel updates for multiple locations concurrently.
        
        Args:
            locations: List of England destinations
            
        Returns:
            Combined travel updates for all locations
        """
        # Validate locations
        valid_locations = [loc for loc in locations if loc in Config.VALID_LOCATIONS]
        if not valid_locations:
            raise LocationNotFoundException("Multiple", Config.VALID_LOCATIONS)
        
        # Fetch news for each location concurrently
        tasks = [
            self.get_location_news(location, [NewsCategory.TRAVEL, NewsCategory.TRANSPORT])
            for location in valid_locations
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        combined_updates = {
            'locations': {},
            'timestamp': datetime.now().isoformat(),
            'summary': {}
        }
        
        for location, result in zip(valid_locations, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to get travel updates for {location}", error=str(result))
                combined_updates['locations'][location] = {
                    'error': 'Failed to fetch updates',
                    'articles': {}
                }
            else:
                combined_updates['locations'][location] = result
        
        # Generate summary
        total_articles = sum(
            loc_data.get('total_articles', 0) 
            for loc_data in combined_updates['locations'].values() 
            if isinstance(loc_data, dict)
        )
        
        combined_updates['summary'] = {
            'total_locations': len(valid_locations),
            'successful_fetches': len([r for r in results if not isinstance(r, Exception)]),
            'total_articles': total_articles,
            'categories': ['travel', 'transport']
        }
        
        return combined_updates
    
    async def search_events(self, location: str, 
                          date_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """
        Search for events and festivals in a location.
        
        Args:
            location: England destination
            date_range: Optional tuple of (start_date, end_date)
            
        Returns:
            Event-focused news articles
        """
        # Default to next 30 days if no date range specified
        if not date_range:
            start_date = datetime.now()
            end_date = start_date + timedelta(days=30)
        else:
            start_date, end_date = date_range
        
        # Get event news
        event_news = await self.get_location_news(location, [NewsCategory.EVENTS])
        
        # Filter by date mentions if possible
        if event_news.get('articles', {}).get('events'):
            articles = event_news['articles']['events']
            
            # Attempt to extract dates from articles (basic implementation)
            for article in articles:
                # This is a simplified date extraction
                # In production, you'd use more sophisticated NLP
                article['estimated_date'] = self._extract_event_date(article)
        
        return event_news
    
    def _extract_event_date(self, article: Dict[str, Any]) -> Optional[str]:
        """Extract potential event date from article (simplified)."""
        import re
        
        text = f"{article.get('title', '')} {article.get('description', '')}"
        
        # Look for date patterns
        date_patterns = [
            r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})\b',
            r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b',
            r'\b(\d{1,2})-(\d{1,2})-(\d{4})\b'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics for monitoring."""
        stats = {
            'category_stats': {},
            'total_cached_items': 0,
            'api_calls_today': self.api_calls_today,
            'rate_limit': self.rate_limit,
            'last_reset_date': self.last_reset_date.isoformat()
        }
        
        for category, cache in self.news_caches.items():
            stats['category_stats'][category.value] = {
                'cached_items': len(cache),
                'max_size': cache.maxsize,
                'ttl_seconds': cache.ttl
            }
            stats['total_cached_items'] += len(cache)
        
        return stats