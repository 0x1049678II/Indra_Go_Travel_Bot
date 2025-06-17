"""
Author: Cameron Murphy (Student ID: 1049678, GitHub: 0x1049678II)
Date: June 14th 2025

Test news service functionality in full.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock
from datetime import datetime

from chatbot.services.news_service import NewsService, NewsCategory, NewsRelevanceScorer
from config.settings import Config
from chatbot.exceptions import LocationNotFoundException, NewsAPIConnectionException


class TestNewsRelevanceScorer:
    """Test news relevance scoring functionality."""
    
    def test_travel_keyword_scoring(self):
        """Test scoring based on travel keywords."""
        article = {
            'title': 'Best Tourist Attractions in Cambridge',
            'description': 'Explore historic colleges and museums',
            'content': 'Visit Cambridge for amazing sightseeing opportunities',
            'publishedAt': datetime.now().isoformat() + 'Z',
            'source': {'name': 'BBC'}
        }
        
        score = NewsRelevanceScorer.calculate_relevance_score(article, 'Cambridge')
        assert score > 0.5  # Should score well for travel content
    
    def test_location_relevance(self):
        """Test location-specific relevance."""
        article = {
            'title': 'Oxford University announces new research center',
            'description': 'Located in Oxford city center',
            'content': 'Oxford residents excited about new facility',
            'publishedAt': datetime.now().isoformat() + 'Z',
            'source': {'name': 'The Times'}
        }
        
        score = NewsRelevanceScorer.calculate_relevance_score(article, 'Oxford')
        assert score >= 0.3  # Location match bonus
    
    def test_negative_keyword_penalty(self):
        """Test penalty for negative keywords."""
        article = {
            'title': 'Crime incident in Cambridge',
            'description': 'Police investigate violent incident',
            'content': 'Accident causes major disruption',
            'publishedAt': datetime.now().isoformat() + 'Z',
            'source': {'name': 'Local News'}
        }
        
        score = NewsRelevanceScorer.calculate_relevance_score(article, 'Cambridge')
        assert score < 0.3  # Should have low score due to negative keywords
    
    def test_recency_bonus(self):
        """Test recency bonus for recent articles."""
        recent_article = {
            'title': 'Travel guide to Cambridge',
            'description': 'Tourism information',
            'content': 'Visit Cambridge attractions',
            'publishedAt': datetime.now().isoformat() + 'Z',
            'source': {'name': 'Travel Guide'}
        }
        
        old_article = {
            'title': 'Travel guide to Cambridge',
            'description': 'Tourism information',
            'content': 'Visit Cambridge attractions',
            'publishedAt': '2024-01-01T00:00:00Z',
            'source': {'name': 'Travel Guide'}
        }
        
        recent_score = NewsRelevanceScorer.calculate_relevance_score(recent_article, 'Cambridge')
        old_score = NewsRelevanceScorer.calculate_relevance_score(old_article, 'Cambridge')
        
        assert recent_score > old_score  # Recent articles should score higher


class TestNewsService:
    """Test news service functionality."""
    
    @pytest.fixture
    def news_service(self):
        """Create news service instance."""
        return NewsService()
    
    def test_news_service_init(self, news_service):
        """Test news service initialization."""
        assert news_service.rate_limit == 500
        assert len(news_service.news_caches) == 8  # 8 categories
        assert news_service.relevance_scorer is not None
    
    def test_cache_key_generation(self, news_service):
        """Test cache key generation."""
        key = news_service._get_cache_key('Cambridge', NewsCategory.TRAVEL)
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hash length
        
        # Different inputs should generate different keys
        key2 = news_service._get_cache_key('Oxford', NewsCategory.TRAVEL)
        assert key != key2
    
    def test_validate_location(self, news_service):
        """Test location validation."""
        # Valid location should not raise exception
        try:
            news_service._check_cache_multi('Cambridge', [NewsCategory.TRAVEL])
        except LocationNotFoundException:
            pytest.fail("Valid location should not raise LocationNotFoundException")
    
    @pytest.mark.asyncio
    async def test_location_news_invalid_location(self, news_service):
        """Test location news with invalid location."""
        with pytest.raises(LocationNotFoundException):
            await news_service.get_location_news('InvalidLocation')
    
    @pytest.mark.asyncio
    async def test_location_news_no_api_key(self):
        """Test behavior when no API key is configured."""
        with patch('chatbot.services.news_service.Config.NEWS_API_KEY', None):
            service = NewsService()
            
            with pytest.raises(NewsAPIConnectionException):
                await service.get_location_news('Cambridge')
    
    @pytest.mark.asyncio
    @patch('newsapi.NewsApiClient')
    async def test_fetch_news_success(self, mock_client, news_service):
        """Test successful news fetching."""
        # Mock NewsAPI response
        mock_response = {
            'status': 'ok',
            'articles': [
                {
                    'title': 'Cambridge Tourism Boom',
                    'description': 'Tourist attractions see record visitors',
                    'content': 'Cambridge travel industry reports growth',
                    'url': 'https://example.com/news1',
                    'source': {'name': 'BBC'},
                    'publishedAt': '2025-06-16T10:00:00Z',
                    'urlToImage': 'https://example.com/image1.jpg'
                },
                {
                    'title': 'University News',
                    'description': 'Academic developments',
                    'content': 'Research updates from Cambridge',
                    'url': 'https://example.com/news2',
                    'source': {'name': 'The Guardian'},
                    'publishedAt': '2025-06-16T09:00:00Z',
                    'urlToImage': None
                }
            ]
        }
        
        # Configure mock
        mock_client_instance = MagicMock()
        mock_client_instance.get_everything.return_value = mock_response
        mock_client.return_value = mock_client_instance
        news_service.client = mock_client_instance
        
        # Test the method
        result = await news_service._fetch_news_for_category('Cambridge', NewsCategory.TRAVEL)
        
        assert isinstance(result, list)
        assert len(result) <= 10  # Should return top 10
        
        # Check that articles have relevance scores
        for article in result:
            assert 'relevance_score' in article
            assert 'category' in article
            assert article['relevance_score'] >= 0.3  # Minimum threshold
    
    @pytest.mark.asyncio
    @patch('newsapi.NewsApiClient')
    async def test_location_news_with_cache(self, mock_client, news_service):
        """Test news retrieval with caching."""
        # Mock successful API response
        mock_response = {
            'status': 'ok',
            'articles': [{
                'title': 'Test Article',
                'description': 'Test content with travel keywords',
                'content': 'Cambridge tourism attractions visit',
                'url': 'https://example.com/test',
                'source': {'name': 'Test Source'},
                'publishedAt': '2025-06-16T10:00:00Z',
                'urlToImage': None
            }]
        }
        
        mock_client_instance = MagicMock()
        mock_client_instance.get_everything.return_value = mock_response
        mock_client.return_value = mock_client_instance
        news_service.client = mock_client_instance
        
        # First call should hit API
        result1 = await news_service.get_location_news('Cambridge', [NewsCategory.TRAVEL])
        
        # Second call should use cache
        result2 = await news_service.get_location_news('Cambridge', [NewsCategory.TRAVEL])
        
        # Results should be similar (cache working)
        assert result1['total_articles'] == result2['total_articles']
        assert 'travel' in result1['articles']
        assert 'travel' in result2['articles']
    
    @pytest.mark.asyncio
    async def test_travel_updates_multiple_locations(self, news_service):
        """Test travel updates for multiple locations."""
        with patch.object(news_service, 'get_location_news') as mock_get_news:
            mock_get_news.return_value = {
                'total_articles': 5,
                'articles': {'travel': [{'title': 'Test'}]},
                'categories_searched': ['travel']
            }
            
            result = await news_service.get_travel_updates(['Cambridge', 'Oxford'])
            
            assert 'locations' in result
            assert 'summary' in result
            assert len(result['locations']) == 2
            assert result['summary']['total_locations'] == 2
    
    @pytest.mark.asyncio
    async def test_search_events(self, news_service):
        """Test event searching functionality."""
        with patch.object(news_service, 'get_location_news') as mock_get_news:
            mock_get_news.return_value = {
                'total_articles': 3,
                'articles': {
                    'events': [
                        {
                            'title': 'Cambridge Music Festival June 2025',
                            'description': 'Annual festival returns',
                            'url': 'https://example.com/event1',
                            'source': 'Event Guide',
                            'published_at': '2025-06-16T10:00:00Z'
                        }
                    ]
                },
                'categories_searched': ['events']
            }
            
            result = await news_service.search_events('Cambridge')
            
            assert 'articles' in result
            assert 'events' in result['articles']
            
            # Check that estimated_date is added
            events = result['articles']['events']
            for event in events:
                assert 'estimated_date' in event
    
    def test_format_article(self, news_service):
        """Test article formatting."""
        article = {
            'title': 'Test Article Title',
            'description': 'A' * 250,  # Long description to test truncation
            'url': 'https://example.com/test',
            'source': {'name': 'Test Source'},
            'publishedAt': '2025-06-16T10:00:00Z',
            'urlToImage': 'https://example.com/image.jpg',
            'relevance_score': 0.8,
            'category': 'travel'
        }
        
        formatted = news_service._format_article(article)
        
        assert formatted['title'] == 'Test Article Title'
        assert len(formatted['description']) <= 203  # 200 + "..."
        assert formatted['source'] == 'Test Source'
        assert formatted['relevance_score'] == 0.8
        assert formatted['category'] == 'travel'
    
    def test_extract_event_date(self, news_service):
        """Test event date extraction."""
        article_with_date = {
            'title': 'Cambridge Festival on June 15',
            'description': 'Annual event returns this summer'
        }
        
        article_without_date = {
            'title': 'General news article',
            'description': 'No specific date mentioned'
        }
        
        date1 = news_service._extract_event_date(article_with_date)
        date2 = news_service._extract_event_date(article_without_date)
        
        assert date1 is not None  # Should find "June 15"
        assert date2 is None      # Should not find any date
    
    def test_get_cache_stats(self, news_service):
        """Test cache statistics retrieval."""
        stats = news_service.get_cache_stats()
        
        assert 'category_stats' in stats
        assert 'total_cached_items' in stats
        assert 'api_calls_today' in stats
        assert 'rate_limit' in stats
        
        # Check category stats structure
        for category in NewsCategory:
            assert category.value in stats['category_stats']
            cat_stats = stats['category_stats'][category.value]
            assert 'cached_items' in cat_stats
            assert 'max_size' in cat_stats
            assert 'ttl_seconds' in cat_stats


def test_manual_news_api_connection():
    """Manual test for NewsAPI connection (requires real API key)."""
    import os
    
    # Only run if API key is available
    api_key = os.getenv('NEWS_API_KEY')
    if not api_key or api_key == 'your_news_api_key_here':
        pytest.skip("No valid NewsAPI key available for integration test")
    
    async def test_real_api():
        service = NewsService()
        try:
            # Test news fetching
            result = await service.get_location_news('Cambridge', [NewsCategory.TRAVEL])
            assert 'total_articles' in result
            assert 'articles' in result
            print(f"SUCCESS: Real NewsAPI test passed - {result['total_articles']} articles")
            
        except Exception as e:
            print(f"FAILED: Real NewsAPI test failed - {e}")
            raise
    
    # Run the async test
    asyncio.run(test_real_api())


if __name__ == "__main__":
    # Run manual test if called directly
    test_manual_news_api_connection()