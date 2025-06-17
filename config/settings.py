"""
Author: Cameron Murphy (Student ID: 1049678, GitHub: 0x1049678II)
Date: June 6th 2025

Configuration management for the Indra chatbot.

Centralized configuration system supporting multiple environments with sensible defaults.
Manages API keys, database connections, caching settings, and England location data.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration with environment variable support."""
    
    # Flask configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'indra-travel-bot-secret-key-2025')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 'yes']
    
    # API Keys
    OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    
    # Database configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data/databases/indra.db')
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Caching configuration
    CACHE_TYPE = os.getenv('CACHE_TYPE', 'simple')
    CACHE_DEFAULT_TIMEOUT = int(os.getenv('CACHE_TTL', '3600'))  # 1 hour
    
    # Rate limiting
    RATE_LIMIT_PER_HOUR = int(os.getenv('RATE_LIMIT_PER_HOUR', '100'))
    OPENWEATHER_RATE_LIMIT = int(os.getenv('OPENWEATHER_RATE_LIMIT', '1000'))  # per day
    NEWS_API_RATE_LIMIT = int(os.getenv('NEWS_API_RATE_LIMIT', '1000'))  # per day
    
    # ML configuration
    PREDICTION_CONFIDENCE_THRESHOLD = float(os.getenv('PREDICTION_CONFIDENCE_THRESHOLD', '0.7'))
    ML_MODEL_UPDATE_INTERVAL = int(os.getenv('ML_MODEL_UPDATE_INTERVAL', '86400'))  # 24 hours
    
    # England-specific settings - exact locations from travel blogger's itinerary
    VALID_LOCATIONS = [
        'Cumbria',          # Lake District region
        'Corfe Castle',     # Dorset
        'The Cotswolds',    # Multiple villages
        'Cambridge',        # University city
        'Bristol',          # Southwest England
        'Oxford',           # University city
        'Norwich',          # Norfolk
        'Stonehenge',       # Wiltshire
        'Watergate Bay',    # Cornwall
        'Birmingham'        # West Midlands
    ]
    
    # Location coordinates for weather API calls
    LOCATION_COORDINATES = {
        'Cumbria': {'lat': 54.4609, 'lon': -3.0886},
        'Corfe Castle': {'lat': 50.6395, 'lon': -2.0566},
        'The Cotswolds': {'lat': 51.8330, 'lon': -1.8433},
        'Cambridge': {'lat': 52.2053, 'lon': 0.1218},
        'Bristol': {'lat': 51.4545, 'lon': -2.5879},
        'Oxford': {'lat': 51.7520, 'lon': -1.2577},
        'Norwich': {'lat': 52.6309, 'lon': 1.2974},
        'Stonehenge': {'lat': 51.1789, 'lon': -1.8262},
        'Watergate Bay': {'lat': 50.4429, 'lon': -5.0553},
        'Birmingham': {'lat': 52.4862, 'lon': -1.8904}
    }
    
    # Weather service configuration
    OPENWEATHER_BASE_URL = 'https://api.openweathermap.org/data/2.5'
    WEATHER_CACHE_TTL = int(os.getenv('WEATHER_CACHE_TTL', '1800'))  # 30 minutes
    WEATHER_FORECAST_DAYS = 5
    
    # News service configuration
    NEWS_API_BASE_URL = 'https://newsapi.org/v2'
    NEWS_CACHE_TTL = int(os.getenv('NEWS_CACHE_TTL', '3600'))  # 1 hour
    NEWS_COUNTRY = 'gb'  # United Kingdom
    NEWS_CATEGORIES = ['general', 'business', 'entertainment', 'health', 'science', 'sports', 'technology']
    
    # Activity service configuration
    ACTIVITY_CACHE_TTL = int(os.getenv('ACTIVITY_CACHE_TTL', '7200'))  # 2 hours
    
    # ChatterBot configuration (using ShoneGK's fork)
    CHATTERPY_DATABASE_URI = 'sqlite:///data/databases/chatterbot.db'
    CHATTERPY_TRAINING_DATA_PATH = 'data/training/conversations.txt'
    
    # Logging configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = os.getenv('LOG_FORMAT', 'json')
    
    # Async configuration
    MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', '10'))
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))  # seconds
    
    # File paths
    DATA_DIR = 'data'
    CACHE_DIR = os.path.join(DATA_DIR, 'cache')
    DATABASE_DIR = os.path.join(DATA_DIR, 'databases')
    TRAINING_DIR = os.path.join(DATA_DIR, 'training')
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return status."""
        issues = []
        warnings = []
        
        # Check required API keys
        if not cls.OPENWEATHER_API_KEY:
            issues.append("OPENWEATHER_API_KEY is required")
        
        if not cls.NEWS_API_KEY:
            warnings.append("NEWS_API_KEY not set - news features will be limited")
        
        # Check directories exist
        for directory in [cls.DATA_DIR, cls.CACHE_DIR, cls.DATABASE_DIR, cls.TRAINING_DIR]:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create directory {directory}: {e}")
        
        # Validate location coordinates
        if len(cls.LOCATION_COORDINATES) != len(cls.VALID_LOCATIONS):
            issues.append("Mismatch between VALID_LOCATIONS and LOCATION_COORDINATES")
        
        # Check training data exists
        if not os.path.exists(cls.CHATTERPY_TRAINING_DATA_PATH):
            warnings.append(f"Training data not found at {cls.CHATTERPY_TRAINING_DATA_PATH}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    @classmethod
    def get_location_info(cls, location: str) -> dict[str, str | dict[str, float] | list[str] | Any] | None:
        """Get comprehensive location information."""
        if location not in cls.VALID_LOCATIONS:
            return None
        
        coordinates = cls.LOCATION_COORDINATES.get(location, {})
        return {
            'name': location,
            'coordinates': coordinates,
            'country': 'England',
            'supported_services': ['weather', 'news', 'activities']
        }
    
    @classmethod
    def get_all_locations_info(cls) -> List[Dict[str, Any]]:
        """Get information for all supported locations."""
        return [cls.get_location_info(location) for location in cls.VALID_LOCATIONS]

class DevelopmentConfig(Config):
    """Development-specific configuration."""
    DEBUG = True
    CACHE_TYPE = 'simple'
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production-specific configuration."""
    DEBUG = False
    CACHE_TYPE = 'redis'
    LOG_LEVEL = 'INFO'
    
    # Enhanced rate limiting for production
    RATE_LIMIT_PER_HOUR = 50
    
    # More conservative caching in production
    WEATHER_CACHE_TTL = 3600  # 1 hour
    NEWS_CACHE_TTL = 7200     # 2 hours

class TestingConfig(Config):
    """Testing-specific configuration."""
    TESTING = True
    DEBUG = True
    CACHE_TYPE = 'null'
    DATABASE_URL = 'sqlite:///:memory:'
    
    # Mock API keys for testing
    OPENWEATHER_API_KEY = 'test_openweather_key'
    NEWS_API_KEY = 'test_news_key'

# Configuration selector
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(config_name: str = None) -> type[DevelopmentConfig | ProductionConfig | TestingConfig]:
    """Get configuration class based on environment."""
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'default')
    
    return config.get(config_name, DevelopmentConfig)