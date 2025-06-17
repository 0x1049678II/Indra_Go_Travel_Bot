"""
Author: Cameron Murphy (Student ID: 1049678, GitHub: 0x1049678II)
Date: June 4th 2025

Legacy synchronous weather service when I started the project - maintained for compatibility.

This is the original synchronous weather service using the requests library.
The async version (async_weather_service.py) is now the primary weather service
used by the bot, but this is kept for any legacy compatibility or testing purposes.
"""

from datetime import datetime
from typing import Dict, Any

import requests
import structlog
from cachetools import TTLCache

from chatbot.exceptions import (
    WeatherDataNotFoundException,
    WeatherAPIConnectionException, LocationNotFoundException,
    RateLimitExceededException
)
from config.locations import EnglandLocations
from config.settings import Config

logger = structlog.get_logger()


def _get_cache_key(location: str, forecast_type: str) -> str:
    """Generate cache key for weather data."""
    return f"weather_{location}_{forecast_type}_{datetime.now().strftime('%Y%m%d_%H')}"


class WeatherService:
    """Simple weather service using requests library."""
    
    def __init__(self):
        self.api_key = Config.OPENWEATHER_API_KEY
        self.base_url = Config.OPENWEATHER_BASE_URL
        self.locations = EnglandLocations()
        
        # TTL Cache for weather data (30 minutes)
        self.weather_cache = TTLCache(maxsize=100, ttl=Config.WEATHER_CACHE_TTL)
        
        # Rate limiting tracking
        self.api_calls_today = 0
        self.last_reset_date = datetime.now().date()
        
        # Create requests session with SSL verification disabled
        self.session = requests.Session()
        self.session.verify = False  # Disable SSL verification
        self.session.timeout = Config.REQUEST_TIMEOUT
        
        # Disable SSL warnings
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        logger.info("Weather service initialized with SSL verification disabled")
    
    def _check_rate_limit(self):
        """Check and update rate limiting."""
        current_date = datetime.now().date()
        
        # Reset daily counter
        if current_date > self.last_reset_date:
            self.api_calls_today = 0
            self.last_reset_date = current_date
        
        if self.api_calls_today >= Config.OPENWEATHER_RATE_LIMIT:
            raise RateLimitExceededException(
                'OpenWeather API', 
                Config.OPENWEATHER_RATE_LIMIT,
                f"Resets at midnight UTC"
            )
        
        self.api_calls_today += 1

    def _validate_location(self, location: str) -> Dict[str, float]:
        """Validate location and return coordinates."""
        if location not in Config.VALID_LOCATIONS:
            raise LocationNotFoundException(location, Config.VALID_LOCATIONS)
        
        coordinates = self.locations.get_coordinates(location)
        if not coordinates:
            raise WeatherDataNotFoundException(location, "Coordinates not available")
        
        return coordinates
    
    def _make_api_request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request with error handling."""
        self._check_rate_limit()
        
        try:
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                logger.info("Weather API request successful", 
                           url=url, status=response.status_code)
                return data
            elif response.status_code == 401:
                raise WeatherAPIConnectionException("OpenWeather", 401, "Invalid API key")
            elif response.status_code == 429:
                raise RateLimitExceededException("OpenWeather API", Config.OPENWEATHER_RATE_LIMIT)
            elif response.status_code == 404:
                raise WeatherDataNotFoundException(params.get('q', 'unknown'), "Location not found by API")
            else:
                raise WeatherAPIConnectionException("OpenWeather", response.status_code, response.text)
                
        except requests.exceptions.RequestException as e:
            logger.error("Weather API request failed", error=str(e), url=url)
            raise WeatherAPIConnectionException("OpenWeather", None, str(e))
    
    def get_current_weather(self, location: str) -> Dict[str, Any]:
        """Get current weather for a location."""
        # Check cache first
        cache_key = _get_cache_key(location, 'current')
        if cache_key in self.weather_cache:
            logger.info("Weather data retrieved from cache", location=location, type='current')
            return self.weather_cache[cache_key]
        
        # Validate location
        coordinates = self._validate_location(location)
        
        # Prepare API request
        url = f"{self.base_url}/weather"
        params = {
            'lat': coordinates['lat'],
            'lon': coordinates['lon'],
            'appid': self.api_key,
            'units': 'metric',
            'lang': 'en'
        }
        
        # Make API request
        raw_data = self._make_api_request(url, params)
        
        # Process and structure the data
        processed_data = self._process_current_weather(raw_data, location)
        
        # Cache the result
        self.weather_cache[cache_key] = processed_data
        
        logger.info("Current weather data retrieved", location=location)
        return processed_data
    
    def get_5_day_forecast(self, location: str) -> Dict[str, Any]:
        """Get 5-day weather forecast for a location."""
        # Check cache first
        cache_key = _get_cache_key(location, '5day')
        if cache_key in self.weather_cache:
            logger.info("Forecast data retrieved from cache", location=location, type='5day')
            return self.weather_cache[cache_key]
        
        # Validate location
        coordinates = self._validate_location(location)
        
        # Prepare API request
        url = f"{self.base_url}/forecast"
        params = {
            'lat': coordinates['lat'],
            'lon': coordinates['lon'],
            'appid': self.api_key,
            'units': 'metric',
            'lang': 'en'
        }
        
        # Make API request
        raw_data = self._make_api_request(url, params)
        
        # Process and structure the data
        processed_data = self._process_5_day_forecast(raw_data, location)
        
        # Cache the result
        self.weather_cache[cache_key] = processed_data
        
        logger.info("5-day forecast data retrieved", location=location)
        return processed_data
    
    def _process_current_weather(self, raw_data: Dict[str, Any], location: str) -> Dict[str, Any]:
        """Process raw current weather data into structured format."""
        return {
            'location': location,
            'timestamp': datetime.now().isoformat(),
            'current': {
                'temperature': round(raw_data['main']['temp'], 1),
                'feels_like': round(raw_data['main']['feels_like'], 1),
                'humidity': raw_data['main']['humidity'],
                'pressure': raw_data['main']['pressure'],
                'description': raw_data['weather'][0]['description'].title(),
                'main': raw_data['weather'][0]['main'],
                'icon': raw_data['weather'][0]['icon'],
                'wind_speed': raw_data.get('wind', {}).get('speed', 0),
                'wind_direction': raw_data.get('wind', {}).get('deg', 0),
                'visibility': raw_data.get('visibility', 0) / 1000,  # Convert to km
                'uv_index': raw_data.get('uvi', 0)
            },
            'sun': {
                'sunrise': datetime.fromtimestamp(raw_data['sys']['sunrise']).strftime('%H:%M'),
                'sunset': datetime.fromtimestamp(raw_data['sys']['sunset']).strftime('%H:%M')
            },
            'location_info': self.locations.get_location(location).__dict__ if self.locations.get_location(location) else {}
        }
    
    def _process_5_day_forecast(self, raw_data: Dict[str, Any], location: str) -> Dict[str, Any]:
        """Process raw 5-day forecast data into structured format."""
        daily_forecasts = {}
        
        # Group forecast data by day
        for item in raw_data['list']:
            dt = datetime.fromtimestamp(item['dt'])
            day_key = dt.strftime('%Y-%m-%d')
            
            if day_key not in daily_forecasts:
                daily_forecasts[day_key] = {
                    'date': day_key,
                    'day_name': dt.strftime('%A'),
                    'forecasts': [],
                    'temperatures': [],
                    'conditions': []
                }
            
            daily_forecasts[day_key]['forecasts'].append({
                'time': dt.strftime('%H:%M'),
                'temperature': round(item['main']['temp'], 1),
                'description': item['weather'][0]['description'].title(),
                'main': item['weather'][0]['main'],
                'icon': item['weather'][0]['icon'],
                'humidity': item['main']['humidity'],
                'wind_speed': item.get('wind', {}).get('speed', 0),
                'precipitation_chance': item.get('pop', 0) * 100
            })
            
            daily_forecasts[day_key]['temperatures'].append(item['main']['temp'])
            daily_forecasts[day_key]['conditions'].append(item['weather'][0]['main'])
        
        # Process daily summaries
        processed_days = []
        for day_key in sorted(daily_forecasts.keys())[:5]:  # Ensure only 5 days
            day_data = daily_forecasts[day_key]
            temps = day_data['temperatures']
            conditions = day_data['conditions']
            
            # Determine dominant weather condition
            condition_counts = {}
            for condition in conditions:
                condition_counts[condition] = condition_counts.get(condition, 0) + 1
            dominant_condition = max(condition_counts, key=condition_counts.get)
            
            processed_days.append({
                'date': day_data['date'],
                'day_name': day_data['day_name'],
                'temperature_min': round(min(temps), 1),
                'temperature_max': round(max(temps), 1),
                'dominant_condition': dominant_condition,
                'detailed_forecasts': day_data['forecasts'],
                'summary': f"{dominant_condition} weather with temperatures {round(min(temps), 1)}-{round(max(temps), 1)}°C"
            })
        
        return {
            'location': location,
            'timestamp': datetime.now().isoformat(),
            'forecast_type': '5-day',
            'days': processed_days,
            'location_info': self.locations.get_location(location).__dict__ if self.locations.get_location(location) else {}
        }