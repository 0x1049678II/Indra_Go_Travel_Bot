"""
Author: Cameron Murphy (Student ID: 1049678, GitHub: 0x1049678II)
Date: June 8th 2025

Async weather service with predictive analytics and activity recommendations.
This module replaces the now legacy (weather_service.py).

The weather service is designed to handle the notoriously unpredictable English weather
with grace and (somewhat)intelligence. It provides async weather data fetching, predictive analytics
for travel planning, and weather-based activity recommendations because let's face it - 
the weather is going to influence your travel plans whether you plan for it or not.

The service uses aiohttp for non-blocking HTTP requests and includes caching
to avoid hammering the OpenWeather API (and hitting rate limits). It also provides
comfort indices and travel recommendations based on weather conditions.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple

import aiohttp
import numpy as np
import structlog
from cachetools import TTLCache
from scipy import stats

from chatbot.exceptions import (
    WeatherAPIException, WeatherDataNotFoundException,
    WeatherAPIConnectionException, LocationNotFoundException,
    RateLimitExceededException, AsyncTimeoutException
)
from config.locations import EnglandLocations
from config.settings import Config

logger = structlog.get_logger()


class WeatherCondition(Enum):
    """Weather condition categories for activity recommendations."""
    CLEAR = "Clear"
    PARTLY_CLOUDY = "Clouds"
    OVERCAST = "Overcast"
    LIGHT_RAIN = "Light Rain"
    HEAVY_RAIN = "Heavy Rain"
    SNOW = "Snow"
    FOG = "Fog"
    STORM = "Thunderstorm"


class ActivityType(Enum):
    """Activity types based on weather suitability."""
    OUTDOOR_EXCELLENT = "outdoor_excellent"
    OUTDOOR_GOOD = "outdoor_good"
    OUTDOOR_POOR = "outdoor_poor"
    INDOOR_PREFERRED = "indoor_preferred"
    INDOOR_REQUIRED = "indoor_required"


@dataclass
class WeatherTrend:
    """Weather trend analysis data."""
    direction: str  # "improving", "deteriorating", "stable"
    confidence: float  # 0.0 to 1.0
    temperature_trend: float  # degrees per day
    precipitation_trend: float  # probability change per day
    description: str


@dataclass
class ActivityRecommendation:
    """Weather-based activity recommendation."""
    activity_type: ActivityType
    suitability_score: float  # 0.0 to 1.0
    reasons: List[str]
    specific_activities: List[str]
    timing_advice: str


class WeatherPredictor:
    """Predictive analytics for weather trends and patterns."""
    
    @staticmethod
    def analyze_temperature_trend(temperatures: List[float], 
                                dates: List[datetime]) -> Tuple[float, float]:
        """
        Analyze temperature trend using linear regression.
        
        Args:
            temperatures: List of temperature values
            dates: Corresponding dates
            
        Returns:
            Tuple of (slope_per_day, r_squared)
        """
        if len(temperatures) < 3:
            return 0.0, 0.0
        
        # Convert dates to days since first date
        first_date = dates[0]
        x = [(date - first_date).days for date in dates]
        y = temperatures
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        return slope, r_value ** 2
    
    @staticmethod
    def predict_weather_pattern(historical_data: List[Dict[str, Any]]) -> WeatherTrend:
        """
        Predict weather patterns based on historical data.
        
        Args:
            historical_data: List of weather data points
            
        Returns:
            WeatherTrend object with prediction
        """
        if len(historical_data) < 5:
            return WeatherTrend(
                direction="insufficient_data",
                confidence=0.0,
                temperature_trend=0.0,
                precipitation_trend=0.0,
                description="Insufficient data for trend analysis"
            )
        
        # Extract data
        temperatures = [float(d.get('temperature', 0)) for d in historical_data]
        precipitation_probs = [float(d.get('precipitation_chance', 0)) for d in historical_data]
        dates = [datetime.fromisoformat(d.get('timestamp', datetime.now().isoformat())) 
                for d in historical_data]
        
        # Analyze temperature trend
        temp_slope, temp_r2 = WeatherPredictor.analyze_temperature_trend(temperatures, dates)
        
        # Analyze precipitation trend
        if len(precipitation_probs) >= 3:
            x = list(range(len(precipitation_probs)))
            precip_slope, _, precip_r_value, _, _ = stats.linregress(x, precipitation_probs)
            precip_r2 = precip_r_value ** 2
        else:
            precip_slope, precip_r2 = 0.0, 0.0
        
        # Determine overall trend direction
        if temp_slope > 0.5 and precip_slope < -5:
            direction = "improving"
            confidence = min(temp_r2 + precip_r2, 1.0) * 0.8
        elif temp_slope < -0.5 and precip_slope > 5:
            direction = "deteriorating"
            confidence = min(temp_r2 + precip_r2, 1.0) * 0.8
        else:
            direction = "stable"
            confidence = 1.0 - abs(temp_slope) * 0.1
        
        # Generate description
        if direction == "improving":
            description = f"Weather improving: temperatures rising {temp_slope:.1f}°C/day, less precipitation likely"
        elif direction == "deteriorating":
            description = f"Weather deteriorating: temperatures dropping {abs(temp_slope):.1f}°C/day, more precipitation likely"
        else:
            description = "Weather conditions remaining relatively stable"
        
        return WeatherTrend(
            direction=direction,
            confidence=confidence,
            temperature_trend=temp_slope,
            precipitation_trend=precip_slope,
            description=description
        )


class ActivityRecommendationEngine:
    """Generate weather-based activity recommendations."""
    
    # Activity mappings for different weather conditions
    ACTIVITY_MAPPINGS = {
        ActivityType.OUTDOOR_EXCELLENT: [
            "Walking tours", "Outdoor photography", "Garden visits", 
            "Castle exploration", "Hiking trails", "Outdoor markets",
            "Sightseeing", "River walks", "Punting", "Cycling"
        ],
        ActivityType.OUTDOOR_GOOD: [
            "Covered markets", "Cathedral visits", "Short walks",
            "Outdoor dining", "Bridge viewing", "Park visits"
        ],
        ActivityType.INDOOR_PREFERRED: [
            "Museums", "Art galleries", "Historic houses", "Shopping centers",
            "Universities tours", "Libraries", "Indoor exhibitions"
        ],
        ActivityType.INDOOR_REQUIRED: [
            "Museums", "Galleries", "Theaters", "Shopping", "Cafes",
            "Pubs", "Indoor entertainment", "Spas", "Cinemas"
        ]
    }
    
    @staticmethod
    def calculate_activity_suitability(weather_data: Dict[str, Any]) -> List[ActivityRecommendation]:
        """
        Calculate activity suitability based on current weather.
        
        Args:
            weather_data: Current weather information
            
        Returns:
            List of activity recommendations sorted by suitability
        """
        current = weather_data.get('current', {})
        temperature = float(current.get('temperature', 0))
        condition = current.get('main', 'Unknown')
        wind_speed = float(current.get('wind_speed', 0))
        humidity = float(current.get('humidity', 50))
        visibility = float(current.get('visibility', 10))
        
        recommendations = []
        
        # Outdoor Excellent conditions
        if (condition == 'Clear' and 15 <= temperature <= 25 and 
            wind_speed < 10 and humidity < 70):
            score = 0.9 + (20 - abs(temperature - 20)) * 0.005
            reasons = [
                f"Clear skies with {temperature}°C",
                f"Light winds ({wind_speed} m/s)",
                f"Low humidity ({humidity}%)"
            ]
            timing = "Perfect conditions all day"
            
            recommendations.append(ActivityRecommendation(
                activity_type=ActivityType.OUTDOOR_EXCELLENT,
                suitability_score=min(score, 1.0),
                reasons=reasons,
                specific_activities=ActivityRecommendationEngine.ACTIVITY_MAPPINGS[ActivityType.OUTDOOR_EXCELLENT],
                timing_advice=timing
            ))
        
        # Outdoor Good conditions
        elif (condition in ['Clear', 'Clouds'] and 10 <= temperature <= 30 and 
              wind_speed < 15):
            score = 0.7 + (25 - abs(temperature - 18)) * 0.01
            reasons = [
                f"{condition.lower()} conditions",
                f"Comfortable {temperature}°C",
                f"Moderate winds ({wind_speed} m/s)"
            ]
            timing = "Good for outdoor activities with layers"
            
            recommendations.append(ActivityRecommendation(
                activity_type=ActivityType.OUTDOOR_GOOD,
                suitability_score=min(score, 1.0),
                reasons=reasons,
                specific_activities=ActivityRecommendationEngine.ACTIVITY_MAPPINGS[ActivityType.OUTDOOR_GOOD],
                timing_advice=timing
            ))
        
        # Indoor Preferred conditions
        elif (condition in ['Rain', 'Drizzle'] or temperature < 5 or 
              temperature > 30 or wind_speed > 20):
            score = 0.8 - (abs(temperature - 18) * 0.02)
            reasons = []
            if condition in ['Rain', 'Drizzle']:
                reasons.append(f"{condition} conditions")
            if temperature < 5:
                reasons.append(f"Cold temperature ({temperature}°C)")
            elif temperature > 30:
                reasons.append(f"Very warm ({temperature}°C)")
            if wind_speed > 20:
                reasons.append(f"Strong winds ({wind_speed} m/s)")
            
            timing = "Indoor activities recommended"
            
            recommendations.append(ActivityRecommendation(
                activity_type=ActivityType.INDOOR_PREFERRED,
                suitability_score=max(score, 0.5),
                reasons=reasons,
                specific_activities=ActivityRecommendationEngine.ACTIVITY_MAPPINGS[ActivityType.INDOOR_PREFERRED],
                timing_advice=timing
            ))
        
        # Indoor Required conditions
        elif (condition in ['Thunderstorm', 'Snow', 'Fog'] or 
              temperature < 0 or wind_speed > 30):
            score = 0.9
            reasons = []
            if condition == 'Thunderstorm':
                reasons.append("Thunderstorm conditions - safety first")
            elif condition == 'Snow':
                reasons.append("Snow conditions")
            elif condition == 'Fog':
                reasons.append(f"Poor visibility ({visibility} km)")
            if temperature < 0:
                reasons.append(f"Freezing temperatures ({temperature}°C)")
            if wind_speed > 30:
                reasons.append(f"Very strong winds ({wind_speed} m/s)")
            
            timing = "Stay indoors for safety and comfort"
            
            recommendations.append(ActivityRecommendation(
                activity_type=ActivityType.INDOOR_REQUIRED,
                suitability_score=score,
                reasons=reasons,
                specific_activities=ActivityRecommendationEngine.ACTIVITY_MAPPINGS[ActivityType.INDOOR_REQUIRED],
                timing_advice=timing
            ))
        
        # Default recommendation if no specific conditions met
        if not recommendations:
            recommendations.append(ActivityRecommendation(
                activity_type=ActivityType.INDOOR_PREFERRED,
                suitability_score=0.6,
                reasons=["Mixed weather conditions"],
                specific_activities=ActivityRecommendationEngine.ACTIVITY_MAPPINGS[ActivityType.INDOOR_PREFERRED],
                timing_advice="Indoor activities recommended as backup"
            ))
        
        return sorted(recommendations, key=lambda x: x.suitability_score, reverse=True)


class AsyncWeatherService:
    """
    Async weather service - your friendly neighborhood meteorological oracle.
    
    This service handles all weather-related operations for the Indra travel bot,
    with async capabilities that would make Zeus himself proud. It fetches weather
    data from OpenWeatherMap, provides predictive analytics for travel planning,
    and recommends activities based on weather conditions.
    
    The service is designed around the principle that English weather is:
    1. Unpredictable (it will rain when you least expect it, a bit like Melbourne)
    2. Conversation-worthy (everyone talks about it for good reason)
    3. Travel-critical (affects every outdoor plan you make)
    
    Features:
        - Async HTTP requests with aiohttp (non-blocking, naturally)
        - Multi-tier caching system (weather, forecasts, analytics)
        - Predictive analytics for travel optimization
        - Activity recommendation engine based on weather
        - Comfort index calculations for travel planning
        - Rate limiting to stay within API quotas
        - Concurrent location comparisons
        
    The service uses context manager pattern for proper session lifecycle management
    because leaving HTTP sessions open is like leaving the car running - wasteful
    and eventually problematic.
    
    Attributes:
        api_key: OpenWeather API key for data access
        base_url: OpenWeather API base URL
        locations: England locations configuration
        predictor: Weather prediction and analytics engine
        activity_engine: Activity recommendation system
        weather_cache: TTL cache for current weather data
        forecast_cache: TTL cache for forecast data (longer TTL)
        analytics_cache: TTL cache for analytics results (longest TTL)
        session: aiohttp session for async HTTP requests
        
    Usage:
        async with AsyncWeatherService() as weather_service:
            weather = await weather_service.get_current_weather_async("Cambridge")
    """
    
    def __init__(self):
        """
        Initialize the async weather service.
        
        Sets up caching, rate limiting, and prepares for async operations.
        The actual HTTP session is created lazily in the context manager
        to avoid resource leaks and ensure proper lifecycle management.
        """
        self.api_key = Config.OPENWEATHER_API_KEY
        self.base_url = Config.OPENWEATHER_BASE_URL
        self.locations = EnglandLocations()
        self.predictor = WeatherPredictor()
        self.activity_engine = ActivityRecommendationEngine()
        
        # Async caching with separate caches for different data types
        self.weather_cache = TTLCache(maxsize=100, ttl=Config.WEATHER_CACHE_TTL)
        self.forecast_cache = TTLCache(maxsize=50, ttl=3600)  # 1 hour for forecasts
        self.analytics_cache = TTLCache(maxsize=30, ttl=7200)  # 2 hours for analytics
        
        # Rate limiting tracking
        self.api_calls_today = 0
        self.last_reset_date = datetime.now().date()
        
        # Async session will be created when needed
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info("Async weather service initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        logger.info("Initializing async weather service session")
        
        # Create session with optimized settings
        timeout = aiohttp.ClientTimeout(total=Config.REQUEST_TIMEOUT)
        connector = aiohttp.TCPConnector(
            limit=Config.MAX_CONCURRENT_REQUESTS,
            ssl=False,  # Disable SSL verification for compatibility
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={'User-Agent': 'Indra-Travel-Bot/1.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
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
    
    def _get_cache_key(self, location: str, data_type: str, 
                      params: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for weather data."""
        key_parts = [location, data_type, datetime.now().strftime('%Y%m%d_%H')]
        if params:
            key_parts.append(str(sorted(params.items())))
        return "_".join(key_parts)
    
    def _validate_location(self, location: str) -> Dict[str, float]:
        """Validate location and return coordinates."""
        if location not in Config.VALID_LOCATIONS:
            raise LocationNotFoundException(location, Config.VALID_LOCATIONS)
        
        coordinates = self.locations.get_coordinates(location)
        if not coordinates:
            raise WeatherDataNotFoundException(location, "Coordinates not available")
        
        return coordinates
    
    async def _make_api_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make async API request with enhanced error handling."""
        if not self.session:
            raise WeatherAPIConnectionException("OpenWeather", None, "Session not initialized")
        
        self._check_rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("Weather API request successful", 
                               endpoint=endpoint, status=response.status)
                    return data
                elif response.status == 401:
                    raise WeatherAPIConnectionException("OpenWeather", 401, "Invalid API key")
                elif response.status == 429:
                    raise RateLimitExceededException("OpenWeather API", Config.OPENWEATHER_RATE_LIMIT)
                elif response.status == 404:
                    raise WeatherDataNotFoundException(params.get('q', 'unknown'), "Location not found by API")
                else:
                    response_text = await response.text()
                    raise WeatherAPIConnectionException("OpenWeather", response.status, response_text)
                    
        except asyncio.TimeoutError:
            raise AsyncTimeoutException("Weather API request", Config.REQUEST_TIMEOUT)
        except aiohttp.ClientError as e:
            raise WeatherAPIConnectionException("OpenWeather", None, str(e))
    
    async def get_current_weather_async(self, location: str) -> Dict[str, Any]:
        """
        Get current weather with activity recommendations.
        
        Args:
            location: England destination name
            
        Returns:
            Weather data with activity recommendations
        """
        # Check cache first
        cache_key = self._get_cache_key(location, 'current_async')
        if cache_key in self.weather_cache:
            logger.info("Weather data retrieved from cache", 
                       location=location, type='current_async')
            return self.weather_cache[cache_key]
        
        # Validate location
        coordinates = self._validate_location(location)
        
        # Prepare API request
        params = {
            'lat': coordinates['lat'],
            'lon': coordinates['lon'],
            'appid': self.api_key,
            'units': 'metric',
            'lang': 'en'
        }
        
        # Make API request
        raw_data = await self._make_api_request('weather', params)
        
        # Process the data with recommendations
        processed_data = self._process_current_weather_with_activities(raw_data, location)
        
        # Cache the result
        self.weather_cache[cache_key] = processed_data
        
        logger.info("Current weather data with recommendations retrieved", location=location)
        return processed_data
    
    def _process_current_weather_with_activities(self, raw_data: Dict[str, Any], 
                                              location: str) -> Dict[str, Any]:
        """Process raw current weather data with activity recommendations."""
        # Basic weather processing
        processed_data = {
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
                'uv_index': raw_data.get('uvi', 0),
                'cloud_cover': raw_data.get('clouds', {}).get('all', 0)
            },
            'sun': {
                'sunrise': datetime.fromtimestamp(raw_data['sys']['sunrise']).strftime('%H:%M'),
                'sunset': datetime.fromtimestamp(raw_data['sys']['sunset']).strftime('%H:%M')
            },
            'location_info': self.locations.get_location(location).__dict__ if self.locations.get_location(location) else {}
        }
        
        # Add activity recommendations
        activity_recommendations: list[ActivityRecommendation] = self.activity_engine.calculate_activity_suitability(processed_data)
        processed_data['activity_recommendations'] = [
            {
                'type': rec.activity_type.value,
                'suitability_score': rec.suitability_score,
                'reasons': rec.reasons,
                'activities': rec.specific_activities[:5],  # Top 5 activities
                'timing_advice': rec.timing_advice
            }
            for rec in activity_recommendations[:3]  # Top 3 recommendation types
        ]
        
        # Add comfort indices
        processed_data['comfort_indices'] = self._calculate_comfort_indices(processed_data['current'])
        
        return processed_data
    
    def _calculate_comfort_indices(self, current_weather: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate various comfort indices for activities."""
        temp = current_weather['temperature']
        humidity = current_weather['humidity']
        wind_speed = current_weather['wind_speed']
        
        # Heat Index (simplified)
        if temp >= 27:
            heat_index = temp + (0.5 * (humidity - 50))
        else:
            heat_index = temp
        
        # Wind Chill (simplified)
        if temp <= 10 and wind_speed > 5:
            wind_chill = temp - (wind_speed * 0.5)
        else:
            wind_chill = temp
        
        # Comfort Score (0-100)
        comfort_score = 100
        if temp < 10 or temp > 25:
            comfort_score -= abs(temp - 17.5) * 2
        if humidity > 70:
            comfort_score -= (humidity - 70) * 0.5
        if wind_speed > 15:
            comfort_score -= (wind_speed - 15) * 2
        
        comfort_score = max(0, min(100, comfort_score))
        
        return {
            'heat_index': round(heat_index, 1),
            'wind_chill': round(wind_chill, 1),
            'comfort_score': round(comfort_score),
            'comfort_level': (
                'Excellent' if comfort_score >= 80 else
                'Good' if comfort_score >= 60 else
                'Fair' if comfort_score >= 40 else
                'Poor'
            )
        }
    
    async def get_forecast_with_analytics(self, location: str, 
                                        days: int = 5) -> Dict[str, Any]:
        """
        Get weather forecast with predictive analytics.
        
        Args:
            location: England destination name
            days: Number of forecast days (1-5)
            
        Returns:
            Forecast data with trend analysis and predictions
        """
        days = min(max(days, 1), 5)  # Ensure 1-5 range
        
        # Check cache first
        cache_key = self._get_cache_key(location, f'forecast_{days}d')
        if cache_key in self.forecast_cache:
            logger.info("Forecast with analytics retrieved from cache", 
                       location=location, days=days)
            return self.forecast_cache[cache_key]
        
        # Validate location
        coordinates = self._validate_location(location)
        
        # Prepare API request
        params = {
            'lat': coordinates['lat'],
            'lon': coordinates['lon'],
            'appid': self.api_key,
            'units': 'metric',
            'lang': 'en'
        }
        
        # Make API request
        raw_data = await self._make_api_request('forecast', params)
        
        # Process and analyze the data
        analyzed_data = self._process_forecast_with_analytics(raw_data, location, days)
        
        # Cache the result
        self.forecast_cache[cache_key] = analyzed_data
        
        logger.info("Forecast with analytics retrieved", location=location, days=days)
        return analyzed_data
    
    def _process_forecast_with_analytics(self, raw_data: Dict[str, Any], 
                                       location: str, days: int) -> Dict[str, Any]:
        """Process forecast data with predictive analytics."""
        daily_forecasts = {}
        historical_points = []
        
        # Group forecast data by day
        for item in raw_data['list'][:days * 8]:  # 8 forecasts per day (3-hour intervals)
            dt = datetime.fromtimestamp(item['dt'])
            day_key = dt.strftime('%Y-%m-%d')
            
            if day_key not in daily_forecasts:
                daily_forecasts[day_key] = {
                    'date': day_key,
                    'day_name': dt.strftime('%A'),
                    'forecasts': [],
                    'temperatures': [],
                    'conditions': [],
                    'precipitation_chances': [],
                    'wind_speeds': []
                }
            
            forecast_point = {
                'time': dt.strftime('%H:%M'),
                'temperature': round(item['main']['temp'], 1),
                'feels_like': round(item['main']['feels_like'], 1),
                'description': item['weather'][0]['description'].title(),
                'main': item['weather'][0]['main'],
                'icon': item['weather'][0]['icon'],
                'humidity': item['main']['humidity'],
                'wind_speed': item.get('wind', {}).get('speed', 0),
                'precipitation_chance': item.get('pop', 0) * 100,
                'pressure': item['main']['pressure']
            }
            
            daily_forecasts[day_key]['forecasts'].append(forecast_point)
            daily_forecasts[day_key]['temperatures'].append(item['main']['temp'])
            daily_forecasts[day_key]['conditions'].append(item['weather'][0]['main'])
            daily_forecasts[day_key]['precipitation_chances'].append(item.get('pop', 0) * 100)
            daily_forecasts[day_key]['wind_speeds'].append(item.get('wind', {}).get('speed', 0))
            
            # Collect data for trend analysis
            historical_points.append({
                'timestamp': dt.isoformat(),
                'temperature': item['main']['temp'],
                'precipitation_chance': item.get('pop', 0) * 100,
                'pressure': item['main']['pressure']
            })
        
        # Process daily summaries with activity recommendations
        processed_days = []
        for day_key in sorted(daily_forecasts.keys())[:days]:
            day_data = daily_forecasts[day_key]
            temps = day_data['temperatures']
            conditions = day_data['conditions']
            precip_chances = day_data['precipitation_chances']
            wind_speeds = day_data['wind_speeds']
            
            # Calculate daily statistics
            temp_min = round(min(temps), 1)
            temp_max = round(max(temps), 1)
            temp_avg = round(sum(temps) / len(temps), 1)
            precip_avg = round(sum(precip_chances) / len(precip_chances), 1)
            wind_avg = round(sum(wind_speeds) / len(wind_speeds), 1)
            
            # Determine dominant weather condition
            condition_counts = {}
            for condition in conditions:
                condition_counts[condition] = condition_counts.get(condition, 0) + 1
            dominant_condition = max(condition_counts, key=condition_counts.get)
            
            # Create mock current weather for activity recommendations
            mock_current = {
                'current': {
                    'temperature': temp_avg,
                    'main': dominant_condition,
                    'wind_speed': wind_avg,
                    'humidity': 60,  # Estimate
                    'visibility': 10  # Estimate
                }
            }
            
            activity_recs = self.activity_engine.calculate_activity_suitability(mock_current)
            
            processed_day = {
                'date': day_data['date'],
                'day_name': day_data['day_name'],
                'temperature_min': temp_min,
                'temperature_max': temp_max,
                'temperature_avg': temp_avg,
                'dominant_condition': dominant_condition,
                'precipitation_chance': precip_avg,
                'wind_speed_avg': wind_avg,
                'detailed_forecasts': day_data['forecasts'],
                'activity_recommendations': [
                    {
                        'type': rec.activity_type.value,
                        'suitability_score': rec.suitability_score,
                        'top_activities': rec.specific_activities[:3],
                        'timing_advice': rec.timing_advice
                    }
                    for rec in activity_recs[:2]  # Top 2 recommendations per day
                ],
                'summary': self._generate_daily_summary(
                    temp_min, temp_max, dominant_condition, precip_avg
                )
            }
            
            processed_days.append(processed_day)
        
        # Perform trend analysis
        weather_trend = self.predictor.predict_weather_pattern(historical_points)
        
        return {
            'location': location,
            'timestamp': datetime.now().isoformat(),
            'forecast_type': f'{days}-day',
            'days': processed_days,
            'analytics': {
                'trend_analysis': {
                    'direction': weather_trend.direction,
                    'confidence': weather_trend.confidence,
                    'temperature_trend': weather_trend.temperature_trend,
                    'precipitation_trend': weather_trend.precipitation_trend,
                    'description': weather_trend.description
                },
                'summary_statistics': self._calculate_period_statistics(processed_days),
                'best_activity_days': self._identify_best_activity_days(processed_days)
            },
            'location_info': self.locations.get_location(location).__dict__ if self.locations.get_location(location) else {}
        }
    
    def _generate_daily_summary(self, temp_min: float, temp_max: float, 
                              condition: str, precip_chance: float) -> str:
        """Generate daily summary with activity advice."""
        base_summary = f"{condition} weather with temperatures {temp_min}-{temp_max}°C"
        
        if precip_chance > 70:
            base_summary += f", {precip_chance:.0f}% chance of precipitation"
        elif precip_chance > 30:
            base_summary += f", {precip_chance:.0f}% chance of light precipitation"
        
        # Add activity advice
        if condition == 'Clear' and temp_min > 12 and temp_max < 25:
            base_summary += ". Excellent day for outdoor activities"
        elif condition in ['Clouds', 'Partly Cloudy'] and temp_min > 8:
            base_summary += ". Good day for mixed indoor/outdoor activities"
        elif condition in ['Rain', 'Drizzle'] or temp_max < 5:
            base_summary += ". Indoor activities recommended"
        else:
            base_summary += ". Check hourly forecast for best activity timing"
        
        return base_summary
    
    def _calculate_period_statistics(self, days_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for the forecast period."""
        all_temps = []
        all_precip = []
        condition_counts = {}
        
        for day in days_data:
            all_temps.extend([day['temperature_min'], day['temperature_max']])
            all_precip.append(day['precipitation_chance'])
            
            condition = day['dominant_condition']
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
        
        return {
            'temperature_range': {
                'absolute_min': min(all_temps),
                'absolute_max': max(all_temps),
                'average': round(sum(all_temps) / len(all_temps), 1)
            },
            'precipitation': {
                'average_chance': round(sum(all_precip) / len(all_precip), 1),
                'max_chance': max(all_precip),
                'rainy_days': len([p for p in all_precip if p > 50])
            },
            'dominant_conditions': dict(sorted(condition_counts.items(), 
                                             key=lambda x: x[1], reverse=True))
        }
    
    def _identify_best_activity_days(self, days_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify the best days for different types of activities."""
        outdoor_scores = []
        indoor_scores = []
        
        for i, day in enumerate(days_data):
            outdoor_score = 0
            indoor_score = 0
            
            for rec in day.get('activity_recommendations', []):
                if 'outdoor' in rec['type']:
                    outdoor_score = max(outdoor_score, rec['suitability_score'])
                else:
                    indoor_score = max(indoor_score, rec['suitability_score'])
            
            outdoor_scores.append((i, day['day_name'], outdoor_score))
            indoor_scores.append((i, day['day_name'], indoor_score))
        
        best_outdoor = max(outdoor_scores, key=lambda x: x[2])
        best_indoor = max(indoor_scores, key=lambda x: x[2])
        
        return {
            'best_outdoor_day': {
                'day': best_outdoor[1],
                'score': best_outdoor[2],
                'recommendation': f"{best_outdoor[1]} is best for outdoor activities"
            },
            'best_indoor_day': {
                'day': best_indoor[1],
                'score': best_indoor[2],
                'recommendation': f"{best_indoor[1]} is best for indoor activities"
            }
        }
    
    async def compare_locations_weather(self, locations: List[str]) -> Dict[str, Any]:
        """
        Compare weather across multiple locations with analytics.
        
        Args:
            locations: List of England destinations
            
        Returns:
            Weather comparison with recommendations
        """
        if len(locations) > 5:
            raise WeatherDataNotFoundException("multiple", "Too many locations for comparison (max 5)")
        
        # Validate all locations
        valid_locations = []
        for location in locations:
            if location in Config.VALID_LOCATIONS:
                valid_locations.append(location)
        
        if not valid_locations:
            raise LocationNotFoundException("multiple", Config.VALID_LOCATIONS)
        
        # Get weather data for all locations concurrently
        tasks = [self.get_current_weather_async(location) for location in valid_locations]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            raise WeatherAPIException(f"Failed to get comparison data: {str(e)}")
        
        # Process results
        comparison_data = {
            'locations': {},
            'comparison_analytics': {},
            'recommendations': {},
            'timestamp': datetime.now().isoformat()
        }
        
        successful_results = {}
        for location, result in zip(valid_locations, results):
            if isinstance(result, Exception):
                logger.warning("Failed to get weather for location", location=location, error=str(result))
                continue
            successful_results[location] = result
            comparison_data['locations'][location] = result
        
        # Generate comparison analytics
        if len(successful_results) >= 2:
            comparison_data['comparison_analytics'] = self._generate_location_comparison(successful_results)
            comparison_data['recommendations'] = self._generate_travel_recommendations(successful_results)
        
        return comparison_data
    
    def _generate_location_comparison(self, weather_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed comparison analytics between locations."""
        temperatures = {loc: data['current']['temperature'] for loc, data in weather_data.items()}
        comfort_scores = {loc: data['comfort_indices']['comfort_score'] for loc, data in weather_data.items()}
        conditions = {loc: data['current']['main'] for loc, data in weather_data.items()}
        
        # Find extremes
        warmest_location = max(temperatures, key=temperatures.get)
        coolest_location = min(temperatures, key=temperatures.get)
        most_comfortable = max(comfort_scores, key=comfort_scores.get)
        
        # Calculate statistics
        temp_variance = np.var(list(temperatures.values()))
        avg_temperature = np.mean(list(temperatures.values()))
        
        return {
            'temperature_analysis': {
                'warmest': {
                    'location': warmest_location,
                    'temperature': temperatures[warmest_location]
                },
                'coolest': {
                    'location': coolest_location,
                    'temperature': temperatures[coolest_location]
                },
                'temperature_spread': round(temperatures[warmest_location] - temperatures[coolest_location], 1),
                'average_temperature': round(avg_temperature, 1),
                'temperature_variance': round(temp_variance, 2)
            },
            'comfort_analysis': {
                'most_comfortable': {
                    'location': most_comfortable,
                    'score': comfort_scores[most_comfortable]
                },
                'comfort_ranking': sorted(comfort_scores.items(), key=lambda x: x[1], reverse=True)
            },
            'condition_diversity': len(set(conditions.values())),
            'similar_conditions': self._group_similar_conditions(conditions)
        }
    
    def _group_similar_conditions(self, conditions: Dict[str, str]) -> Dict[str, List[str]]:
        """Group locations with similar weather conditions."""
        condition_groups = {}
        for location, condition in conditions.items():
            if condition not in condition_groups:
                condition_groups[condition] = []
            condition_groups[condition].append(location)
        
        return condition_groups
    
    def _generate_travel_recommendations(self, weather_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate travel recommendations based on weather comparison."""
        recommendations = {
            'outdoor_activities': [],
            'indoor_activities': [],
            'general_travel': []
        }
        
        for location, data in weather_data.items():
            current = data['current']
            comfort = data['comfort_indices']
            activity_recs = data.get('activity_recommendations', [])
            
            # Outdoor activity recommendations
            if comfort['comfort_score'] > 75 and current['main'] == 'Clear':
                recommendations['outdoor_activities'].append({
                    'location': location,
                    'score': comfort['comfort_score'],
                    'reason': f"Excellent conditions: {current['temperature']}°C, {current['description']}"
                })
            
            # Indoor activity recommendations
            if current['main'] in ['Rain', 'Snow'] or current['temperature'] < 5:
                recommendations['indoor_activities'].append({
                    'location': location,
                    'reason': f"Indoor activities recommended: {current['description']}, {current['temperature']}°C"
                })
            
            # General travel advice
            if activity_recs:
                top_rec = activity_recs[0]
                recommendations['general_travel'].append({
                    'location': location,
                    'suitability_score': top_rec['suitability_score'],
                    'activity_type': top_rec['type'],
                    'advice': top_rec['timing_advice']
                })
        
        # Sort recommendations by score where applicable
        recommendations['outdoor_activities'].sort(key=lambda x: x['score'], reverse=True)
        recommendations['general_travel'].sort(key=lambda x: x['suitability_score'], reverse=True)
        
        return recommendations
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get comprehensive service statistics for monitoring."""
        return {
            'cache_statistics': {
                'weather_cache': {
                    'size': len(self.weather_cache),
                    'maxsize': self.weather_cache.maxsize,
                    'ttl': self.weather_cache.ttl
                },
                'forecast_cache': {
                    'size': len(self.forecast_cache),
                    'maxsize': self.forecast_cache.maxsize,
                    'ttl': self.forecast_cache.ttl
                },
                'analytics_cache': {
                    'size': len(self.analytics_cache),
                    'maxsize': self.analytics_cache.maxsize,
                    'ttl': self.analytics_cache.ttl
                }
            },
            'api_usage': {
                'calls_today': self.api_calls_today,
                'rate_limit': Config.OPENWEATHER_RATE_LIMIT,
                'last_reset_date': self.last_reset_date.isoformat(),
                'calls_remaining': Config.OPENWEATHER_RATE_LIMIT - self.api_calls_today
            },
            'session_status': {
                'session_active': self.session is not None and not self.session.closed,
                'session_created': hasattr(self, 'session') and self.session is not None
            }
        }
    
    async def get_forecast(self, location: str, days: int = 5) -> Dict[str, Any]:
        """
        Get weather forecast (compatibility method at this stage).
        
        Args:
            location: England destination name
            days: Number of forecast days (1-5)
            
        Returns:
             forecast data for compatibility
        """
        # Use the full analytics forecast method
        full_forecast = await self.get_forecast_with_analytics(location, days)
        
        # Extract and simplify for backward compatibility
        simplified_forecast = {
            'location': location,
            'forecast_days': days,
            'days': []  # Use 'days' key to match IndraBot expectations
        }
        
        for day in full_forecast.get('days', []):
            simplified_day = {
                'date': day['date'],
                'day_name': day['day_name'],
                'temperature_min': day['temperature_min'],
                'temperature_max': day['temperature_max'],
                'dominant_condition': day['dominant_condition'],  # Use teh 'dominant_condition' key
                'condition': day['dominant_condition'],  # Keeping both of them for compatibility!
                'precipitation_chance': day['precipitation_chance'],
                'summary': day['summary']
            }
            simplified_forecast['days'].append(simplified_day)
        
        return simplified_forecast