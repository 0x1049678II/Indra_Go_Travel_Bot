"""
Author: Cameron Murphy (Student ID: 1049678, GitHub: 0x1049678II)
Date: June 6th 2025

Predictive analytics module for travel and weather insights.

Analytics engine that processes weather data to provide travel optimization
recommendations. Uses statistical analysis to predict optimal travel times and
provides comfort indices for different types of travelers. Because sometimes you
need more than just "it's going to rain" - you need "it's going to rain, but the
afternoon will be better for outdoor activities."
"""

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger()


class AnalyticsType(Enum):
    """Types of analytics available."""
    WEATHER_TREND = "weather_trend"
    TRAVEL_OPTIMIZATION = "travel_optimization"
    SEASONAL_ANALYSIS = "seasonal_analysis"
    ACTIVITY_PREDICTION = "activity_prediction"
    CROWD_PREDICTION = "crowd_prediction"


@dataclass
class TravelOptimization:
    """Travel optimization recommendation."""
    best_time: str
    confidence: float
    reasons: List[str]
    weather_score: float
    crowd_score: float
    overall_score: float
    alternative_times: List[Dict[str, Any]]


@dataclass
class SeasonalInsight:
    """Seasonal analysis insight."""
    season: str
    avg_temperature: float
    avg_precipitation: float
    crowd_level: str
    recommended_activities: List[str]
    pros: List[str]
    cons: List[str]


class TravelAnalytics:
    """
    Analytics for travel planning and optimization.
    Provides predictive insights based on weather, seasonality, and travel patterns.
    """
    
    def __init__(self):
        """Initialize the analytics engine."""
        self.seasonal_data = self._load_seasonal_baselines()
        self.crowd_patterns = self._load_crowd_patterns()
        logger.info("Travel analytics engine initialized")
    
    def _load_seasonal_baselines(self) -> Dict[str, Dict[str, Any]]:
        """Load seasonal baseline data for England destinations."""
        return {
            'spring': {
                'temp_range': (8, 16),
                'precipitation_chance': 45,
                'crowd_level': 'moderate',
                'peak_months': ['April', 'May'],
                'weather_reliability': 0.7
            },
            'summer': {
                'temp_range': (15, 22),
                'precipitation_chance': 35,
                'crowd_level': 'high',
                'peak_months': ['July', 'August'],
                'weather_reliability': 0.8
            },
            'autumn': {
                'temp_range': (6, 14),
                'precipitation_chance': 55,
                'crowd_level': 'moderate',
                'peak_months': ['September', 'October'],
                'weather_reliability': 0.6
            },
            'winter': {
                'temp_range': (2, 8),
                'precipitation_chance': 60,
                'crowd_level': 'low',
                'peak_months': ['December', 'January'],
                'weather_reliability': 0.5
            }
        }
    
    def _load_crowd_patterns(self) -> Dict[str, Dict[str, float]]:
        """Load crowd level patterns by location and time."""
        return {
            'Cambridge': {
                'weekday_multiplier': 0.7,
                'weekend_multiplier': 1.2,
                'university_term_multiplier': 1.5,
                'summer_multiplier': 1.8,
                'winter_multiplier': 0.6
            },
            'Oxford': {
                'weekday_multiplier': 0.8,
                'weekend_multiplier': 1.3,
                'university_term_multiplier': 1.4,
                'summer_multiplier': 1.9,
                'winter_multiplier': 0.5
            },
            'Bristol': {
                'weekday_multiplier': 0.9,
                'weekend_multiplier': 1.1,
                'university_term_multiplier': 1.0,
                'summer_multiplier': 1.4,
                'winter_multiplier': 0.7
            },
            'Norwich': {
                'weekday_multiplier': 0.6,
                'weekend_multiplier': 1.1,
                'university_term_multiplier': 1.0,
                'summer_multiplier': 1.2,
                'winter_multiplier': 0.5
            },
            'Birmingham': {
                'weekday_multiplier': 1.0,
                'weekend_multiplier': 1.0,
                'university_term_multiplier': 1.0,
                'summer_multiplier': 1.3,
                'winter_multiplier': 0.8
            },
            'Cumbria': {
                'weekday_multiplier': 0.5,
                'weekend_multiplier': 1.4,
                'university_term_multiplier': 1.0,
                'summer_multiplier': 2.0,
                'winter_multiplier': 0.3
            },
            'The Cotswolds': {
                'weekday_multiplier': 0.5,
                'weekend_multiplier': 1.5,
                'university_term_multiplier': 1.0,
                'summer_multiplier': 1.8,
                'winter_multiplier': 0.4
            },
            'Stonehenge': {
                'weekday_multiplier': 0.7,
                'weekend_multiplier': 1.6,
                'university_term_multiplier': 1.0,
                'summer_multiplier': 2.2,
                'winter_multiplier': 0.3
            },
            'Corfe Castle': {
                'weekday_multiplier': 0.6,
                'weekend_multiplier': 1.3,
                'university_term_multiplier': 1.0,
                'summer_multiplier': 1.6,
                'winter_multiplier': 0.4
            },
            'Watergate Bay': {
                'weekday_multiplier': 0.4,
                'weekend_multiplier': 1.7,
                'university_term_multiplier': 1.0,
                'summer_multiplier': 2.5,
                'winter_multiplier': 0.2
            }
        }
    
    def analyze_travel_timing(self, location: str, 
                             weather_forecast: List[Dict[str, Any]],
                             travel_preferences: Dict[str, Any] = None) -> TravelOptimization:
        """
        Analyze optimal travel timing based on weather and crowd predictions.
        
        Args:
            location: England destination
            weather_forecast: 5-day weather forecast data
            travel_preferences: User preferences (optional)
            
        Returns:
            Travel optimization recommendation
        """
        if not travel_preferences:
            travel_preferences = {
                'weather_priority': 0.7,
                'crowd_priority': 0.3,
                'preferred_activities': ['outdoor', 'sightseeing']
            }
        
        daily_scores = []
        
        for day_data in weather_forecast:
            weather_score = self._calculate_weather_score(day_data, travel_preferences)
            crowd_score = self._predict_crowd_level(location, day_data.get('date'))
            
            # Weighted overall score
            overall_score = (
                weather_score * travel_preferences.get('weather_priority', 0.7) +
                crowd_score * travel_preferences.get('crowd_priority', 0.3)
            )
            
            daily_scores.append({
                'date': day_data.get('date'),
                'day_name': day_data.get('day_name'),
                'weather_score': weather_score,
                'crowd_score': crowd_score,
                'overall_score': overall_score,
                'reasons': self._generate_score_reasons(day_data, weather_score, crowd_score)
            })
        
        # Handle edge case: no forecast data
        if not daily_scores:
            return TravelOptimization(
                best_time="No forecast data available",
                confidence=0.0,
                reasons=["Insufficient weather forecast data for analysis"],
                weather_score=0.0,
                crowd_score=0.0,
                overall_score=0.0,
                alternative_times=[]
            )
        
        # Find best day
        best_day = max(daily_scores, key=lambda x: x['overall_score'])
        
        # Generate alternatives
        alternatives = sorted(
            [day for day in daily_scores if day != best_day],
            key=lambda x: x['overall_score'],
            reverse=True
        )[:2]
        
        return TravelOptimization(
            best_time=f"{best_day['day_name']} ({best_day['date']})",
            confidence=min(best_day['overall_score'], 0.95),
            reasons=best_day['reasons'],
            weather_score=best_day['weather_score'],
            crowd_score=best_day['crowd_score'],
            overall_score=best_day['overall_score'],
            alternative_times=alternatives
        )
    
    def _calculate_weather_score(self, day_data: Dict[str, Any], 
                                preferences: Dict[str, Any]) -> float:
        """Calculate weather suitability score for a day."""
        temp_avg = day_data.get('temperature_avg', 15)
        condition = day_data.get('dominant_condition', 'Unknown')
        precip_chance = day_data.get('precipitation_chance', 50)
        wind_speed = day_data.get('wind_speed_avg', 10)
        
        # Base score from temperature (optimal around 18-20°C)
        temp_score = max(0, 1 - abs(temp_avg - 19) * 0.05)
        
        # Condition score
        condition_scores = {
            'Clear': 1.0,
            'Clouds': 0.8,
            'Partly Cloudy': 0.85,
            'Rain': 0.3,
            'Drizzle': 0.5,
            'Snow': 0.2,
            'Thunderstorm': 0.1
        }
        condition_score = condition_scores.get(condition, 0.6)
        
        # Precipitation penalty
        precip_score = max(0, 1 - precip_chance * 0.01)
        
        # Wind penalty
        wind_score = max(0.2, 1 - max(0, wind_speed - 10) * 0.05)
        
        # Weighted combination
        weather_score = (
            temp_score * 0.3 +
            condition_score * 0.4 +
            precip_score * 0.2 +
            wind_score * 0.1
        )
        
        return min(weather_score, 1.0)
    
    def _predict_crowd_level(self, location: str, date_str: str) -> float:
        """Predict crowd level for a specific location and date."""
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
        except:
            return 0.5  # Default moderate score
        
        crowd_data = self.crowd_patterns.get(location, {})
        base_score = 0.5
        
        # Weekend factor
        if date.weekday() >= 5:  # Saturday or Sunday
            base_score *= crowd_data.get('weekend_multiplier', 1.2)
        else:
            base_score *= crowd_data.get('weekday_multiplier', 0.8)
        
        # Seasonal factor
        month = date.month
        if month in [6, 7, 8]:  # Summer
            base_score *= crowd_data.get('summer_multiplier', 1.7)
        elif month in [12, 1, 2]:  # Winter
            base_score *= crowd_data.get('winter_multiplier', 0.6)
        
        # University term factor (rough approximation)
        if location in ['Cambridge', 'Oxford'] and month in [10, 11, 1, 2, 4, 5]:
            base_score *= crowd_data.get('university_term_multiplier', 1.4)
        
        # Convert to score (lower crowds = higher score)
        crowd_score = max(0.1, min(1.0, 1.5 - base_score))
        
        return crowd_score
    
    def _generate_score_reasons(self, day_data: Dict[str, Any], 
                               weather_score: float, crowd_score: float) -> List[str]:
        """Generate reasons for the calculated scores."""
        reasons = []
        
        # Weather reasons
        temp = day_data.get('temperature_avg', 15)
        condition = day_data.get('dominant_condition', 'Unknown')
        precip = day_data.get('precipitation_chance', 50)
        
        if weather_score > 0.8:
            reasons.append(f"Excellent weather: {temp}°C with {condition.lower()}")
        elif weather_score > 0.6:
            reasons.append(f"Good weather conditions: {temp}°C")
        else:
            reasons.append(f"Weather challenges: {condition.lower()}, {precip}% rain chance")
        
        # Crowd reasons
        if crowd_score > 0.7:
            reasons.append("Lower crowd levels expected")
        elif crowd_score > 0.5:
            reasons.append("Moderate crowd levels")
        else:
            reasons.append("Higher tourist activity expected")
        
        return reasons
    
    def analyze_seasonal_patterns(self, location: str) -> List[SeasonalInsight]:
        """
        Analyze seasonal patterns for a location.
        
        Args:
            location: England destination
            
        Returns:
            List of seasonal insights
        """
        insights = []
        
        for season, data in self.seasonal_data.items():
            temp_min, temp_max = data['temp_range']
            
            # Activity recommendations based on season
            activities = self._get_seasonal_activities(season, location)
            
            # Pros and cons
            pros, cons = self._get_seasonal_pros_cons(season, data)
            
            insight = SeasonalInsight(
                season=season.title(),
                avg_temperature=(temp_min + temp_max) / 2,
                avg_precipitation=data['precipitation_chance'],
                crowd_level=data['crowd_level'],
                recommended_activities=activities,
                pros=pros,
                cons=cons
            )
            
            insights.append(insight)
        
        return insights
    
    def _get_seasonal_activities(self, season: str, location: str) -> List[str]:
        """Get recommended activities for a season and location."""
        activity_map = {
            'spring': ['Garden visits', 'Walking tours', 'Outdoor photography', 'Castle exploration'],
            'summer': ['Punting', 'Outdoor markets', 'River walks', 'Cycling', 'Festivals'],
            'autumn': ['Heritage sites', 'Museums', 'Covered markets', 'Historic houses'],
            'winter': ['Museums', 'Art galleries', 'Theaters', 'Pubs', 'Shopping']
        }
        
        base_activities = activity_map.get(season, [])
        
        # Location-specific additions
        if location == 'Cambridge' and season in ['spring', 'summer']:
            base_activities.append('College tours')
        elif location == 'Bath' and season == 'winter':
            base_activities.append('Roman Baths')
        
        return base_activities
    
    def _get_seasonal_pros_cons(self, season: str, data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Get pros and cons for a season."""
        pros_cons_map = {
            'spring': {
                'pros': ['Mild temperatures', 'Blooming gardens', 'Moderate crowds'],
                'cons': ['Variable weather', 'Occasional rain showers']
            },
            'summer': {
                'pros': ['Warmest weather', 'Long daylight hours', 'Outdoor events'],
                'cons': ['High tourist crowds', 'Higher accommodation prices']
            },
            'autumn': {
                'pros': ['Beautiful foliage', 'Comfortable temperatures', 'Lower crowds'],
                'cons': ['Shorter days', 'Increased rainfall']
            },
            'winter': {
                'pros': ['Lowest crowds', 'Festive atmosphere', 'Indoor attractions'],
                'cons': ['Cold temperatures', 'Short daylight', 'Higher rain chance']
            }
        }
        
        season_data = pros_cons_map.get(season, {'pros': [], 'cons': []})
        return season_data['pros'], season_data['cons']
    
    def predict_activity_suitability(self, weather_data: Dict[str, Any], 
                                   activity_type: str) -> Dict[str, Any]:
        """
        Predict suitability for specific activity types.
        
        Args:
            weather_data: Current or forecast weather data
            activity_type: Type of activity ('outdoor', 'indoor', 'mixed')
            
        Returns:
            Suitability prediction with confidence and timing advice
        """
        current = weather_data.get('current', {})
        temperature = current.get('temperature', 15)
        condition = current.get('main', 'Unknown')
        wind_speed = current.get('wind_speed', 5)
        humidity = current.get('humidity', 60)
        
        if activity_type == 'outdoor':
            suitability = self._calculate_outdoor_suitability(
                temperature, condition, wind_speed, humidity
            )
        elif activity_type == 'indoor':
            suitability = self._calculate_indoor_suitability(
                temperature, condition, wind_speed
            )
        else:  # mixed
            outdoor_suit = self._calculate_outdoor_suitability(
                temperature, condition, wind_speed, humidity
            )
            indoor_suit = self._calculate_indoor_suitability(
                temperature, condition, wind_speed
            )
            suitability = max(outdoor_suit, indoor_suit * 0.8)
        
        # Generate timing advice
        timing_advice = self._generate_timing_advice(
            weather_data, activity_type, suitability
        )
        
        return {
            'activity_type': activity_type,
            'suitability_score': round(suitability, 2),
            'confidence': min(0.95, suitability * 1.1),
            'timing_advice': timing_advice,
            'weather_factors': {
                'temperature': temperature,
                'condition': condition,
                'wind_speed': wind_speed
            }
        }
    
    def _calculate_outdoor_suitability(self, temp: float, condition: str, 
                                     wind: float, humidity: float) -> float:
        """Calculate outdoor activity suitability."""
        # Temperature factor (optimal 15-22°C)
        if 15 <= temp <= 22:
            temp_factor = 1.0
        elif 10 <= temp <= 25:
            temp_factor = 0.8
        elif 5 <= temp <= 28:
            temp_factor = 0.6
        else:
            temp_factor = 0.3
        
        # Condition factor
        condition_factors = {
            'Clear': 1.0,
            'Clouds': 0.8,
            'Rain': 0.2,
            'Snow': 0.1,
            'Thunderstorm': 0.0
        }
        condition_factor = condition_factors.get(condition, 0.6)
        
        # Wind factor (penalty for strong winds)
        wind_factor = max(0.3, 1 - max(0, wind - 15) * 0.05)
        
        # Humidity factor (penalty for high humidity)
        humidity_factor = max(0.5, 1 - max(0, humidity - 70) * 0.01)
        
        return temp_factor * condition_factor * wind_factor * humidity_factor
    
    def _calculate_indoor_suitability(self, temp: float, condition: str, wind: float) -> float:
        """Calculate indoor activity suitability (inverse of outdoor for poor weather)."""
        # Indoor activities are more appealing in poor weather
        if condition in ['Rain', 'Snow', 'Thunderstorm']:
            return 0.9
        elif temp < 5 or temp > 28:
            return 0.8
        elif wind > 20:
            return 0.7
        else:
            return 0.6  # Always an option, but less appealing in good weather
    
    def _generate_timing_advice(self, weather_data: Dict[str, Any], 
                               activity_type: str, suitability: float) -> str:
        """Generate timing advice based on weather and activity suitability."""
        if suitability > 0.8:
            if activity_type == 'outdoor':
                return "Excellent conditions - perfect time for outdoor activities"
            else:
                return "Great time for indoor exploration"
        elif suitability > 0.6:
            return "Good conditions with minor weather considerations"
        elif suitability > 0.4:
            if activity_type == 'outdoor':
                return "Outdoor activities possible with appropriate preparation"
            else:
                return "Indoor activities recommended"
        else:
            if activity_type == 'outdoor':
                return "Indoor alternatives strongly recommended"
            else:
                return "Perfect weather for indoor activities"
    
    def generate_analytics_summary(self, location: str, 
                                  weather_forecast: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive analytics summary.
        
        Args:
            location: England destination
            weather_forecast: Multi-day weather forecast
            
        Returns:
            Complete analytics summary
        """
        # Travel optimization
        travel_opt = self.analyze_travel_timing(location, weather_forecast)
        
        # Seasonal insights
        seasonal_insights = self.analyze_seasonal_patterns(location)
        
        # Activity predictions for different types
        activity_predictions = []
        if weather_forecast:
            current_weather = {'current': weather_forecast[0]}
            for activity_type in ['outdoor', 'indoor', 'mixed']:
                prediction = self.predict_activity_suitability(current_weather, activity_type)
                activity_predictions.append(prediction)
        
        return {
            'location': location,
            'analysis_timestamp': datetime.now().isoformat(),
            'travel_optimization': {
                'best_time': travel_opt.best_time,
                'confidence': travel_opt.confidence,
                'reasons': travel_opt.reasons,
                'alternatives': travel_opt.alternative_times
            },
            'seasonal_insights': [
                {
                    'season': insight.season,
                    'avg_temperature': insight.avg_temperature,
                    'crowd_level': insight.crowd_level,
                    'recommended_activities': insight.recommended_activities,
                    'pros': insight.pros,
                    'cons': insight.cons
                }
                for insight in seasonal_insights
            ],
            'activity_predictions': activity_predictions,
            'analytics_confidence': 0.85
        }