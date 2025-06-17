"""
Author: Cameron Murphy (Student ID: 1049678, GitHub: 0x1049678II)
Date: June 14th 2025

Test analytics module functionality.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from chatbot.analytics import TravelAnalytics, AnalyticsType, TravelOptimization, SeasonalInsight


class TestTravelAnalytics:
    """Test travel analytics functionality."""
    
    @pytest.fixture
    def analytics(self):
        """Create analytics instance."""
        return TravelAnalytics()
    
    @pytest.fixture
    def sample_weather_forecast(self):
        """Sample weather forecast data."""
        return [
            {
                'date': '2025-06-16',
                'day_name': 'Monday',
                'temperature_avg': 18,
                'dominant_condition': 'Clear',
                'precipitation_chance': 10,
                'wind_speed_avg': 8
            },
            {
                'date': '2025-06-17',
                'day_name': 'Tuesday',
                'temperature_avg': 22,
                'dominant_condition': 'Clouds',
                'precipitation_chance': 30,
                'wind_speed_avg': 12
            },
            {
                'date': '2025-06-18',
                'day_name': 'Wednesday',
                'temperature_avg': 15,
                'dominant_condition': 'Rain',
                'precipitation_chance': 80,
                'wind_speed_avg': 15
            }
        ]
    
    def test_analytics_initialization(self, analytics):
        """Test analytics engine initialization."""
        assert analytics.seasonal_data is not None
        assert analytics.crowd_patterns is not None
        assert len(analytics.seasonal_data) == 4  # 4 seasons
        assert 'Cambridge' in analytics.crowd_patterns
    
    def test_weather_score_calculation(self, analytics):
        """Test weather score calculation."""
        day_data = {
            'temperature_avg': 20,
            'dominant_condition': 'Clear',
            'precipitation_chance': 10,
            'wind_speed_avg': 5
        }
        preferences = {'weather_priority': 0.7}
        
        score = analytics._calculate_weather_score(day_data, preferences)
        assert 0.8 <= score <= 1.0  # Should be high for perfect conditions
    
    def test_weather_score_poor_conditions(self, analytics):
        """Test weather score for poor conditions."""
        day_data = {
            'temperature_avg': 2,
            'dominant_condition': 'Rain',
            'precipitation_chance': 90,
            'wind_speed_avg': 25
        }
        preferences = {'weather_priority': 0.7}
        
        score = analytics._calculate_weather_score(day_data, preferences)
        assert score <= 0.4  # Should be low for poor conditions
    
    def test_crowd_level_prediction(self, analytics):
        """Test crowd level prediction."""
        # Monday (weekday) in Cambridge
        score_weekday = analytics._predict_crowd_level('Cambridge', '2025-06-16')
        
        # Saturday (weekend) in Cambridge
        score_weekend = analytics._predict_crowd_level('Cambridge', '2025-06-21')
        
        assert score_weekday > score_weekend  # Lower crowds on weekdays
    
    def test_travel_timing_analysis(self, analytics, sample_weather_forecast):
        """Test travel timing analysis."""
        result = analytics.analyze_travel_timing('Cambridge', sample_weather_forecast)
        
        assert isinstance(result, TravelOptimization)
        assert result.best_time is not None
        assert 0 <= result.confidence <= 1
        assert len(result.reasons) > 0
        assert len(result.alternative_times) <= 2
    
    def test_seasonal_analysis(self, analytics):
        """Test seasonal pattern analysis."""
        insights = analytics.analyze_seasonal_patterns('Cambridge')
        
        assert len(insights) == 4  # 4 seasons
        assert all(isinstance(insight, SeasonalInsight) for insight in insights)
        
        # Check that summer has higher temperatures than winter
        summer = next(i for i in insights if i.season == 'Summer')
        winter = next(i for i in insights if i.season == 'Winter')
        assert summer.avg_temperature > winter.avg_temperature
    
    def test_outdoor_activity_suitability(self, analytics):
        """Test outdoor activity suitability calculation."""
        # Perfect conditions
        perfect_score = analytics._calculate_outdoor_suitability(20, 'Clear', 5, 50)
        assert perfect_score >= 0.8
        
        # Poor conditions
        poor_score = analytics._calculate_outdoor_suitability(2, 'Rain', 25, 90)
        assert poor_score <= 0.3
    
    def test_indoor_activity_suitability(self, analytics):
        """Test indoor activity suitability calculation."""
        # Poor outdoor weather should make indoor activities more appealing
        rainy_score = analytics._calculate_indoor_suitability(15, 'Rain', 10)
        sunny_score = analytics._calculate_indoor_suitability(20, 'Clear', 5)
        
        assert rainy_score > sunny_score
    
    def test_activity_prediction(self, analytics):
        """Test activity suitability prediction."""
        weather_data = {
            'current': {
                'temperature': 18,
                'main': 'Clear',
                'wind_speed': 8,
                'humidity': 60
            }
        }
        
        # Test outdoor activity prediction
        outdoor_pred = analytics.predict_activity_suitability(weather_data, 'outdoor')
        assert 'suitability_score' in outdoor_pred
        assert 'timing_advice' in outdoor_pred
        assert outdoor_pred['activity_type'] == 'outdoor'
        
        # Test indoor activity prediction
        indoor_pred = analytics.predict_activity_suitability(weather_data, 'indoor')
        assert indoor_pred['activity_type'] == 'indoor'
    
    def test_timing_advice_generation(self, analytics):
        """Test timing advice generation."""
        weather_data = {
            'current': {'temperature': 20, 'main': 'Clear'}
        }
        
        # High suitability advice
        advice_high = analytics._generate_timing_advice(weather_data, 'outdoor', 0.9)
        assert 'excellent' in advice_high.lower() or 'perfect' in advice_high.lower()
        
        # Low suitability advice
        advice_low = analytics._generate_timing_advice(weather_data, 'outdoor', 0.2)
        assert 'indoor' in advice_low.lower()
    
    def test_comprehensive_analytics_summary(self, analytics, sample_weather_forecast):
        """Test comprehensive analytics summary generation."""
        summary = analytics.generate_analytics_summary('Cambridge', sample_weather_forecast)
        
        assert 'location' in summary
        assert 'travel_optimization' in summary
        assert 'seasonal_insights' in summary
        assert 'activity_predictions' in summary
        assert 'analytics_confidence' in summary
        
        # Check travel optimization structure
        travel_opt = summary['travel_optimization']
        assert 'best_time' in travel_opt
        assert 'confidence' in travel_opt
        assert 'reasons' in travel_opt
        
        # Check seasonal insights structure
        seasonal = summary['seasonal_insights']
        assert len(seasonal) == 4
        assert all('season' in insight for insight in seasonal)
        
        # Check activity predictions
        activities = summary['activity_predictions']
        assert len(activities) == 3  # outdoor, indoor, mixed
    
    def test_seasonal_activities_by_location(self, analytics):
        """Test location-specific seasonal activities."""
        cambridge_spring = analytics._get_seasonal_activities('spring', 'Cambridge')
        bath_winter = analytics._get_seasonal_activities('winter', 'Bath')
        
        # Cambridge should have college tours in spring
        assert any('college' in activity.lower() for activity in cambridge_spring)
        
        # Bath should have Roman Baths in winter
        assert any('roman' in activity.lower() for activity in bath_winter)
    
    def test_score_reasons_generation(self, analytics):
        """Test score reasons generation."""
        day_data = {
            'temperature_avg': 25,
            'dominant_condition': 'Clear',
            'precipitation_chance': 5
        }
        
        reasons = analytics._generate_score_reasons(day_data, 0.9, 0.8)
        assert len(reasons) >= 2  # Should have weather and crowd reasons
        assert any('weather' in reason.lower() for reason in reasons)
    
    def test_travel_preferences_impact(self, analytics, sample_weather_forecast):
        """Test impact of travel preferences on recommendations."""
        # Weather-focused preferences
        weather_pref = {'weather_priority': 0.9, 'crowd_priority': 0.1}
        result_weather = analytics.analyze_travel_timing('Cambridge', sample_weather_forecast, weather_pref)
        
        # Crowd-focused preferences
        crowd_pref = {'weather_priority': 0.1, 'crowd_priority': 0.9}
        result_crowd = analytics.analyze_travel_timing('Cambridge', sample_weather_forecast, crowd_pref)
        
        # Results may differ based on preferences
        assert result_weather.best_time is not None
        assert result_crowd.best_time is not None
    
    def test_edge_cases(self, analytics):
        """Test edge cases and error handling."""
        # Empty forecast
        empty_result = analytics.analyze_travel_timing('Cambridge', [])
        assert isinstance(empty_result, TravelOptimization)
        
        # Invalid date format
        invalid_crowd = analytics._predict_crowd_level('Cambridge', 'invalid-date')
        assert 0 <= invalid_crowd <= 1
        
        # Unknown location
        unknown_crowd = analytics._predict_crowd_level('UnknownLocation', '2025-06-16')
        assert 0 <= unknown_crowd <= 1
    
    def test_seasonal_pros_cons(self, analytics):
        """Test seasonal pros and cons generation."""
        spring_pros, spring_cons = analytics._get_seasonal_pros_cons('spring', {})
        summer_pros, summer_cons = analytics._get_seasonal_pros_cons('summer', {})
        
        assert len(spring_pros) > 0
        assert len(spring_cons) > 0
        assert len(summer_pros) > 0
        assert len(summer_cons) > 0
        
        # Summer should mention crowds as a con
        assert any('crowd' in con.lower() for con in summer_cons)


if __name__ == "__main__":
    # Run basic functionality test
    analytics = TravelAnalytics()
    
    sample_forecast = [
        {
            'date': '2025-06-16',
            'day_name': 'Monday',
            'temperature_avg': 18,
            'dominant_condition': 'Clear',
            'precipitation_chance': 10,
            'wind_speed_avg': 8
        }
    ]
    
    try:
        # Test basic functionality
        result = analytics.analyze_travel_timing('Cambridge', sample_forecast)
        print(f"SUCCESS: Travel timing analysis completed - {result.best_time}")
        
        seasonal = analytics.analyze_seasonal_patterns('Cambridge')
        print(f"SUCCESS: Seasonal analysis completed - {len(seasonal)} seasons analyzed")
        
        weather_data = {'current': {'temperature': 18, 'main': 'Clear', 'wind_speed': 8, 'humidity': 60}}
        activity = analytics.predict_activity_suitability(weather_data, 'outdoor')
        print(f"SUCCESS: Activity prediction completed - {activity['suitability_score']} score")
        
        summary = analytics.generate_analytics_summary('Cambridge', sample_forecast)
        print(f"SUCCESS: Analytics summary generated - confidence {summary['analytics_confidence']}")
        
    except Exception as e:
        print(f"FAILED: Analytics test failed - {e}")
        raise