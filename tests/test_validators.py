"""
Author: Cameron Murphy
Date: June 14th 2025 (Student ID: 1049678, GitHub: 0x1049678II)

Test validation system functionality.
"""

import pytest
from unittest.mock import patch, MagicMock

from chatbot.validators import (
    EnglandLocationValidator, ValidationResult, LocationValidationResult,
    ValidationType, ValidationSeverity
)


class TestEnglandLocationValidator:
    """Test England location validation functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return EnglandLocationValidator()
    
    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator.locations is not None
        assert validator.valid_locations is not None
        assert len(validator.valid_locations) > 0
        assert validator.fuzzy_threshold == 0.6
        assert validator.high_confidence_threshold == 0.8
        assert len(validator.location_aliases) > 0
    
    def test_exact_location_match(self, validator):
        """Test exact location matching."""
        # Test exact match
        result = validator.validate_location("Cambridge")
        assert result.is_valid
        assert result.validated_location == "Cambridge"
        assert len(result.validation_issues) == 0
        
        # Test case insensitive match
        result_lower = validator.validate_location("cambridge")
        assert result_lower.is_valid
        assert result_lower.validated_location == "Cambridge"
    
    def test_alias_recognition(self, validator):
        """Test recognition of location aliases."""
        # Test common misspelling
        result = validator.validate_location("cambrige")
        assert result.is_valid
        assert result.validated_location == "Cambridge"
        assert len(result.validation_issues) > 0
        assert result.validation_issues[0].severity == ValidationSeverity.INFO
        
        # Test abbreviation
        result_short = validator.validate_location("camb")
        assert result_short.is_valid
        assert result_short.validated_location == "Cambridge"
    
    def test_fuzzy_matching_high_confidence(self, validator):
        """Test fuzzy matching with high confidence."""
        # Minor typo should get high confidence match
        result = validator.validate_location("Cambridg")
        assert result.is_valid
        assert result.validated_location == "Cambridge"
        assert any(issue.confidence > 0.8 for issue in result.validation_issues)
    
    def test_fuzzy_matching_low_confidence(self, validator):
        """Test fuzzy matching with low confidence."""
        # Major typos should get suggestions but not validate
        result = validator.validate_location("Camb123")
        assert not result.is_valid
        assert len(result.fuzzy_matches) > 0
        assert len(result.validation_issues) > 0
    
    def test_no_match_found(self, validator):
        """Test behavior when no match is found."""
        result = validator.validate_location("NonExistentPlace")
        assert not result.is_valid
        assert result.validated_location is None
        assert len(result.validation_issues) > 0
        assert result.validation_issues[0].severity == ValidationSeverity.ERROR
        assert len(result.validation_issues[0].suggestions) > 0
    
    def test_coordinate_validation_valid(self, validator):
        """Test coordinate validation for valid England coordinates."""
        # Cambridge coordinates
        result = validator.validate_coordinates(52.2053, 0.1218)
        assert result.is_valid
        assert result.severity == ValidationSeverity.INFO
        
        # London coordinates
        result_london = validator.validate_coordinates(51.5074, -0.1278)
        assert result_london.is_valid
    
    def test_coordinate_validation_invalid_latitude(self, validator):
        """Test coordinate validation for invalid latitude."""
        # Too far north (Scotland)
        result = validator.validate_coordinates(60.0, 0.0)
        assert not result.is_valid
        assert result.severity == ValidationSeverity.ERROR
        assert "latitude" in result.message.lower()
        
        # Too far south
        result_south = validator.validate_coordinates(40.0, 0.0)
        assert not result_south.is_valid
    
    def test_coordinate_validation_invalid_longitude(self, validator):
        """Test coordinate validation for invalid longitude."""
        # Too far east
        result = validator.validate_coordinates(52.0, 10.0)
        assert not result.is_valid
        assert result.severity == ValidationSeverity.ERROR
        assert "longitude" in result.message.lower()
        
        # Too far west
        result_west = validator.validate_coordinates(52.0, -10.0)
        assert not result_west.is_valid
    
    def test_coordinate_precision_warning(self, validator):
        """Test warning for overly precise coordinates."""
        # Very precise coordinates should trigger warning
        result = validator.validate_coordinates(52.2053123456789, 0.1218123456789)
        assert result.is_valid  # Still valid
        assert result.severity == ValidationSeverity.WARNING
        assert "precise" in result.message.lower()
    
    def test_similar_location_suggestions(self, validator):
        """Test similar location suggestions."""
        suggestions = validator.suggest_similar_locations("Oxfor", max_suggestions=3)
        assert len(suggestions) <= 3
        assert any("Oxford" in suggestion[0] for suggestion in suggestions)
        assert all(isinstance(score, float) for _, score in suggestions)
        assert all(0 <= score <= 1 for _, score in suggestions)
    
    def test_travel_query_validation_weather(self, validator):
        """Test travel query validation for weather queries."""
        query = "What's the weather like in Cambridge today?"
        result = validator.validate_travel_query(query)
        
        assert result['location_validation'].is_valid
        assert result['location_validation'].validated_location == "Cambridge"
        assert result['query_type'] == 'weather'
        assert 'weather' in result['weather_keywords']
        assert result['validation_summary']['location_valid']
        assert result['validation_summary']['has_weather_context']
    
    def test_travel_query_validation_activities(self, validator):
        """Test travel query validation for activity queries."""
        query = "What can I visit in Oxford this weekend?"
        result = validator.validate_travel_query(query)
        
        assert result['location_validation'].is_valid
        assert result['query_type'] == 'activities'
        assert 'visit' in result['activity_keywords']
        assert result['validation_summary']['has_activities']
    
    def test_travel_query_validation_news(self, validator):
        """Test travel query validation for news queries."""
        query = "Any events happening in Stonehenge?"
        result = validator.validate_travel_query(query)
        
        assert result['location_validation'].is_valid
        assert result['query_type'] == 'news'
        assert result['validation_summary']['location_valid']
    
    def test_location_extraction_from_text(self, validator):
        """Test location extraction from complex text."""
        candidates = validator._extract_location_candidates(
            "I'm planning to visit Cambridge and maybe Oxford next week"
        )
        assert "Cambridge" in candidates or "cambridge" in [c.lower() for c in candidates]
        assert "Oxford" in candidates or "oxford" in [c.lower() for c in candidates]
    
    def test_activity_keyword_extraction(self, validator):
        """Test activity keyword extraction."""
        query = "I want to visit museums and see castles in Cambridge"
        keywords = validator._extract_activity_keywords(query)
        assert 'visit' in keywords
        assert 'museum' in keywords
        assert 'castle' in keywords
    
    def test_time_indicator_extraction(self, validator):
        """Test time indicator extraction."""
        query = "What's the weather in London tomorrow morning?"
        indicators = validator._extract_time_indicators(query)
        assert 'tomorrow' in indicators
        assert 'morning' in indicators
    
    def test_weather_keyword_extraction(self, validator):
        """Test weather keyword extraction."""
        query = "Will it rain in York this sunny afternoon?"
        keywords = validator._extract_weather_keywords(query)
        assert 'rain' in keywords
        assert 'sunny' in keywords
    
    def test_query_type_determination(self, validator):
        """Test query type determination logic."""
        # Weather query
        weather_type = validator._determine_query_type(
            "weather forecast", [], ['weather', 'forecast']
        )
        assert weather_type == 'weather'
        
        # Activity query
        activity_type = validator._determine_query_type(
            "things to do", ['visit', 'see'], []
        )
        assert activity_type == 'activities'
        
        # News query
        news_type = validator._determine_query_type(
            "what's happening", [], []
        )
        assert news_type == 'news'
    
    def test_complex_location_variations(self, validator):
        """Test complex location name variations."""
        # Test "The" prefix
        result_cotswolds = validator.validate_location("the cotswolds")
        assert result_cotswolds.is_valid
        assert "Cotswolds" in result_cotswolds.validated_location
        
        # Test hyphenated names
        result_stratford = validator.validate_location("crofe castle")
        assert result_stratford.is_valid
        assert "Corfe Castle" in result_stratford.validated_location
    
    def test_case_insensitive_matching(self, validator):
        """Test case-insensitive matching."""
        test_cases = ["CAMBRIDGE", "cambridge", "CaMbRiDgE", "Cambridge"]
        
        for case in test_cases:
            result = validator.validate_location(case)
            assert result.is_valid
            assert result.validated_location == "Cambridge"
    
    def test_multiple_location_extraction(self, validator):
        """Test extraction when multiple locations mentioned."""
        query = "Should I visit Cambridge, Oxford, or Bath?"
        result = validator.validate_location(query)
        
        # Should pick one of the valid locations
        assert result.is_valid
        assert result.validated_location in ["Cambridge", "Oxford", "Bath"]
    
    def test_validation_statistics(self, validator):
        """Test validation statistics reporting."""
        stats = validator.get_validation_statistics()
        
        assert 'total_valid_locations' in stats
        assert 'location_aliases' in stats
        assert 'fuzzy_threshold' in stats
        assert 'high_confidence_threshold' in stats
        assert 'validation_patterns' in stats
        assert 'supported_location_types' in stats
        
        assert stats['total_valid_locations'] > 0
        assert stats['location_aliases'] > 0
        assert isinstance(stats['supported_location_types'], list)
    
    def test_partial_matches(self, validator):
        """Test partial word matching."""
        # Test partial words that should match
        result = validator.validate_location("Cam")  # Should suggest Cambridge
        
        if result.is_valid:
            assert result.validated_location == "Cambridge"
        else:
            # Should at least have Cambridge in suggestions
            assert len(result.fuzzy_matches) > 0
            assert any("Cambridge" in match[0] for match in result.fuzzy_matches)
    
    def test_whitespace_handling(self, validator):
        """Test whitespace handling in location names."""
        test_cases = [
            "  Cambridge  ",
            "\tCambridge\n",
            "Cambridge ",
            " Cambridge"
        ]
        
        for case in test_cases:
            result = validator.validate_location(case)
            assert result.is_valid
            assert result.validated_location == "Cambridge"
    
    def test_special_characters_handling(self, validator):
        """Test handling of special characters."""
        # Test common punctuation that might appear
        test_cases = [
            "Cambridge?",
            "Cambridge!",
            "Cambridge.",
            "Cambridge,",
            "(Cambridge)"
        ]
        
        for case in test_cases:
            result = validator.validate_location(case)
            # Should either validate or at least find Cambridge in suggestions
            if result.is_valid:
                assert result.validated_location == "Cambridge"
            else:
                assert any("Cambridge" in match[0] for match in result.fuzzy_matches)
    
    def test_empty_input_handling(self, validator):
        """Test handling of empty or invalid input."""
        # Empty string
        result_empty = validator.validate_location("")
        assert not result_empty.is_valid
        
        # Whitespace only
        result_whitespace = validator.validate_location("   ")
        assert not result_whitespace.is_valid
        
        # Very short input
        result_short = validator.validate_location("a")
        assert not result_short.is_valid


def test_manual_location_validation():
    """Manual test for location validation"""
    try:
        validator = EnglandLocationValidator()
        
        # Test some real scenarios
        test_cases = [
            "Cambridge",
            "cambrige",  # misspelling
            "Oxford", 
            "What's the weather in the cotswolds?",
            "I want to visit Stonehenge",
            "NonExistentPlace"
        ]
        
        for test_case in test_cases:
            result = validator.validate_location(test_case)
            print(f"Input: '{test_case}' -> Valid: {result.is_valid}, Location: {result.validated_location}")
            
            if result.fuzzy_matches:
                print(f"  Fuzzy matches: {result.fuzzy_matches[:3]}")
            
            if result.validation_issues:
                for issue in result.validation_issues:
                    print(f"  Issue: {issue.severity.value} - {issue.message}")
        
        print("SUCCESS: Manual location validation test completed")
        
    except Exception as e:
        print(f"FAILED: Manual location validation test failed - {e}")
        raise


if __name__ == "__main__":
    # Run manual test if called directly
    test_manual_location_validation()