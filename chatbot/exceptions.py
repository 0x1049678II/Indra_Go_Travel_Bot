"""
Author: Cameron Murphy (Student ID: 1049678, GitHub: 0x1049678II)
Date: June 5th 2025

Custom exception classes for the Indra chatbot.

Comprehensive exception hierarchy for handling various failure modes in the chatbot.
Because when things go wrong (and they will), you want informative error messages.
"""

from typing import Optional, Dict, Any

class IndraException(Exception):
    """Base exception for all Indra chatbot errors."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            'error': self.error_code,
            'message': self.message,
            'details': self.details
        }

class WeatherAPIException(IndraException):
    """Base exception for weather API errors."""
    pass

class LocationNotFoundException(WeatherAPIException):
    """Raised when location is not found in England."""
    
    def __init__(self, location: str, valid_locations: Optional[list] = None):
        message = f"Location '{location}' is not supported. Please use one of the valid England locations."
        details = {'location': location}
        if valid_locations:
            details['valid_locations'] = valid_locations
        super().__init__(message, 'LOCATION_NOT_FOUND', details)

class RateLimitExceededException(IndraException):
    """Raised when API rate limit is exceeded."""
    
    def __init__(self, service: str, limit: int, reset_time: Optional[str] = None):
        message = f"Rate limit exceeded for {service}. Limit: {limit} requests."
        details = {'service': service, 'limit': limit}
        if reset_time:
            details['reset_time'] = reset_time
        super().__init__(message, 'RATE_LIMIT_EXCEEDED', details)

class WeatherDataNotFoundException(WeatherAPIException):
    """Raised when weather data is not available for a location."""
    
    def __init__(self, location: str, reason: str = "Data unavailable"):
        message = f"Weather data for '{location}' is currently unavailable: {reason}"
        details = {'location': location, 'reason': reason}
        super().__init__(message, 'WEATHER_DATA_NOT_FOUND', details)

class WeatherAPIConnectionException(WeatherAPIException):
    """Raised when unable to connect to weather API."""
    
    def __init__(self, api_name: str, status_code: Optional[int] = None, response_text: Optional[str] = None):
        message = f"Failed to connect to {api_name} API"
        details = {'api_name': api_name}
        if status_code:
            message += f" (Status: {status_code})"
            details['status_code'] = status_code
        if response_text:
            details['response'] = response_text[:200]  # Limit response text
        super().__init__(message, 'WEATHER_API_CONNECTION_ERROR', details)

class PredictionConfidenceException(IndraException):
    """Raised when prediction confidence is too low."""
    
    def __init__(self, confidence_score: float, threshold: float):
        message = f"Prediction confidence ({confidence_score:.2f}) is below threshold ({threshold:.2f})"
        details = {'confidence_score': confidence_score, 'threshold': threshold}
        super().__init__(message, 'PREDICTION_CONFIDENCE_LOW', details)

class NewsAPIException(IndraException):
    """Base exception for news API errors."""
    pass

class NewsDataNotFoundException(NewsAPIException):
    """Raised when news data is not available."""
    
    def __init__(self, query: str, reason: str = "No articles found"):
        message = f"News data not found for query '{query}': {reason}"
        details = {'query': query, 'reason': reason}
        super().__init__(message, 'NEWS_DATA_NOT_FOUND', details)

class NewsAPIConnectionException(NewsAPIException):
    """Raised when unable to connect to news API."""
    
    def __init__(self, status_code: Optional[int] = None, response_text: Optional[str] = None):
        message = "Failed to connect to news API"
        details = {}
        if status_code:
            message += f" (Status: {status_code})"
            details['status_code'] = status_code
        if response_text:
            details['response'] = response_text[:200]
        super().__init__(message, 'NEWS_API_CONNECTION_ERROR', details)

class ActivityServiceException(IndraException):
    """Base exception for activity service errors."""
    pass

class ActivityDataNotFoundException(ActivityServiceException):
    """Raised when activity data is not available for a location."""
    
    def __init__(self, location: str, weather_condition: Optional[str] = None):
        message = f"Activity recommendations not available for '{location}'"
        details = {'location': location}
        if weather_condition:
            message += f" with weather condition '{weather_condition}'"
            details['weather_condition'] = weather_condition
        super().__init__(message, 'ACTIVITY_DATA_NOT_FOUND', details)

class ChatterPyException(IndraException):
    """Base exception for ChatterPy-related errors."""
    pass

class ChatterPyTrainingException(ChatterPyException):
    """Raised when ChatterPy training fails."""
    
    def __init__(self, reason: str, training_data_path: Optional[str] = None):
        message = f"ChatterPy training failed: {reason}"
        details = {'reason': reason}
        if training_data_path:
            details['training_data_path'] = training_data_path
        super().__init__(message, 'CHATTERPY_TRAINING_ERROR', details)

class ChatterPyResponseException(ChatterPyException):
    """Raised when ChatterPy fails to generate a response."""
    
    def __init__(self, user_input: str, reason: str = "Unable to generate response"):
        message = f"ChatterPy failed to respond to: '{user_input[:50]}...'"
        details = {'user_input': user_input, 'reason': reason}
        super().__init__(message, 'CHATTERPY_RESPONSE_ERROR', details)

class DatabaseException(IndraException):
    """Base exception for database errors."""
    pass

class DatabaseConnectionException(DatabaseException):
    """Raised when unable to connect to database."""
    
    def __init__(self, database_url: str, reason: str):
        message = f"Failed to connect to database: {reason}"
        details = {'database_url': database_url, 'reason': reason}
        super().__init__(message, 'DATABASE_CONNECTION_ERROR', details)

class DatabaseOperationException(DatabaseException):
    """Raised when database operation fails."""
    
    def __init__(self, operation: str, table: str, reason: str):
        message = f"Database {operation} operation failed on table '{table}': {reason}"
        details = {'operation': operation, 'table': table, 'reason': reason}
        super().__init__(message, 'DATABASE_OPERATION_ERROR', details)

class CacheException(IndraException):
    """Base exception for caching errors."""
    pass

class CacheConnectionException(CacheException):
    """Raised when unable to connect to cache."""
    
    def __init__(self, cache_type: str, reason: str):
        message = f"Failed to connect to {cache_type} cache: {reason}"
        details = {'cache_type': cache_type, 'reason': reason}
        super().__init__(message, 'CACHE_CONNECTION_ERROR', details)

class CacheOperationException(CacheException):
    """Raised when cache operation fails."""
    
    def __init__(self, operation: str, key: str, reason: str):
        message = f"Cache {operation} operation failed for key '{key}': {reason}"
        details = {'operation': operation, 'key': key, 'reason': reason}
        super().__init__(message, 'CACHE_OPERATION_ERROR', details)

class ValidationException(IndraException):
    """Base exception for validation errors."""
    pass

class LocationValidationException(ValidationException):
    """Raised when location validation fails."""
    
    def __init__(self, location: str, validation_errors: list):
        message = f"Location '{location}' failed validation"
        details = {'location': location, 'validation_errors': validation_errors}
        super().__init__(message, 'LOCATION_VALIDATION_ERROR', details)

class InputValidationException(ValidationException):
    """Raised when user input validation fails."""
    
    def __init__(self, input_data: str, validation_errors: list):
        message = "User input validation failed"
        details = {'input_data': input_data[:100], 'validation_errors': validation_errors}
        super().__init__(message, 'INPUT_VALIDATION_ERROR', details)

class ConfigurationException(IndraException):
    """Base exception for configuration errors."""
    pass

class MissingAPIKeyException(ConfigurationException):
    """Raised when required API key is missing."""
    
    def __init__(self, api_name: str, env_var_name: str):
        message = f"Missing API key for {api_name}. Please set {env_var_name} environment variable."
        details = {'api_name': api_name, 'env_var_name': env_var_name}
        super().__init__(message, 'MISSING_API_KEY', details)

class InvalidConfigurationException(ConfigurationException):
    """Raised when configuration is invalid."""
    
    def __init__(self, config_key: str, config_value: Any, reason: str):
        message = f"Invalid configuration for '{config_key}': {reason}"
        details = {'config_key': config_key, 'config_value': str(config_value), 'reason': reason}
        super().__init__(message, 'INVALID_CONFIGURATION', details)

class AsyncOperationException(IndraException):
    """Base exception for async operation errors."""
    pass

class AsyncTimeoutException(AsyncOperationException):
    """Raised when async operation times out."""
    
    def __init__(self, operation: str, timeout: int):
        message = f"Async operation '{operation}' timed out after {timeout} seconds"
        details = {'operation': operation, 'timeout': timeout}
        super().__init__(message, 'ASYNC_TIMEOUT', details)

class ConcurrentRequestException(AsyncOperationException):
    """Raised when concurrent request limit is exceeded."""
    
    def __init__(self, current_requests: int, max_requests: int):
        message = f"Concurrent request limit exceeded: {current_requests}/{max_requests}"
        details = {'current_requests': current_requests, 'max_requests': max_requests}
        super().__init__(message, 'CONCURRENT_REQUEST_LIMIT', details)

# Exception handling utilities
class ExceptionHandler:
    """Utility class for handling exceptions consistently."""
    
    @staticmethod
    def handle_api_error(response, api_name: str):
        """Handle common API error responses."""
        if response.status_code == 401:
            raise MissingAPIKeyException(api_name, f"{api_name.upper()}_API_KEY")
        elif response.status_code == 429:
            raise RateLimitExceededException(api_name, limit=1000)  # Default limit
        elif response.status_code == 404:
            raise WeatherDataNotFoundException("Unknown", "API endpoint not found")
        elif response.status_code >= 500:
            raise WeatherAPIConnectionException(api_name, response.status_code, response.text)
        else:
            raise WeatherAPIConnectionException(api_name, response.status_code, response.text)
    
    @staticmethod
    def log_exception(logger, exception: IndraException, context: Dict[str, Any] = None):
        """Log exception with structured context."""
        log_data = {
            'exception_type': exception.__class__.__name__,
            'error_code': exception.error_code,
            'message': exception.message,
            'details': exception.details
        }
        if context:
            log_data['context'] = context
        
        logger.error("Indra exception occurred", **log_data)
    
    @staticmethod
    def get_user_friendly_message(exception: IndraException) -> str:
        """Convert technical exception to user-friendly message."""
        user_friendly_messages = {
            'LOCATION_NOT_FOUND': "I don't have information for that location. Please try one of the supported England destinations.",
            'RATE_LIMIT_EXCEEDED': "I'm receiving too many requests right now. Please try again in a few minutes.",
            'WEATHER_DATA_NOT_FOUND': "I couldn't get weather information for that location right now. Please try again later.",
            'WEATHER_API_CONNECTION_ERROR': "I'm having trouble accessing weather data. Please try again in a moment.",
            'NEWS_DATA_NOT_FOUND': "I couldn't find recent news for that topic. Would you like to try a different search?",
            'PREDICTION_CONFIDENCE_LOW': "I'm not confident enough in my weather prediction. Let me get you the current forecast instead.",
            'CHATTERPY_TRAINING_ERROR': "I'm still learning! Please be patient while I improve my responses.",
            'MISSING_API_KEY': "I'm having technical difficulties accessing external services. Please contact support.",
            'ASYNC_TIMEOUT': "That request is taking longer than expected. Please try again.",
        }
        
        return user_friendly_messages.get(
            exception.error_code, 
            "I encountered an unexpected issue. Please try rephrasing your question."
        )