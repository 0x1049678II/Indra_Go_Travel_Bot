"""
Author: Cameron Murphy (Student ID: 1049678, GitHub: 0x1049678II)
Date: June 10th 2025

Utility functions and decorators for the Indra chatbot.

This module contains the utilities for the Indra chatbot. It contains all
the helpful functions that don't quite fit anywhere else but are essential
for making everything work smoothly.

The utilities here handle async operations, retries, performance monitoring,
and various other tasks.

Features:
    - Async retry decorators with exponential backoff
    - Performance monitoring and metrics collection
    - Safe async gathering with error handling
    - Async timeout decorators
    - Rate limiting utilities
    - Sync-to-async conversion helpers

These utilities are designed to handle the messiness of network
requests, API failures, and the general unpredictability of distributed systems.
"""

import asyncio
import functools
import time
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()

T = TypeVar('T')


@dataclass
class PerformanceMetrics:
    """Performance metrics for async operations."""
    execution_time: float
    success: bool
    error_type: Optional[str]
    retry_count: int


def async_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Async retry decorator - for when at first you don't succeed, try, try again.
    
    This decorator wraps async functions with retry logic and exponential backoff.
    It's useful for API calls that might fail due to network issues, rate limiting,
    or the general unreliability of the internet.
    
    The decorator implements exponential backoff to avoid hammering failing services
    and includes  logging so you can see what's happening when things go wrong.
    
    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        delay: Initial delay between retries in seconds (default: 1.0)
        backoff: Backoff multiplier for delays (default: 2.0)
        
    Returns:
        Decorated async function with retry logic
        
    Example:
        @async_retry(max_retries=5, delay=0.5, backoff=1.5)
        async def fetch_weather_data(location):
            # This will retry up to 5 times if it fails
            return await api_call(location)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            "Async operation failed after retries",
                            function=func.__name__,
                            attempts=attempt + 1,
                            error=str(e)
                        )
                        raise e
                    
                    logger.warning(
                        "Async operation retry",
                        function=func.__name__,
                        attempt=attempt + 1,
                        delay=current_delay,
                        error=str(e)
                    )
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            # This should never be reached, but for type safety
            if last_exception:
                raise last_exception
            return None

        return wrapper
    return decorator


def async_timeout(seconds: float):
    """
    Decorator to add timeout to async functions.
    
    Args:
        seconds: Timeout in seconds
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(
                    "Async operation timed out",
                    function=func.__name__,
                    timeout=seconds
                )
                raise
        return wrapper
    return decorator


def performance_monitor(log_performance: bool = True):
    """
    Decorator to monitor async function performance.
    
    Args:
        log_performance: Whether to log performance metrics
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            start_time = time.time()
            success = False
            error_type = None
            
            try:
                result = await func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                error_type = type(e).__name__
                raise
            finally:
                execution_time = time.time() - start_time
                
                if log_performance:
                    logger.info(
                        "Async operation performance",
                        function=func.__name__,
                        execution_time=round(execution_time, 3),
                        success=success,
                        error_type=error_type
                    )
        
        return wrapper
    return decorator


async def safe_gather(*aws, return_exceptions: bool = True) -> list:
    """
    Safe wrapper around asyncio.gather that provides better error handling.
    
    Args:
        *aws: Awaitable objects
        return_exceptions: Whether to return exceptions instead of raising
        
    Returns:
        List of results or exceptions
    """
    try:
        results = await asyncio.gather(*aws, return_exceptions=return_exceptions)
        
        if return_exceptions:
            # Log any exceptions found
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(
                        "Exception in concurrent operation",
                        operation_index=i,
                        error=str(result),
                        error_type=type(result).__name__
                    )
        
        return results
    except Exception as e:
        logger.error("Failed to execute concurrent operations", error=str(e))
        raise


def create_async_session_manager(session_factory: Callable):
    """
    Create an async session manager for resource lifecycle management.
    
    Args:
        session_factory: Function that creates the session object
    """
    class AsyncSessionManager:
        def __init__(self):
            self.session = None
            self.factory = session_factory
        
        async def __aenter__(self):
            self.session = await self.factory()
            return self.session
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if self.session and hasattr(self.session, 'close'):
                await self.session.close()
            elif self.session and hasattr(self.session, '__aexit__'):
                await self.session.__aexit__(exc_type, exc_val, exc_tb)
    
    return AsyncSessionManager()


def validate_async_inputs(**validators):
    """
    Decorator to validate inputs for async functions.
    
    Args:
        **validators: Dict of parameter_name: validation_function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each specified parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"Invalid value for parameter '{param_name}': {value}")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def rate_limit(calls_per_second: float):
    """
    Decorator to rate limit async function calls.
    
    Args:
        calls_per_second: Maximum calls allowed per second
    """
    min_interval = 1.0 / calls_per_second
    last_called = {}
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            now = time.time()
            func_key = id(func)
            
            if func_key in last_called:
                elapsed = now - last_called[func_key]
                if elapsed < min_interval:
                    wait_time = min_interval - elapsed
                    await asyncio.sleep(wait_time)
            
            last_called[func_key] = time.time()
            return await func(*args, **kwargs)
        return wrapper
    return decorator


class AsyncCache:
    """Simple async-aware cache with TTL support."""
    
    def __init__(self, default_ttl: float = 300.0):
        """
        Initialize async cache.
        
        Args:
            default_ttl: Default time-to-live in seconds
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() < entry['expires']:
                return entry['value']
            else:
                del self.cache[key]
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache with TTL."""
        if ttl is None:
            ttl = self.default_ttl
        
        self.cache[key] = {
            'value': value,
            'expires': time.time() + ttl
        }
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = time.time()
        active_entries = sum(1 for entry in self.cache.values() if entry['expires'] > now)
        expired_entries = len(self.cache) - active_entries
        
        return {
            'total_entries': len(self.cache),
            'active_entries': active_entries,
            'expired_entries': expired_entries,
            'hit_ratio': getattr(self, '_hit_count', 0) / max(getattr(self, '_access_count', 1), 1)
        }


def sync_to_async(func: Callable) -> Callable:
    """
    Convert a synchronous function to async using thread executor.
    
    Args:
        func: Synchronous function to convert
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
    return wrapper


def handle_async_errors(default_return: Any = None, log_errors: bool = True):
    """
    Decorator to handle async function errors gracefully.
    
    Args:
        default_return: Value to return on error
        log_errors: Whether to log errors
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Union[T, Any]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Union[T, Any]:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(
                        "Async function error handled",
                        function=func.__name__,
                        error=str(e),
                        error_type=type(e).__name__
                    )
                return default_return
        return wrapper
    return decorator


class AsyncContextualLogger:
    """Async-aware contextual logger for request tracking."""
    
    def __init__(self, base_logger=None):
        self.base_logger = base_logger or logger
        self.context = {}
    
    def add_context(self, **kwargs):
        """Add context to all subsequent log messages."""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear all context."""
        self.context.clear()
    
    def log(self, level: str, message: str, **kwargs):
        """Log message with context."""
        combined_context = {**self.context, **kwargs}
        getattr(self.base_logger, level)(message, **combined_context)
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self.log('info', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context."""
        self.log('error', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self.log('warning', message, **kwargs)


# Global instances for common use
global_cache = AsyncCache()
contextual_logger = AsyncContextualLogger()


def format_response_time(seconds: float) -> str:
    """Format response time for human readability."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f}μs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    else:
        return f"{seconds:.2f}s"


def create_task_with_logging(coro, name: str = None):
    """Create asyncio task with automatic logging."""
    if name is None:
        name = coro.__name__ if hasattr(coro, '__name__') else 'unnamed_task'
    
    task = asyncio.create_task(coro, name=name)
    
    def log_completion(task):
        if task.exception():
            logger.error(
                "Async task failed",
                task_name=name,
                error=str(task.exception())
            )
        else:
            logger.info("Async task completed", task_name=name)
    
    task.add_done_callback(log_completion)
    return task