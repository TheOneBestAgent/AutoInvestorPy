import functools
import logging
import time
from typing import Callable, Any, Type, Union, Tuple, Optional
import requests

logger = logging.getLogger(__name__)

class AutoInvestorError(Exception):
    """Base exception class for auto-investor"""
    pass

class DataError(AutoInvestorError):
    """Exception for data-related errors"""
    pass

class APIError(AutoInvestorError):
    """Exception for API-related errors"""
    pass

class TradingError(AutoInvestorError):
    """Exception for trading-related errors"""
    pass

class ConfigurationError(AutoInvestorError):
    """Exception for configuration-related errors"""
    pass

def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 2.0, 
                      exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """Decorator for retrying functions with exponential backoff"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries - 1:
                        logger.error(f"Function {func.__name__} failed after {max_retries} attempts: {e}")
                        raise
                    
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                except Exception as e:
                    # Don't retry for exceptions not in the exceptions tuple
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            
        return wrapper
    return decorator

def handle_api_errors(func: Callable) -> Callable:
    """Decorator for handling API errors gracefully"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except requests.exceptions.Timeout as e:
            logger.error(f"API timeout in {func.__name__}: {e}")
            raise APIError(f"API request timed out: {e}")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"API connection error in {func.__name__}: {e}")
            raise APIError(f"API connection failed: {e}")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error in {func.__name__}: {e}")
            raise APIError(f"HTTP error: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed in {func.__name__}: {e}")
            raise APIError(f"API request failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise
    
    return wrapper

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default

def safe_get_dict_value(dictionary: dict, key: str, default: Any = None) -> Any:
    """Safely get value from dictionary with error handling"""
    try:
        return dictionary.get(key, default)
    except (AttributeError, TypeError):
        return default

def log_and_reraise(exception: Exception, message: str = "", logger_obj: logging.Logger = logger):
    """Log an exception and re-raise it"""
    if message:
        logger_obj.error(f"{message}: {exception}", exc_info=True)
    else:
        logger_obj.error(f"Exception occurred: {exception}", exc_info=True)
    raise exception

class ErrorContext:
    """Context manager for handling errors in specific contexts"""
    
    def __init__(self, context_name: str, reraise: bool = True, 
                 logger_obj: logging.Logger = logger):
        self.context_name = context_name
        self.reraise = reraise
        self.logger = logger_obj
        self.exception_occurred = False
        self.exception = None
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.exception_occurred = True
            self.exception = exc_val
            
            self.logger.error(f"Error in {self.context_name}: {exc_val}", exc_info=True)
            
            if not self.reraise:
                return True  # Suppress the exception
        
        return False  # Let exception propagate

def handle_data_errors(func: Callable) -> Callable:
    """Decorator specifically for data-related error handling"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except (KeyError, AttributeError, IndexError) as e:
            logger.error(f"Data access error in {func.__name__}: {e}")
            raise DataError(f"Data access error: {e}")
        except (ValueError, TypeError) as e:
            logger.error(f"Data validation error in {func.__name__}: {e}")
            raise DataError(f"Data validation error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise
    
    return wrapper

def graceful_shutdown(cleanup_functions: list = None):
    """Gracefully handle shutdown and cleanup"""
    logger.info("Initiating graceful shutdown...")
    
    if cleanup_functions is not None:
        for cleanup_func in cleanup_functions:
            try:
                logger.info(f"Running cleanup function: {cleanup_func.__name__}")
                cleanup_func()
            except Exception as e:
                logger.error(f"Error in cleanup function {cleanup_func.__name__}: {e}")
    
    logger.info("Shutdown complete")

def validate_and_convert_numeric(value: Any, name: str, 
                                min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    """Validate and convert a value to numeric with range checking"""
    try:
        numeric_value = float(value)
        
        if min_val is not None and numeric_value < min_val:
            raise ValueError(f"{name} cannot be less than {min_val}")
        
        if max_val is not None and numeric_value > max_val:
            raise ValueError(f"{name} cannot be greater than {max_val}")
        
        return numeric_value
        
    except (ValueError, TypeError) as e:
        raise DataError(f"Invalid numeric value for {name}: {value} - {e}")

class CircuitBreaker:
    """Circuit breaker pattern for API calls"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func: Callable, *args, **kwargs):
        """Call function with circuit breaker protection"""
        if self.state == "OPEN":
            if self.last_failure_time is not None and time.time() - self.last_failure_time >= self.timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker moving to HALF_OPEN state")
            else:
                raise APIError("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise
    
    def on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            logger.info("Circuit breaker moving to CLOSED state")
    
    def on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning("Circuit breaker moving to OPEN state")
            raise APIError("Circuit breaker tripped - too many failures")