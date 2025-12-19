from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type


class DiamondPricePredictorError(Exception):
    """Base exception for the Diamond Price Predictor."""

    def __init__(self, message: str, suggestion: Optional[str] = None):
        self.message = message
        self.suggestion = suggestion
        super().__init__(message)


class InputValidationError(DiamondPricePredictorError):
    """Exception for input validation errors."""

    def __init__(
        self,
        message: str = "Input validation failed.",
        suggestion: str = "Please check the input values and try again.",
        errors: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, suggestion)
        self.errors = errors


class ModelNotFoundError(DiamondPricePredictorError):
    """Exception for when the model is not found."""

    def __init__(
        self,
        message: str = "Model not found.",
        suggestion: str = "Please train the model before running the application.",
    ):
        super().__init__(message, suggestion)


class PredictionError(DiamondPricePredictorError):
    """Exception for prediction errors."""

    def __init__(
        self,
        message: str = "An error occurred during prediction.",
        suggestion: str = "Please try again later or contact support.",
    ):
        super().__init__(message, suggestion)


# --- Error Reporter and Report Generator Interfaces ---
class ErrorReport:
    """A simple dataclass to hold error report information."""

    def __init__(
        self, error_type: str, message: str, suggestion: Optional[str] = None
    ):
        self.error_type = error_type
        self.message = message
        self.suggestion = suggestion

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "suggestion": self.suggestion,
        }


class ErrorReportGenerator(ABC):
    """Abstract base class for error report generators."""

    @abstractmethod
    def generate_report(self, error: Exception) -> ErrorReport:
        """Generates an error report from an exception."""
        pass


class DiamondPricePredictorReportGenerator(ErrorReportGenerator):
    """Report generator for DiamondPricePredictor exceptions."""

    def generate_report(self, error: Exception) -> ErrorReport:
        if isinstance(error, DiamondPricePredictorError):
            return ErrorReport(
                error_type=error.__class__.__name__,
                message=error.message,
                suggestion=error.suggestion,
            )
        # Fallback for unexpected errors
        return ErrorReport(
            error_type="UnhandledException",
            message="An unexpected error occurred.",
            suggestion="Please contact support for assistance.",
        )


class ErrorReporter:
    """A class to report errors using a specific report generator."""

    def __init__(self, report_generator: ErrorReportGenerator):
        self.report_generator = report_generator

    def report(self, error: Exception) -> ErrorReport:
        """Reports an error."""
        return self.report_generator.generate_report(error)
