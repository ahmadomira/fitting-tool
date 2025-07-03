"""
Base class for all fitting algorithms in the fitting_app.
Defines the interface and shared logic for fitting routines.
"""
import abc

class BaseFittingAlgorithm(abc.ABC):
    """Abstract base class for fitting algorithms."""

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """Run the fitting routine. Must be implemented by subclasses."""
        pass

    @abc.abstractmethod
    def load_data(self, file_path):
        """Load and preprocess data from a file."""
        pass

    @abc.abstractmethod
    def export_results(self, *args, **kwargs):
        """Export fitting results to file or other formats."""
        pass
