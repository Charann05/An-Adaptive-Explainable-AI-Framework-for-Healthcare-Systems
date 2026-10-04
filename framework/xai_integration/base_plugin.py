from abc import ABC, abstractmethod

from .output_schema import ExplanationResult


class XAIPlugin(ABC):

    name: str = "UNNAMED"

    @abstractmethod
    def is_compatible(self, model_info: dict) -> bool:
        """
        Return True when this plugin can explain the supplied model.
        """
        raise NotImplementedError

    @abstractmethod
    def explain(
        self,
        model,
        input_data,
        model_info: dict,
        adapter=None,
    ) -> ExplanationResult:
        """
        Execute the XAI technique.
        """
        raise NotImplementedError