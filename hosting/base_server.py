import sys
import logging
from abc import ABC, abstractmethod
from typing import Any
from time import perf_counter
from .base_model import BaseModel
# Set up logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

class BaseServer(ABC):

    """A server class to lead the model and tokenizer"""

    request_schema: Any = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "minLength": 1, "maxLength": 1000},
        "labels": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 10},
            "minItems": 1,
        },
    },
        "required": ["text", "labels"],
    }

    def __init__(self,
                model: BaseModel,
                request_schema: Any = None,
                 ) -> None:
        self.model = model
        if request_schema is not None:
            self.request_schema = request_schema


    @abstractmethod
    def health_check(self) -> Any:
        """Health check endpoint"""
        self.model.load()
        return {"status": "ok"}


    @abstractmethod
    def predict(self, input: Any, **kwargs) -> Any:
        result = self.model.predict(input)
        return result
