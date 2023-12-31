from abc import ABC, abstractmethod

from .base_model import BaseModel


class BaseServer(ABC):

    """A server class to lead the model and tokenizer"""

    def __init__(self,
                model: BaseModel,
                 ) -> None:
        self.model = model.load_model()
        self.tokenizer = model.load_tokenizer()

    @abstractmethod
    def serve(self):
        pass

