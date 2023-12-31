

from abc import ABC, abstractmethod

class BaseModel(ABC):

    """A model class to lead the model and tokenizer"""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def load_model():
        pass

    @abstractmethod
    def load_tokenizer():
        pass