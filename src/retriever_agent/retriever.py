from typing import List
from src.dataclass import RetrievedDocument


class Retriever:
    def __init__(self):
        pass

    def aretrieve(self, query: str, **kwargs) -> List[RetrievedDocument]:
        raise NotImplementedError("Subclasses must implement the aretrieve method.")
