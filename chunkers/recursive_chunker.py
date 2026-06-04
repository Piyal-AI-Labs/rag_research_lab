#Langchain's recursive splitting

from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base import BaseChunker

class RecursiveChunker(BaseChunker):

    def __init__(
        self,
        chunk_size = 512,
        chunk_overlap = 50
    ):
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def chunk(
        self,
        text: str
    ) -> list[str]:
        
        return self.splitter.split_text(text)