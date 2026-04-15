from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

class TextSplitter:
    def __init__(self):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.splitter = SemanticChunker(embeddings)

    def split(self, text: str):
        return self.splitter.split_text(text)