from pathlib import Path

# GenAI Core Framework
from genai_core.configs.loader import ConfigLoader
from genai_core.embeddings.factory import EmbeddingFactory
from genai_core.llms.factory import LLMFactory
from genai_core.prompts.manager import PromptManager

# RAG Research Lab
from chunkers.fixed_chunker import FixedChunker
from vectorstores.faiss_store import FAISSStore
from retrievers.vector_retriever import VectorRetriever
from pipelines.basic_rag_pipeline import BasicRAGPipeline


# Load Config
config = ConfigLoader.load(
    "configs/rag.yaml"
)

# Load models
llm = LLMFactory.create(
    provider=config.llm.provider,
    model=config.llm.model
)

embedding_model = EmbeddingFactory.create(
    provider=config.embeddings.provider,
    model=config.embeddings.model
)

# Load document
document = Path(
    "data/sample.text"
).read_text(
    encoding="utf-8"
)

# Chunk document
chunker = FixedChunker(
    chunk_size=256,
    chunk_overlap=25
)

chunks = chunker.chunk(
    document
)

print(
    f"Created {len(chunks)} chunks"
)

# create embeddings
embeddings = embedding_model.embed_documents(chunks)

# build vector store
vector_store = FAISSStore(
    dimension=len(embeddings[0])
)

vector_store.add_documents(
    embeddings, chunks
)

# Retriever
retriever = VectorRetriever(
    vector_store=vector_store,
    embedding_model=embedding_model,
    top_k=5
)

# Prompt manager
prompt_manager = PromptManager(template_dir="templates")

# RAG Pipeline
pipeline = BasicRAGPipeline(
    retriever=retriever,
    llm=llm,
    prompt_manager=prompt_manager
)

question = (
    "What is Retrieval-Augmented Generation?"
)

result = pipeline.generate(
    question
)

# Output
print("\nQUESTION")
print(question)

print("\nRETRIEVED CONTEXT")

for idx, context in enumerate(
    result["contexts"],
    start=1
):
    print(
        f"\nContext {idx}:"
    )
    print(context)

print("\nANSWER")
print(result["answer"])