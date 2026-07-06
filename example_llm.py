from genai_core.llms.factory import LLMFactory
from dotenv import load_dotenv
import os

load_dotenv()

llm = LLMFactory.create(
    provider="openai",
    model="gpt-4o",
    api_key=os.environ.get('OPENAI_API_KEY')
)

response = llm.generate(
    "What is RAG?"
)

print(response)