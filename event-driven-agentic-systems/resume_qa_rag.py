import os
from dotenv import load_dotenv
from llama_index.core.workflow import (
    Event,
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Context,
)
from llama_index.llms.azure_openai import AzureOpenAI, AsyncAzureOpenAI  # noqa
import asyncio
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.tools import FunctionTool  # noqa
from llama_index.core.agent import FunctionCallingAgent  # noqa
from pprint import pprint

load_dotenv()

llm = AzureOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-07-01-preview",
    temperature=0.1,
    max_tokens=1024,
)

embedding_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-07-01-preview",
)


class QueryEvent(Event):
    query: str


class RAGWorkflow(Workflow):
    storage_dir = "./storage"
    llm = llm
    query_engine = VectorStoreIndex

    @step
    async def set_up(self, ctx: Context, event: StartEvent) -> QueryEvent:
        if not event.resume_file:
            raise ValueError("No resume file provided. Resume file is required")

        self.llm = llm

        if os.path.exists(self.storage_dir):
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            index = load_index_from_storage(storage_context)
        else:
            documents = LlamaParse(
                api_key=os.getenv("LLAMA_INDEX_API_KEY"),
                result_type="markdown",
                content_guideline_instruction="This is a resume, gather related facts together and format it as bullet points with headers",
            ).load_data(event.resume_file)
            index = VectorStoreIndex.from_documents(documents, embedding_model)
            index.storage_context.persist(persist_dir=self.storage_dir)

        self.query_engine = index.as_query_engine(llm=self.llm, similarity_top_k=3)

        return QueryEvent(query=event.query)

    @step
    async def fetch_query_response(self, ctx: Context, event: QueryEvent) -> StopEvent:
        response = self.query_engine.query(
            f"Answer the question related to the specific resume we have in our database: {event.query}"
        )
        return StopEvent(result=response.response)


async def main():
    workflow = RAGWorkflow(timeout=120, verbose=True)
    result = await workflow.run(
        resume_file="data/resume.pdf",
        query="What is the name of the candidate and what are the primary skills of the candidate?",
    )
    pprint(result)


if __name__ == "__main__":
    asyncio.run(main())
