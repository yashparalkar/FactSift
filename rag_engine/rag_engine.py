import os
from dotenv import load_dotenv
from typing import List, Tuple

from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub

from rag_engine.google_news_links import simple_google_search
from rag_engine.quality_filtering import credibility_scores
from rag_engine.news_article import load_web_content

load_dotenv()
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH-KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI-KEY")

class RAGPipeline:
    def __init__(self):
        self.llm = init_chat_model("gpt-4.1", model_provider="openai")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.prompt = hub.pull("rlm/rag-prompt")

    def load_documents(self, query: str) -> List[Document]:
        urls = simple_google_search(query)
        docs = []
        for url in urls:
            try:
                docs.append(load_web_content(url))
            except Exception as e:
                print(f"[Error loading {url}]: {e}")
        credibility_scores(docs)
        return docs

    def index_documents(self, docs: List[Document]):
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200, add_start_index=True)
        chunks = splitter.split_documents(docs)
        self.vector_store.add_documents(documents=chunks)

    def retrieve_context(self, question: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        return self.vector_store.similarity_search_with_score(question, k=top_k)

    def score_and_select_context(self, context: List[Tuple[Document, float]], top_n: int = 3) -> List[Tuple[Document, float]]:
        for doc, sim_score in context:
            quality = doc.metadata.get("quality_score", 0.5)
            doc.metadata["final_score"] = 0.8 * sim_score + 0.2 * quality
        return sorted(context, key=lambda x: x[0].metadata["final_score"], reverse=True)[:top_n]

    def generate_answer(self, question: str, context: List[Tuple[Document, float]], chat_history: List[dict]) -> str:
        docs_content = "\n\n".join([doc.page_content for doc, _ in context])

        # Construct messages list
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on news articles."}
        ] + chat_history + [
            {"role": "user", "content": f"Answer the following question using this context:\n\n{docs_content}\n\nQuestion: {question}"}
        ]

        response = self.llm.invoke(messages)
        return response.content


def initialize_rag_pipeline() -> RAGPipeline:
    return RAGPipeline()

def process_query(query: str, pipeline: RAGPipeline, chat_history: List[dict]) -> Tuple[str, List[dict]]:
    docs = pipeline.load_documents(query)
    pipeline.index_documents(docs)
    raw_context = pipeline.retrieve_context(query)
    final_context = pipeline.score_and_select_context(raw_context)
    answer = pipeline.generate_answer(query, final_context, chat_history)

    # Append user and assistant turn
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})

    return {"answer": answer, "chat_history": chat_history}