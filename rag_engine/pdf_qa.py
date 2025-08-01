import os
from dotenv import load_dotenv
from typing_extensions import List, TypedDict

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain import hub

import fitz  # PyMuPDF
from PIL import Image
import pytesseract


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class PDFContextRetriever:
    def __init__(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.prompt = hub.pull('rlm/rag-prompt')
        self.llm = init_chat_model(model="gpt-4.1-nano", model_provider='openai')
        self._prepare_documents()

    def has_extractable_text(self):
        doc = fitz.open(self.file_path)
        for page in doc:
            if page.get_text():
                return True
        return False
    
    def perform_ocr(self):
        doc = fitz.open(self.file_path)
        t = ''
        # Iterate over PDF pages and convert to images
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img)
            # t += f"Page {i + 1} text:\n{text}\n"
            t += text
        return t

    def _prepare_documents(self):
        if self.has_extractable_text():
            print('PDF Extractable')
            loader = PyPDFLoader(self.file_path)
            docs = loader.load()
        else:
            print('PDF needs OCR')
            raw_text = self.perform_ocr()
            docs = [Document(page_content=raw_text)]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True
        )
        all_splits = splitter.split_documents(docs)  
        self.vector_store.add_documents(documents=all_splits)

    def retrieve_context(self, question: str, top_k: int = 2):
        results = self.vector_store.similarity_search(question, k=top_k)
        return results
    
    def generate(self, question: str, context: List[Document]):
        docs_content = "\n\n".join([doc.page_content for doc in context])
        messages = self.prompt.invoke({"question": question, "context": docs_content})
        response = self.llm.invoke(messages)
        return response.content

# load_dotenv()
# question = "Who Daenerys Targaryen wed?"
# question = 'What is the total price?'
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI-KEY")
# retriever = PDFContextRetriever(file_path='Purchase Order - JSW Mineral - BOBSNS - TWL.pdf')
# context = retriever.retrieve_context(question)
# answer = retriever.generate(question, context)
# print(answer)