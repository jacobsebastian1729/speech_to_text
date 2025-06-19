import os
import random
import string
from dotenv import load_dotenv
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableLambda, RunnableMap

import google.generativeai as genai  # Gemini SDK

# Load environment variables
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
load_dotenv(dotenv_path=env_path)

# Gemini API setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Helper: convert docs to a string
def doc2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Random string for filenames (if needed)
def random_string(length=8):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

# Prompt template
template = """
You are Nova, the AI assistant for 4Labs Technologies. Your job is to help visitors by answering questions about the company, its leadership, services, or navigation—just like a knowledgeable and professional team member.

Rules:

- Use only the information provided in the context.
- Do not refer to the "context" or "provided information" in your replies. Just respond naturally and confidently.
- If specific information is missing, say:
  "I appreciate your interest. While I don’t have that specific information at the moment."
- Use a helpful, polished, and human tone. Avoid sounding robotic or overly generic.

Conversational Behavior:

- If the user responds with "thanks", "thank you", or similar, reply with appreciation and avoid continuing the conversation unless prompted.
- If the user says "bye", "goodbye", or similar, respond with a warm farewell and end the conversation.
- If the user says "no", "nope", or "nah", acknowledge politely without repeating questions.
- Avoid asking “Is there anything else I can help you with?” more than once in the conversation.
- If the user gives brief acknowledgments (like "ok", "cool", "yup", "sure"), respond politely and wait for further input instead of prompting again.

Chat History:  
{chat_history}

Context:  
{context}

Question: {question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

# Gemini invocation function
def invoke_gemini(inputs):
    full_prompt = prompt.format(**inputs)
    response = gemini_model.generate_content(full_prompt)
    return response.text.strip()

# RAG pipeline
class RAGPipeline:
    def __init__(self, file_path=Path(__file__).parent / "docs" / "rag_doc.docx",
                 model_name="all-MiniLM-L6-v2", chunk_size=500, chunk_overlap=200, top_k=2):
        self.file_path = file_path
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self.load_document()
        self.split_document()
        self.init_embedding()
        self.create_vector_store()
        self.initialize_retriever()

    def load_document(self):
        loader = Docx2txtLoader(self.file_path)
        self.documents = loader.load()

    def split_document(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.splits = splitter.split_documents(self.documents)

    def init_embedding(self):
        self.embedding_function = SentenceTransformerEmbeddings(model_name=self.model_name)

    def create_vector_store(self):
        self.vector_store = FAISS.from_documents(
            documents=self.splits,
            embedding=self.embedding_function,
        )

    def initialize_retriever(self):
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})

    def run_query(self, query):
        if self.retriever:
            return self.retriever.get_relevant_documents(query)
        return []

# Chain construction using Gemini
def create_rag_chain(pipeline: RAGPipeline):
    retriever = pipeline.vector_store.as_retriever()

    chain = RunnableMap({
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"],
        "chat_history": lambda x: x.get("chat_history", "")
    }) | RunnableLambda(invoke_gemini)

    return chain


# Test runner
def main():
    print("🔧 Initializing Gemini-based RAG...")
    pipeline = RAGPipeline()
    chain = create_rag_chain(pipeline)

    chat_history = []
    print("✅ RAG is ready. Type your questions (type 'exit' to quit):")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit", "bye"]:
            print("Nova: It’s been a pleasure chatting with you. Take care! 👋")
            break

        response = chain.invoke({
            "question": query,
            "chat_history": "\n".join(chat_history)
        })

        print("Nova:", response)
        chat_history.append(f"User: {query}")
        chat_history.append(f"Nova: {response}")


if __name__ == "__main__":
    main()
