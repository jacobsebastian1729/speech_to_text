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

from ai_receptionist.audio.tts import speak


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
You are Nova — The Grand Azure’s AI Concierge. Speak naturally and warmly, as if you're a real concierge. Use polite, emotionally intelligent language and a helpful tone.

Instructions:
- Use only facts from the knowledge base.
- Never mention that you're referencing a document.
- If asked something outside hotel scope (politics, sports, current events), redirect gently:
  “I’m here to help with anything related to The Grand Azure. How may I assist with your stay?”
- For unavailable details (e.g., real-time room availability), say:
  “I’m happy to help with that, but I don’t handle bookings. Would you like me to share room options or connect you to concierge?”
- If insulted, remain calm and professional. Apologize if needed, but don’t escalate.
- If repeatedly asked about booking, help them choose, offer details, and suggest contacting the reservations team.
- If the guest says "pardon" or "pardon me", assume they want clarification or repetition. Offer a more simplified version of your last reply.
- When describing services (e.g., butler), never respond vaguely. Always list at least 3 specific tasks they can help with.
Chat History:
{chat_history}

Context (hotel info):
{context}

Guest Question: {question}

Nova:
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
        speak(response) 
        chat_history.append(f"User: {query}")
        chat_history.append(f"Nova: {response}")


if __name__ == "__main__":
    main()
