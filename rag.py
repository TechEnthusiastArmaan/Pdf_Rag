import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Third-party imports
import streamlit as st
from dotenv import load_dotenv
import PyPDF2
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF text extraction and processing"""
    
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
                
                logger.info(f"Successfully extracted text from {pdf_path}")
                return text
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def create_chunks(self, text: str) -> List[str]:
        """Split text into chunks for vector storage"""
        try:
            # Simple text chunking by sentences/paragraphs
            chunks = []
            words = text.split()
            
            for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                chunk = " ".join(words[i:i + self.chunk_size])
                if chunk.strip():
                    chunks.append(chunk.strip())
            
            logger.info(f"Created {len(chunks)} text chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating text chunks: {str(e)}")
            raise

class SimpleVectorStore:
    """Simple vector store using TF-IDF and cosine similarity"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.vectors = None
        self.documents = []
        self.is_fitted = False
    
    def create_vectorstore(self, documents: List[str]) -> None:
        """Create vector store from documents"""
        try:
            self.documents = documents
            self.vectors = self.vectorizer.fit_transform(documents)
            self.is_fitted = True
            logger.info("Successfully created vector store")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Perform similarity search"""
        try:
            if not self.is_fitted:
                raise ValueError("Vector store not initialized")
            
            # Transform query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.vectors).flatten()
            
            # Get top k results
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only include non-zero similarities
                    results.append({
                        'content': self.documents[idx],
                        'score': similarities[idx]
                    })
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise
    
    def save_vectorstore(self, path: str) -> None:
        """Save vector store to disk"""
        try:
            data = {
                'vectorizer': self.vectorizer,
                'vectors': self.vectors,
                'documents': self.documents,
                'is_fitted': self.is_fitted
            }
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Vector store saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def load_vectorstore(self, path: str) -> None:
        """Load vector store from disk"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            self.vectorizer = data['vectorizer']
            self.vectors = data['vectors']
            self.documents = data['documents']
            self.is_fitted = data['is_fitted']
            
            logger.info(f"Vector store loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise

class GroqLLM:
    """Interface for Groq API calls"""
    
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        logger.info(f"Initialized Groq LLM with model: {model}")
    
    def generate_response(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Generate response using Groq API"""
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 1024,
                "top_p": 1,
                "stream": False
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

class SimpleRAGSystem:
    """Main RAG system using simple TF-IDF embeddings"""
    
    def __init__(self, groq_api_key: str, model_name: str = "llama3-8b-8192"):
        self.pdf_processor = PDFProcessor()
        self.vector_store = SimpleVectorStore()
        self.llm = GroqLLM(groq_api_key, model_name)
        self.is_initialized = False
        logger.info("Simple RAG system initialized")
    
    def process_pdf(self, pdf_path: str) -> None:
        """Process PDF and create vector store"""
        try:
            # Extract text from PDF
            text = self.pdf_processor.extract_text_from_pdf(pdf_path)
            
            # Create chunks
            documents = self.pdf_processor.create_chunks(text)
            
            # Create vector store
            self.vector_store.create_vectorstore(documents)
            
            self.is_initialized = True
            logger.info("PDF processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    def query(self, question: str, context_length: int = 4) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            if not self.is_initialized:
                raise ValueError("RAG system not initialized. Please process a PDF first.")
            
            # Retrieve relevant context
            relevant_docs = self.vector_store.similarity_search(question, k=context_length)
            
            # Prepare context for the LLM
            context = "\n\n".join([doc['content'] for doc in relevant_docs])
            
            # Create prompt
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context. Use only the information from the context to answer questions. If the answer is not in the context, say so clearly."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                }
            ]
            
            # Generate response
            answer = self.llm.generate_response(messages)
            
            return {
                "answer": answer,
                "context": context,
                "source_documents": relevant_docs,
                "scores": [doc['score'] for doc in relevant_docs]
            }
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {str(e)}")
            raise

def load_environment():
    """Load environment variables"""
    load_dotenv()
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    return groq_api_key

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Simple RAG PDF Q&A",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Simple RAG PDF Question Answering")
    st.markdown("Upload a PDF document and ask questions about its content! (Using TF-IDF embeddings)")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'processed_file' not in st.session_state:
        st.session_state.processed_file = None
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        model_options = [
            "llama3-8b-8192",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
            "gemma-7b-it"
        ]
        selected_model = st.selectbox("Select Model", model_options)
        
        # Context length
        context_length = st.slider("Context Length", 1, 10, 4)
        
        st.info("✅ No complex embeddings required!")
        st.info("📝 Make sure to set your GROQ_API_KEY in the .env file")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📄 PDF Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to query"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Process PDF if it's a new file
            if st.session_state.processed_file != uploaded_file.name:
                try:
                    with st.spinner("Processing PDF..."):
                        # Load environment and initialize RAG system
                        groq_api_key = load_environment()
                        st.session_state.rag_system = SimpleRAGSystem(groq_api_key, selected_model)
                        
                        # Process the PDF
                        st.session_state.rag_system.process_pdf(temp_path)
                        st.session_state.processed_file = uploaded_file.name
                    
                    st.success("✅ PDF processed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    
    with col2:
        st.header("❓ Ask Questions")
        
        if st.session_state.rag_system and st.session_state.processed_file:
            # Question input
            question = st.text_input(
                "Enter your question:",
                placeholder="What is the main topic of this document?"
            )
            
            if st.button("Get Answer", type="primary"):
                if question:
                    try:
                        with st.spinner("Generating answer..."):
                            result = st.session_state.rag_system.query(question, context_length)
                        
                        # Display answer
                        st.subheader("Answer:")
                        st.write(result["answer"])
                        
                        # Display relevance scores
                        if result["scores"]:
                            avg_score = sum(result["scores"]) / len(result["scores"])
                            st.info(f"📊 Average relevance score: {avg_score:.3f}")
                        
                        # Display context (expandable)
                        with st.expander("View Retrieved Context"):
                            st.text_area("Context:", result["context"], height=200)
                        
                    except Exception as e:
                        st.error(f"❌ Error generating answer: {str(e)}")
                else:
                    st.warning("⚠️ Please enter a question")
        else:
            st.info("📋 Please upload and process a PDF file first")
    
    # Footer
    st.markdown("---")
    st.markdown("🚀 Built with Streamlit, TF-IDF, and Groq API | No complex embeddings needed!")

if __name__ == "__main__":
    main()