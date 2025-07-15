# RAG PDF Question Answering System

A professional-grade Retrieval-Augmented Generation (RAG) application that extracts text from PDFs, creates vector embeddings, and queries Groq-hosted LLMs for intelligent question answering.

## Features

- 📄 **PDF Text Extraction**: Extract and process text from PDF documents
- 🔍 **Vector Search**: Create embeddings and perform similarity search
- 🤖 **Groq LLM Integration**: Query powerful models like Llama 3, Mixtral, and Gemma
- 🖥️ **Multiple Interfaces**: Streamlit web app and CLI interface
- 🔧 **Modular Architecture**: Clean, reusable components
- 🔐 **Secure Configuration**: Environment-based API key management

## Architecture

```
RAG System Components:
├── PDFProcessor: Extract and chunk text from PDFs
├── VectorStore: Create embeddings and similarity search
├── GroqLLM: Interface with Groq API
└── RAGSystem: Orchestrate all components
```

## Installation

### 1. Clone and Setup

```bash
# Create project directory
mkdir rag_pdf_app
cd rag_pdf_app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```bash
# Copy template
cp .env.template .env

# Edit with your API key
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 3. Get Groq API Key

1. Visit [Groq Console](https://console.groq.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Generate a new API key
5. Copy the key to your `.env` file

## Usage

### Web Interface (Streamlit)

```bash
# Start the web application
streamlit run rag_app.py
```

Features:
- Upload PDF files through web interface
- Select different Groq models
- Adjust context length
- Interactive question answering
- View retrieved context

### Command Line Interface

```bash
# Single question
python cli_rag.py --pdf document.pdf --query "What is the main topic?"

# Interactive mode
python cli_rag.py --pdf document.pdf --interactive

# With specific model
python cli_rag.py --pdf document.pdf --query "Summarize key points" --model llama3-70b-8192

# Verbose output
python cli_rag.py --pdf document.pdf --query "Your question" --verbose
```

## Available Models

- `llama3-8b-8192`: Fast, efficient for general use
- `llama3-70b-8192`: More capable, better reasoning
- `mixtral-8x7b-32768`: Large context window
- `gemma-7b-it`: Google's instruction-tuned model

## Configuration Options

### Environment Variables

```bash
# Required
GROQ_API_KEY=your_api_key

# Optional
DEFAULT_MODEL=llama3-8b-8192
VECTOR_STORE_PATH=./vectorstore
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Text Chunking Parameters

```python
# In PDFProcessor class
chunk_size=1000        # Size of text chunks
chunk_overlap=200      # Overlap between chunks
```

### Vector Search Parameters

```python
# In similarity_search method
k=4                    # Number of similar documents to retrieve
```

## Advanced Usage

### Custom Embedding Model

```python
# Initialize with custom model
vector_store = VectorStore(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
```

### Save/Load Vector Store

```python
# Save processed embeddings
rag_system.vector_store.save_vectorstore("./my_vectorstore")

# Load existing embeddings
rag_system.vector_store.load_vectorstore("./my_vectorstore")
```

### Batch Processing

```python
# Process multiple PDFs
pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
for pdf in pdf_files:
    rag_system.process_pdf(pdf)
```

## Troubleshooting

### Common Issues

1. **API Key Error**
   ```
   Error: GROQ_API_KEY not found in environment variables
   ```
   Solution: Ensure `.env` file exists with correct API key

2. **PDF Reading Error**
   ```
   Error extracting text from PDF
   ```
   Solution: Check PDF file is not corrupted or password-protected

3. **Memory Issues**
   ```
   CUDA out of memory
   ```
   Solution: Use CPU embeddings or reduce chunk size

### Performance Optimization

1. **For Large PDFs**: Increase chunk size, reduce overlap
2. **For Better Accuracy**: Use more context documents (higher k)
3. **For Speed**: Use smaller embedding models
4. **For Memory**: Use CPU instead of GPU embeddings

## Development

### Project Structure

```
rag_pdf_app/
├── rag_app.py           # Main Streamlit application
├── cli_rag.py           # Command-line interface
├── requirements.txt     # Dependencies
├── .env                 # Environment variables
├── .env.template        # Environment template
└── README.md           # This file
```

### Adding New Features

1. **New Document Types**: Extend `PDFProcessor` class
2. **Different LLM Providers**: Create new LLM interface classes
3. **Custom Retrievers**: Extend `VectorStore` class
4. **New Embeddings**: Modify embedding model initialization

### Testing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# With coverage
pytest --cov=rag_app tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Security Notes

- Never commit API keys to version control
- Use environment variables for sensitive data
- Regularly rotate API keys
- Monitor API usage and costs

## Support

For issues and questions:
- Check the troubleshooting section
- Review Groq API documentation
- Open an issue on GitHub