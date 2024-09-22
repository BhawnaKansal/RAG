
# Interactive QA Bot

This project implements an interactive Question Answering (QA) bot using Retrieval-Augmented Generation (RAG). Users can upload PDF documents and ask questions about their content.

## Features

- PDF document upload and processing
- Real-time question answering based on document content
- Display of retrieved document segments alongside answers
- Dockerized application for easy deployment

## Prerequisites

- Python 3.9+
- Docker
- Pinecone API key
- Cohere API key

## Setup and Installation

1. Clone this repository:
   ```
   git clone https://github.com/BhawnaKansal/RAG
   cd RAG
   ```

2. Create a `.env` file in the project root and add your API keys:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   COHERE_API_KEY=your_cohere_api_key
   PINECONE_ENV=your_pinecone_environment
   ```

3. Build the Docker image:
   ```
   docker build -t qa-bot .
   ```

4. Run the Docker container:
   ```
   docker run -p 8501:8501 --env-file .env qa-bot
   ```

5. Open your browser and go to `http://localhost:8501` to access the application.

## Usage

1. Upload a PDF document using the file uploader.
2. Wait for the document to be processed.
3. Type your question in the text input field.
4. View the generated answer and retrieved document segments.

## Development

To run the application locally for development:

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   python -m streamlit run  rag.py 
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
