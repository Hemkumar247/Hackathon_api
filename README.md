# HackRx RAG API

## Description

This project is a FastAPI-based API developed for the HackRx 6.0 hackathon. It implements a Retrieval-Augmented Generation (RAG) system to answer questions based on a provided set of documents. The API leverages large language models (LLMs) and a vector database to provide accurate, context-aware answers.

## Features

- **RAG Implementation**: Utilizes a RAG pipeline to retrieve relevant document snippets and generate answers.
- **FastAPI Backend**: A robust and fast API server built with FastAPI.
- **Vector Database**: Uses AstraDB as a vector store for efficient document retrieval.
- **LLM Integration**: Powered by Google's Gemini models for generative capabilities.
- **Sentence Transformers**: Employs Hugging Face's sentence transformers for creating document embeddings.
- **Environment-based Configuration**: Securely manages credentials and settings using a `.env` file.

## Technologies Used

- **Python 3.x**
- **FastAPI**: For building the API.
- **LangChain**: As the framework for building the RAG application.
- **Google Generative AI**: For the core language model (`gemini-1.5-flash-latest`).
- **Hugging Face Transformers**: For sentence embeddings (`all-MiniLM-L6-v2`).
- **AstraDB**: As the vector store for document embeddings.
- **Pydantic**: For data validation.
- **Uvicorn**: As the ASGI server.
- **Render**: For deployment.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Hemkumar247/Hackathon_api.git
    cd Hackathon_api
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file** in the root directory and add the necessary environment variables (see below).

## Environment Variables

You need to create a `.env` file with the following variables:

```
ASTRA_DB_API_ENDPOINT="your_astra_db_api_endpoint"
ASTRA_DB_APPLICATION_TOKEN="your_astra_db_application_token"
```

## API Endpoints

### `POST /ask`

This is the main endpoint for asking questions.

-   **Request Body**:
    ```json
    {
      "question": "Your question here"
    }
    ```

-   **Successful Response (200 OK)**:
    ```json
    {
      "answer": "The generated answer.",
      "source_documents": [
        {
          "page_content": "The content of the source document.",
          "metadata": {}
        }
      ]
    }
    ```

-   **Error Response (500 Internal Server Error)**:
    ```json
    {
      "detail": "RAG chain is not available due to a startup error."
    }
    ```

### `GET /`

A simple health check endpoint.

-   **Response**:
    ```json
    {
      "status": "API is online and ready."
    }
    ```

## How to Run Locally

To run the application locally, use the following command:

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## Deployment

This project is configured for deployment on Render. The `render.yaml` file defines the services and build commands. Pushing to the connected GitHub repository will trigger a new deployment on Render.
