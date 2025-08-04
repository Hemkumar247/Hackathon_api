# High-Accuracy HackRx RAG API

## Description

This project is an advanced FastAPI-based API developed for the HackRx 6.0 hackathon. It implements a dynamic Retrieval-Augmented Generation (RAG) system that can analyze any PDF document from a URL and answer questions based *only* on the content within that document.

The API is designed for high accuracy, ensuring that answers are strictly derived from the provided document and preventing the model from using external knowledge.

## Features

-   **Dynamic Document Processing**: Ingests PDFs directly from a URL for on-the-fly analysis.
-   **In-Memory Vector Store**: Creates a temporary, request-specific `FAISS` vector store for fast and efficient semantic search.
-   **High-Accuracy RAG Pipeline**: Utilizes a precisely engineered prompt to force the LLM to answer questions based only on the retrieved context.
-   **FastAPI Backend**: A robust and fast API server built with FastAPI.
-   **LLM Integration**: Powered by Google's `gemini-1.5-flash-latest` model for generation.
-   **State-of-the-Art Embeddings**: Employs Hugging Face's `sentence-transformers/all-MiniLM-L6-v2` for creating high-quality text embeddings.

## Technologies Used

-   **Python 3.x**
-   **FastAPI**: For building the API.
-   **LangChain**: As the framework for building the RAG application.
-   **Google Generative AI**: For the core language model.
-   **Hugging Face Transformers**: For sentence embeddings.
-   **FAISS**: For the in-memory vector store.
-   **Pydantic**: For data validation.
-   **Uvicorn**: As the ASGI server.
-   **Render**: For deployment.

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

4.  **Create a `.env` file** in the root directory if you need to manage any API keys (e.g., for Google AI).

## API Endpoints

### `POST /hackrx/run`

This is the main endpoint for processing a document and asking questions.

-   **Request Body**:
    ```json
    {
      "documents": "URL_TO_YOUR_PDF_DOCUMENT",
      "questions": [
        "Your first question here",
        "Your second question here"
      ]
    }
    ```

-   **Successful Response (200 OK)**:
    ```json
    {
      "answers": [
        "Answer to the first question.",
        "Answer to the second question."
      ]
    }
    ```

-   **Error Response (400 Bad Request)**: If the document URL is invalid or inaccessible.
-   **Error Response (500 Internal Server Error)**: For any other internal processing errors.

### `GET /`

A simple health check endpoint.

-   **Response**:
    ```json
    {
      "status": "High-Accuracy API is online and ready."
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
