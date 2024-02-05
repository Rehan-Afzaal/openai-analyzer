# Python Flask Application for Document Analysis 

## Overview
This Flask application is designed for analyzing documents and extracting specific information based on various prompt types. It supports multiple functionalities including analyzing contracts, private placements, and money market documents.
![WhatsApp Image 2024-02-06 at 03 16 57](https://github.com/Rehan-Afzaal/openai-analyzer/assets/42688016/8f71488d-e37c-4efa-a06e-a07b18028148)

## Main Features
![WhatsApp Image 2024-02-06 at 03 16 57](https://github.com/Rehan-Afzaal/openai-analyzer/assets/42688016/b63f718d-b701-4df6-999f-6280c735fa82)



## API Endpoints

### 1. Document Analyzer
- **Endpoint**: `/analyzer`
- **Method**: POST
- **Description**: Analyzes uploaded documents and processes them based on the provided prompt type.
- **Payload**:
  ```json
  {
    "userId": "string", 
    "propertyId": "string", (required)
    "docType": "string",
    "promptType": "string",
    "file": "file" (required)
  }
  ```
- **Response**:
  - Success: `{"message": "Files are being processed"}`
  - Error: `{"error": "Error message"}`

### 2. Money Market Analysis
- **Endpoint**: `/moneymarket`
- **Method**: POST
- **Description**: Processes money market-related documents.
- **Payload**:
  ```json
  {
    "userId": "string",
    "propertyId": "string",
    "promptType": "string",
    "file": "file" (required)
  }
  ```
- **Response**:
  - Success: `{"message": "Files have been processed"}`
  - Error: `{"error": "Error message"}`

### 3. General Prompt Analysis
- **Endpoint**: `/promptsanalysis`
- **Method**: POST
- **Description**: Processes documents based on a general prompt for analysis.
- **Payload**:
  ```json
  {
    "userId": "string", (required)
    "documentId": "string",
    "promptType": "string",
    "file": "file" (required)
  }
  ```
- **Response**:
  - Success: `{"message": "Files are being processed"}`
  - Error: `{"error": "Error message"}`

### 4. File Upload
- **Endpoint**: `/upload`
- **Method**: POST
- **Description**: Uploads a file for processing.
- **Payload**:
  ```json
  {
    "user_id": "string", (required)
    "file": "file" (required)
  }
  ```
- **Response**:
  - Success: `{"message": "File upload received, processing started", "document_id": "string"}`
  - Error: `{"error": "Error message"}`

### 5. Chat Interaction
- **Endpoint**: `/chat`
- **Method**: POST
- **Description**: Handles chat-based interactions with the document content.
- **Payload**:
  ```json
  {
    "question": "string", (required)
    "document_id": "string", (required)
    "user_id": "string" (required)
  }
  ```
- **Response**:
  - Success: `{"response": "Chatbot response"}`
  - Error: `{"error": "Error message"}`

## Setup and Installation
- Ensure Python 3.x is installed.
- On macOS: `brew install poppler`
- Install dependencies: `pip install -r requirements.txt`
- Set environment variables in `.env`.
- Run the application: `python app.py`


## Overview of DocumentGPT (Chat with documents)

This application is a Flask-based web server designed to manage and process PDF documents. It provides functionalities for uploading PDF files, extracting text from them, and then performing natural language processing tasks using OpenAI's gpt-4-1106-preview API. The core features of the application include:

1. **PDF Upload and Processing**: Users can upload PDF documents through the `/upload` endpoint. The server saves these documents temporarily, extracts text from them using `pytesseract` and `pdf2image`, and processes the text in a background thread.

2. **Document Management with MongoDB**: Each uploaded document is assigned a unique ID and its details (including extracted text chunks) are stored in a MongoDB database. This allows for efficient management and retrieval of document data for further processing.

3. **Integration with OpenAI's gpt-4-1106-preview**: The extracted text is used as input for OpenAI's gpt-4-1106-preview, enabling advanced language processing capabilities. This integration allows the application to generate sophisticated text-based responses, summaries, or analyses based on the document's content.

4. **Interactive Chat Feature**: The `/chat` endpoint leverages the gpt-4-1106-preview model to create an interactive chat experience. Users can ask questions or make requests related to the document's content, and the system, utilizing both the stored text and the gpt-4-1106-preview model, generates relevant responses.

5. **Asynchronous Processing**: To ensure responsiveness and efficiency, the application handles the PDF processing and OpenAI API communication asynchronously. This design choice allows for a non-blocking operation of the web server, providing a smooth user experience.

6. **Scalability and Security**: The application is designed with scalability and security in mind. MongoDB provides a robust platform for handling large amounts of data, and the use of environment variables for sensitive information like database URIs and API keys ensures security best practices.

This software serves as a versatile tool for document management and processing, making it suitable for scenarios where quick analysis, information extraction, and interactive querying of PDF documents are required. It can be particularly useful in corporate, legal, or academic environments where handling and extracting insights from large volumes of documents is a common requirement.#
