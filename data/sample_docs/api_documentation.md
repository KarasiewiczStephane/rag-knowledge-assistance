# API Documentation

## Overview

This document describes the REST API endpoints for the Knowledge Assistant application. The API provides programmatic access to document management, querying, and evaluation features.

## Authentication

All API endpoints require an API key passed in the Authorization header:

```
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

### Health Check

**GET /health**

Returns the health status of the application.

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "documents_indexed": 42
}
```

### Document Management

#### Upload Document

**POST /documents/upload**

Upload a document for ingestion into the knowledge base.

Parameters:
- file (multipart/form-data): The document file (PDF, DOCX, MD, TXT)

Response:
```json
{
  "status": "success",
  "document_id": "abc123",
  "chunks_created": 15,
  "file_name": "report.pdf"
}
```

#### List Documents

**GET /documents**

List all indexed documents.

Response:
```json
{
  "documents": [
    {
      "source_file": "report.pdf",
      "chunk_count": 15,
      "file_hash": "abc123def456",
      "title": "report"
    }
  ]
}
```

#### Delete Document

**DELETE /documents/{document_id}**

Remove a document and its chunks from the knowledge base.

Response:
```json
{
  "status": "deleted",
  "chunks_removed": 15
}
```

### Query

#### Ask Question

**POST /query**

Submit a question to the RAG pipeline.

Request Body:
```json
{
  "question": "What is machine learning?",
  "session_id": "optional-session-id",
  "top_k": 5
}
```

Response:
```json
{
  "answer": "Machine learning is a subset of AI...",
  "confidence": 0.85,
  "citations": [
    {
      "source_file": "ml_intro.md",
      "page_number": 1,
      "relevance_score": 0.92,
      "excerpt": "Machine learning (ML) is a subset..."
    }
  ]
}
```

### Sessions

#### Create Session

**POST /sessions**

Create a new conversation session.

Response:
```json
{
  "session_id": "uuid-string"
}
```

#### Get Session History

**GET /sessions/{session_id}**

Retrieve conversation history for a session.

Response:
```json
{
  "session_id": "uuid-string",
  "messages": [
    {
      "role": "user",
      "content": "What is ML?",
      "timestamp": "2024-01-15T10:30:00"
    },
    {
      "role": "assistant",
      "content": "Machine learning is...",
      "timestamp": "2024-01-15T10:30:05"
    }
  ]
}
```

## Error Handling

All errors follow a consistent format:

```json
{
  "error": "Error description",
  "code": 400
}
```

Common error codes:
- 400: Bad request (invalid parameters)
- 401: Unauthorized (missing or invalid API key)
- 404: Not found (document or session not found)
- 500: Internal server error
