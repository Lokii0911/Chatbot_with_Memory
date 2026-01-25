# Nova AI- Streaming Chat Assistant with Memory & Auth
Nova AI is a full-stack AI chat application built with FastAPI, LangChain, SQL Server, JWT authentication, and Streamlit.
It supports multi-user chat sessions, token-streaming responses (SSE), conversation memory with summarization, and a clean chat UI.
This project is designed to go beyond tutorials and reflect real-world system design choices.
Authentication & Security


# JWT-based authentication (Login / Register)
1.Secure password hashing

2.Token expiry handling

3.Request size limits

4.Rate limiting per user


# Features
## Authentication & Security
1.JWT-based authentication (Login / Register)

2.Secure password hashing

3.Token expiry handling

4.Request size limits

5.Rate limiting per user


## Chat System
1.Multiple chat sessions per user

2.Session-scoped conversations

3.Persistent chat history stored in SQL Server


## Streaming AI Responses
1.Server-Sent Events (SSE) streaming from backend

2.Token-by-token rendering in Streamlit


## Memory Architecture
1.Database-backed conversation history

2.ConversationSummaryBufferMemory

     1.Keeps recent messages 
     
     2.Summarizes older context automatically

## Frontend (Streamlit)

1.Login / Register UI

2.Sidebar chat history

3.Create new chats with titles

4.Live streaming chat interface

## Architecture Overview
<img width="336" height="309" alt="image" src="https://github.com/user-attachments/assets/3ec96d97-dcaf-4ffd-bc12-910b6c040197" />

# Tech Stack
## Backend

1.FastAPI

2.LangChain

3.OpenRouter / OpenAI-compatible LLMs

4.SQL Server (via pyodbc)

5.python-jose (JWT)

6.passlib + bcrypt

7.SlowAPI (rate limiting)

## Frontend

1.Streamlit

2.Requests (SSE streaming)

# UI Screenshots

<img width="725" height="407" alt="image" src="https://github.com/user-attachments/assets/e9fc65ed-59bd-4153-8e3c-7c63c882828c" />

<img width="959" height="545" alt="image" src="https://github.com/user-attachments/assets/7094763f-0917-44a1-882e-22670faf870d" />

