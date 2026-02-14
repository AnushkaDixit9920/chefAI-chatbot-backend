from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import os
from datetime import datetime, timedelta
from typing import Dict

# ---------------------------
# Load Environment Variables
# ---------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI(
    title="FlavorAI Backend (Groq)",
    version="3.1",
    description="AI-powered Food & Health Assistant using Groq"
)

# ---------------------------
# CORS
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# In-Memory Storage
# ---------------------------
chat_sessions: Dict[str, Dict] = {}

MAX_HISTORY_LENGTH = 8
SESSION_TIMEOUT_MINUTES = 30
MAX_REQUESTS_PER_MINUTE = 15

# ---------------------------
# Request Schema
# ---------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: str

# ---------------------------
# Cleanup Old Sessions
# ---------------------------
def cleanup_sessions():
    now = datetime.now()
    expired = []

    for sid, data in chat_sessions.items():
        last_active = datetime.fromisoformat(data["last_activity"])
        if now - last_active > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
            expired.append(sid)

    for sid in expired:
        del chat_sessions[sid]

# ---------------------------
# Root
# ---------------------------
@app.get("/")
def home():
    return {
        "status": "running",
        "version": "3.1",
        "active_sessions": len(chat_sessions)
    }

# ---------------------------
# Chat Endpoint
# ---------------------------
@app.post("/chat")
def chat(request: ChatRequest):

    cleanup_sessions()

    user_message = request.message.strip()
    session_id = request.session_id.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if session_id not in chat_sessions:
        chat_sessions[session_id] = {
            "messages": [],
            "last_activity": datetime.now().isoformat(),
            "request_count": 0,
            "rate_limit_reset": datetime.now()
        }

    session = chat_sessions[session_id]
    now = datetime.now()

    # ---------------------------
    # Rate limiting
    # ---------------------------
    if now - session["rate_limit_reset"] > timedelta(minutes=1):
        session["request_count"] = 0
        session["rate_limit_reset"] = now

    if session["request_count"] >= MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Too many requests")

    session["request_count"] += 1
    session["last_activity"] = now.isoformat()

    # Add user message
    session["messages"].append({
        "role": "user",
        "content": user_message
    })

    session["messages"] = session["messages"][-MAX_HISTORY_LENGTH:]

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # ✅ updated model
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are ChefAI, a food and nutrition assistant.\n"
                        "Give SHORT, clear, structured answers.\n"
                        "Maximum 5 bullet points.\n"
                        "No long paragraphs.\n"
                        "Be concise and practical."
                    )
                }
            ] + session["messages"],
            temperature=0.6,
            max_tokens=250
        )

        ai_response = completion.choices[0].message.content

        # Save assistant response
        session["messages"].append({
            "role": "assistant",
            "content": ai_response
        })

        session["messages"] = session["messages"][-MAX_HISTORY_LENGTH:]

        return {
            "response": ai_response,
            "session_id": session_id,
            "timestamp": now.isoformat()
        }

    except Exception as e:
        print("Groq Error:", str(e))
        raise HTTPException(status_code=500, detail="AI service error")
