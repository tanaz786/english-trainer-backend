from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import httpx
import os
import io
import tempfile
import base64

load_dotenv("../.env", override=False)

app = FastAPI()
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
DID_API_KEY = os.getenv("DID_API_KEY")
LUNA_IMAGE_URL = "https://i.imgur.com/med0l1n.jpeg"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are "Luna", a warm, friendly and encouraging English communication coach. You speak like a real human friend, not a robot.

Your personality:
- Warm, fun, supportive and patient
- You celebrate small wins and keep the user motivated
- You speak naturally, use contractions (I'm, you're, let's, etc.)
- You keep responses SHORT and conversational (2-4 sentences max)

Your job:
- Have natural English conversations with the user
- Gently correct grammar mistakes by naturally using the correct form in your reply
- Teach vocabulary and phrases in context
- Help the user become a confident English communicator

When user makes a mistake, DON'T say "you made a mistake". Instead naturally use the correct form and move the conversation forward. Occasionally you can say things like "By the way, we usually say it like this: ..."

Always end with either a follow-up question or encouragement to keep talking."""

class ChatRequest(BaseModel):
    message: str
    history: list = []
    scenario: str = "Have a friendly English conversation."

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        result = groq.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=("audio.m4a", f, "audio/m4a")
        )
    os.unlink(tmp_path)
    return {"text": result.text}

@app.post("/chat-and-speak")
async def chat_and_speak(data: ChatRequest):
    messages = [{"role": "system", "content": SYSTEM_PROMPT + f"\n\nCurrent scenario: {data.scenario}"}]
    for h in data.history:
        messages.append(h)
    messages.append({"role": "user", "content": data.message})

    response = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=150
    )
    reply = response.choices[0].message.content

    # Generate talking video with D-ID
    async with httpx.AsyncClient() as http:
        did_response = await http.post(
            "https://api.d-id.com/talks",
            headers={
                "Authorization": f"Basic {DID_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "source_url": LUNA_IMAGE_URL,
                "script": {
                    "type": "text",
                    "input": reply,
                    "provider": {
                        "type": "microsoft",
                        "voice_id": "en-US-JennyNeural"
                    }
                },
                "config": {"fluent": True, "pad_audio": 0}
            },
            timeout=30
        )
        did_data = did_response.json()
        talk_id = did_data.get("id")

    # Poll for video result
    video_url = None
    if talk_id:
        async with httpx.AsyncClient() as http:
            for _ in range(20):
                import asyncio
                await asyncio.sleep(2)
                status_res = await http.get(
                    f"https://api.d-id.com/talks/{talk_id}",
                    headers={"Authorization": f"Basic {DID_API_KEY}"}
                )
                status_data = status_res.json()
                if status_data.get("status") == "done":
                    video_url = status_data.get("result_url")
                    break

    return {"reply": reply, "video_url": video_url}
