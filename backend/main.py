from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess, os, uuid, requests

# ----------------- Configuration -----------------
app = FastAPI()

# Add your frontend URLs here, or use ["*"] for testing
origins = [
    "https://cap-it.vercel.app",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # allow your frontend
    allow_credentials=True,
    allow_methods=["*"],        # allows POST, OPTIONS, GET etc.
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable not set!")

# ----------------- Models -----------------
class VideoRequest(BaseModel):
    url: str

# ----------------- Root endpoint -----------------
@app.get("/")
def root():
    return {"message": "YouTube Summarizer API running. POST to /summarize with {url}."}

# ----------------- Summarize endpoint -----------------
@app.post("/summarize")
def summarize(req: VideoRequest):
    vid_id = str(uuid.uuid4())
    audio_path = f"/tmp/{vid_id}.mp3"

    try:
        # 1️⃣ Download audio
        subprocess.run([
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "mp3",
            "--no-check-certificate",
            "-o", audio_path,
            req.url
        ], check=True)

        # 2️⃣ Transcribe via Groq Whisper
        try:
            with open(audio_path, "rb") as f:
                whisper_resp = requests.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    files={"file": f},
                    data={"model": "whisper-large-v3"}
                )
            whisper_resp.raise_for_status()
            transcript = whisper_resp.json().get("text", "")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

        # 3️⃣ Summarize via Groq LLaMA
        prompt = f"Summarize the following YouTube transcript:\n\n{transcript}"
        try:
            llm_resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": "llama-3.1-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.5
                }
            )
            llm_resp.raise_for_status()
            summary = llm_resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

        return {"summary": summary}

    finally:
        # 4️⃣ Cleanup audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
