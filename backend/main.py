from fastapi import FastAPI
from pydantic import BaseModel
import subprocess, os, uuid, requests

app = FastAPI()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class VideoRequest(BaseModel):
    url: str

@app.post("/summarize")
def summarize(req: VideoRequest):
    vid = str(uuid.uuid4())
    audio_path = f"/tmp/{vid}.mp3"

    # Download audio
    subprocess.run(["yt-dlp", "-x", "--audio-format", "mp3", "-o", audio_path, req.url], check=True)

    # Transcribe via Groq Whisper
    whisper_resp = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": open(audio_path, "rb")},
        data={"model": "whisper-large-v3"},
    )
    text = whisper_resp.json().get("text", "")

    # Summarize via Groq LLaMA
    prompt = f"Summarize the following YouTube transcript:\n\n{text}"
    llm_resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": "llama-3.1-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
        },
    )
    summary = llm_resp.json()["choices"][0]["message"]["content"]

    os.remove(audio_path)
    return {"summary": summary}
