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
        print("STEP 1: Downloading audio...")
        subprocess.run([
            "yt-dlp", "--extract-audio", "--audio-format", "mp3",
            "--no-check-certificate",
            "--cookies", "cookies.txt",
            "-o", audio_path, req.url
        ], check=True)
        print("✅ Audio downloaded at", audio_path)

        # --- Transcription ---
        print("STEP 2: Transcribing via Groq Whisper...")
        with open(audio_path, "rb") as f:
            whisper_resp = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": f},
                data={"model": "whisper-large-v3"}
            )

        print("Transcription response code:", whisper_resp.status_code)
        print("Transcription response text:", whisper_resp.text[:200])

        whisper_resp.raise_for_status()
        transcript = whisper_resp.json().get("text", "")
        if not transcript:
            raise Exception("Empty transcript returned.")

        # --- Summarization ---
        print("STEP 3: Summarizing via Groq LLaMA...")
        llm_resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.1-70b-versatile",
                "messages": [{"role": "user", "content": f"Summarize:\n{transcript}"}],
                "temperature": 0.5
            }
        )

        print("LLaMA response code:", llm_resp.status_code)
        print("LLaMA response text:", llm_resp.text[:200])

        llm_resp.raise_for_status()
        data = llm_resp.json()
        if "choices" not in data:
            raise Exception(f"Unexpected response: {data}")

        summary = data["choices"][0]["message"]["content"]
        print("✅ Summary generated successfully")
        return {"summary": summary}

    except Exception as e:
        print("❌ ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

