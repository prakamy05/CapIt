from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess, os, uuid, requests

# ----------------- Configuration -----------------
app = FastAPI()

# Add your frontend URLs here
origins = [
    "https://cap-it.vercel.app",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
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
        # --------- STEP 1: Download audio ---------
        print("STEP 1: Downloading audio...")
        result = subprocess.run(
            [
                "yt-dlp", "--extract-audio", "--audio-format", "mp3",
                "-o", f"/tmp/{vid_id}.%(ext)s", req.url
            ],
            check=True,
            capture_output=True,
            text=True
        )
        print("yt-dlp output:", result.stdout)
        if not os.path.exists(audio_path):
            raise Exception(f"Audio file not found at {audio_path}")

        print("✅ Audio downloaded at", audio_path)

        # --------- STEP 2: Transcription ---------
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
            raise Exception("Empty transcript returned from Whisper API.")

        # --------- STEP 3: Summarization ---------
        print("STEP 3: Summarizing via Groq LLaMA...")
        llm_resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": f"Summarize:\n{transcript}"}],
                "temperature": 0.5
            }
        )

        print("LLaMA response code:", llm_resp.status_code)
        print("LLaMA response text:", llm_resp.text[:200])
        llm_resp.raise_for_status()

        data = llm_resp.json()
        if "choices" not in data or not data["choices"]:
            raise Exception(f"Unexpected response from LLaMA: {data}")

        summary = data["choices"][0]["message"]["content"]
        print("✅ Summary generated successfully")
        return {"summary": summary}

    except subprocess.CalledProcessError as e:
        print("❌ yt-dlp failed:", e.stderr)
        raise HTTPException(status_code=500, detail=f"Audio download failed: {e.stderr}")

    except Exception as e:
        print("❌ ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
