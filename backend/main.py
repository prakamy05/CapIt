from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess, os, uuid, requests, time
import textwrap

# ----------------- Configuration -----------------
app = FastAPI()

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

# ----------------- Helper functions -----------------
def summarize_chunk(text_chunk):
    """Summarize a chunk using LLaMA API."""
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": f"Summarize:\n{text_chunk}"}],
                "temperature": 0.5
            }
        )
        resp.raise_for_status()
        data = resp.json()
        if "choices" not in data or not data["choices"]:
            raise Exception(f"Unexpected LLaMA response: {data}")
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print("❌ Chunk summarization failed:", e)
        return f"[Error summarizing chunk: {e}]"

def chunk_text(text, max_chars=5000):
    """Split text into chunks of max_chars length."""
    return textwrap.wrap(text, max_chars)

# ----------------- Summarize endpoint -----------------
@app.post("/summarize")
def summarize(req: VideoRequest):
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())
    audio_path = f"/tmp/audio_{timestamp}_{unique_id}.mp3"

    try:
        # --------- STEP 1: Download audio ---------
        try:
            print("STEP 1: Downloading audio...")
            result = subprocess.run(
                [
                    "yt-dlp", "--extract-audio", "--audio-format", "mp3",
                    "--cookies", "cookies.txt",
                    "-o", f"/tmp/audio_{timestamp}_{unique_id}.%(ext)s", req.url
                ],
                check=True,
                capture_output=True,
                text=True
            )
            print("yt-dlp output:", result.stdout)
            if not os.path.exists(audio_path):
                raise Exception(f"Audio file not found at {audio_path}")
            print("✅ Audio downloaded at", audio_path)
        except subprocess.CalledProcessError as e:
            print("❌ yt-dlp failed:", e.stderr)
            return {"error": f"Audio download failed: {e.stderr}"}

        # --------- STEP 2: Transcription ---------
        try:
            print("STEP 2: Transcribing via Groq Whisper...")
            with open(audio_path, "rb") as f:
                whisper_resp = requests.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    files={"file": f},
                    data={"model": "whisper-large-v3"}
                )

            whisper_resp.raise_for_status()
            transcript = whisper_resp.json().get("text", "")
            if not transcript:
                raise Exception("Empty transcript returned from Whisper API.")
            print("✅ Transcription completed. Transcript length:", len(transcript))
        except Exception as e:
            print("❌ Transcription failed:", e)
            return {"error": f"Transcription failed: {e}"}

        # --------- STEP 3: Summarization in batches ---------
        try:
            print("STEP 3: Splitting transcript into chunks for summarization...")
            chunks = chunk_text(transcript, max_chars=5000)
            print(f"Transcript split into {len(chunks)} chunks")

            partial_summaries = []
            for idx, chunk in enumerate(chunks, start=1):
                print(f"Summarizing chunk {idx}/{len(chunks)}...")
                summary = summarize_chunk(chunk)
                partial_summaries.append(summary)

            # Concatenate partial summaries
            final_summary = "\n\n".join(partial_summaries)
            print("✅ Final summary generated successfully")
            return {"summary": final_summary}

        except Exception as e:
            print("❌ Summarization failed:", e)
            return {"error": f"Summarization failed: {e}"}

    finally:
        # --------- Cleanup audio file ---------
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                print("✅ Temporary audio file deleted:", audio_path)
            except Exception as e:
                print("⚠️ Failed to delete temp file:", e)
