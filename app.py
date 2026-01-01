import os
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

API_KEY = os.getenv("API_KEY", "dev-key")

app = FastAPI()

class Prompt(BaseModel):
    prompt: str

@app.get("/health")
def health():
    return {"status": "ok"}

def verify_key(request: Request):
    key = request.headers.get("X-API-KEY")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post("/text-to-image")
def text_to_image(data: Prompt, request: Request):
    verify_key(request)

    # MOCK response (replace with real AI later)
    return {
        "status": "success",
        "prompt": data.prompt,
        "image_url": "https://via.placeholder.com/512"
    }
