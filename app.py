import os
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
from diffusers import StableDiffusionPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from io import BytesIO

# -------------------------
# CONFIG
# -------------------------
API_KEY = os.getenv("API_KEY", "dev-key")
device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="AI Image API")

# -------------------------
# MODELS
# -------------------------
sd = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    torch_dtype=torch.float16,
    device_map="auto",
    use_safetensors=True
)

# -------------------------
# AUTH
# -------------------------
def check_key(x_api_key: str):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# -------------------------
# SCHEMAS
# -------------------------
class TextPrompt(BaseModel):
    prompt: str

# -------------------------
# ROUTES
# -------------------------

@app.post("/text-to-image")
def text_to_image(data: TextPrompt, x_api_key: str = Header(None)):
    check_key(x_api_key)

    image = sd(data.prompt, num_inference_steps=30).images[0]
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.post("/image-to-text")
async def image_to_text(
    file: UploadFile = File(...),
    x_api_key: str = Header(None)
):
    check_key(x_api_key)

    image = Image.open(file.file).convert("RGB")
    inputs = blip_processor(image, return_tensors="pt").to(device)

    output = blip_model.generate(
        **inputs,
        max_length=30,
        num_beams=5,
        repetition_penalty=1.5
    )

    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return {"caption": caption}