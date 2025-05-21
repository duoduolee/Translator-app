from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Load model
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

LANG_CODE = {
    "zh": "zho_Hans",  # Simplified Chinese
    "en": "eng_Latn",  # English
}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": ""})

@app.post("/translate", response_class=HTMLResponse)
async def translate(
    request: Request,
    text: str = Form(...),
    direction: str = Form(...)
):
    try:
        src, tgt = direction.split("-")
        tokenizer.src_lang = LANG_CODE[src]
        encoded = tokenizer(text, return_tensors="pt").to("cuda")
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(LANG_CODE[tgt])
        )
        translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    except Exception as e:
        translation = f"❌ Error: {str(e)}"
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": translation,
        "input_text": text,
        "selected": direction
    })

