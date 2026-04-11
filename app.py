from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import shutil
import os

from rag import process_pdf, ask_question

app = FastAPI()
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    process_pdf(file_path)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": "PDF uploaded successfully!"
    })


@app.post("/ask", response_class=HTMLResponse)
def ask(request: Request, question: str = Form(...)):

    answer = ask_question(question)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "question": question,
        "answer": answer
    })