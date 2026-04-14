from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

model = SentenceTransformer('all-MiniLM-L6-v2')
qa_pipeline = pipeline("text-generation", model="google/flan-t5-base")

chunks = []
index = None


def process_pdf(file):

    global chunks, index

    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))


def ask_question(question):

    question_embedding = model.encode([question])
    D, I = index.search(np.array(question_embedding), k=1)

    context = chunks[I[0][0]]

    prompt = f"""
    Answer the question based on the context below:

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = qa_pipeline(prompt, max_length=200)

    return response[0]['generated_text']