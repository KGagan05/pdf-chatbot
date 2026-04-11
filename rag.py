from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Store data
chunks = []
index = None

def process_pdf(file):

    global chunks, index

    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    # Split into chunks
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))


def ask_question(question):

    question_embedding = model.encode([question])

    D, I = index.search(np.array(question_embedding), k=1)

    return chunks[I[0][0]]