from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import fitz

app = FastAPI()

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qa_model = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"  
)

def load_pdf_chunks(path: str, chunk_size: int = 500):
    """Load PDF and split text into chunks of a given size"""
    doc = fitz.open(path)
    chunks = []
    for page in doc:
        text = page.get_text("text")
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            if chunk.strip():
                chunks.append(chunk)
    return chunks

pdf_path = "healthGuard.pdf"
chunks = load_pdf_chunks(pdf_path)

embeddings = embedding_model.encode(chunks, convert_to_tensor=False)
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def search_context(query: str, top_k: int = 3):
    """Return the most relevant PDF chunks for a query"""
    query_vec = embedding_model.encode([query], convert_to_tensor=False)
    D, I = index.search(query_vec, top_k)
    return [chunks[i] for i in I[0]]

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask(req: Question):
    """Return answer using RAG (retrieval + generation)"""
    retrieved = search_context(req.question)
    context = "\n\n".join(retrieved)

    # ---- 6a. Build prompt in Portuguese ----
    prompt = f"""
        Você é um assistente especialista na máquina Midea HealthGuard.
        Responda de forma clara e objetiva usando SOMENTE o manual abaixo.
        Se não houver resposta no manual, diga que não está no manual e sugira contatar a assistência técnica.
        RESPONDER EM PORTUGUES BRASIL
        
        Manual (trechos relevantes):
        {context}
        
        Pergunta: {req.question}
        """

    output = qa_model(prompt, max_length=300, do_sample=True, temperature=0.3)
    answer = output[0]["generated_text"]

    return {"question": req.question, "answer": answer, "retrieved_chunks": retrieved}
