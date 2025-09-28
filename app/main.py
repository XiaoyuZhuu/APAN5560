from fastapi import FastAPI
from pydantic import BaseModel
from app.bigram_model import BigramModel
import spacy

# Initialize FastAPI app
app = FastAPI()

# ---------- Bigram Section ----------
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]
bigram_model = BigramModel(corpus)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

# ---------- Embedding Section ----------
nlp = spacy.load("en_core_web_lg")

class EmbeddingRequest(BaseModel):
    word: str

@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    doc = nlp(request.word)
    embedding = doc.vector.tolist()
    return {
        "word": request.word,
        "embedding_dim": len(embedding),
        "embedding": embedding[:10]  # 返回前10个数，避免太长
    }
