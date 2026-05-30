import os
import json
import uuid
import logging
import re
import shutil
import requests
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from datetime import datetime

from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings

# ---------------- Config ----------------
load_dotenv()
UPLOAD_FOLDER = "uploads"
INDEX_FOLDER = "indices"
ALLOWED_EXTENSIONS = {"pdf"}
META_FILE = "hotels.json"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)

# Point static folder to React build output folder
app = Flask(__name__, static_folder="../frontend/dist", static_url_path="")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY missing in .env")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing in .env")

try:
    embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=COHERE_API_KEY)
    logger.info("Cohere embeddings initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Cohere embeddings: {e}")
    raise

INDICES = {}

# --------- Helpers ----------
def load_meta():
    """Load hotel metadata from file"""
    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in {META_FILE}, starting fresh")
            return {}
    return {}

def save_meta(data):
    """Save hotel metadata to file"""
    try:
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Metadata saved successfully")
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")
        raise

HOTELS = load_meta()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(path):
    """Extract text from PDF file with error handling"""
    try:
        reader = PdfReader(path)
        text_parts = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            except Exception as e:
                logger.warning(f"Failed to extract text from page {i} in {path}: {e}")

        if not text_parts:
            raise ValueError("No text could be extracted from the PDF")

        return "\n".join(text_parts)
    except Exception as e:
        logger.error(f"Failed to extract text from {path}: {e}")
        raise

def index_pdf(path, key):
    """Index a PDF file into vector store"""
    try:
        logger.info(f"Starting indexing for {path} with key {key}")
        text = extract_text_from_pdf(path)

        if len(text.strip()) < 100:
            raise ValueError("PDF contains too little text to index")

        splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )
        chunks = splitter.split_text(text)
        valid_chunks = [c for c in chunks if len(c.strip()) > 50]

        if not valid_chunks:
            raise ValueError("No valid text chunks found after processing")

        docs = [
            Document(
                page_content=chunk.strip(),
                metadata={
                    "source": os.path.basename(path),
                    "chunk": i,
                    "indexed_at": datetime.now().isoformat()
                }
            )
            for i, chunk in enumerate(valid_chunks)
        ]

        vs = FAISS.from_documents(docs, embeddings)
        save_path = os.path.join(INDEX_FOLDER, key)
        vs.save_local(save_path)
        INDICES[key] = vs

        logger.info(f"Successfully indexed {len(docs)} chunks for {key}")
        return len(docs)

    except Exception as e:
        logger.error(f"Failed to index PDF {path}: {e}")
        raise

def load_index(key):
    """Load vector store index for a hotel"""
    if key in INDICES:
        return INDICES[key]

    path = os.path.join(INDEX_FOLDER, key)
    if os.path.exists(path):
        try:
            vs = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
            INDICES[key] = vs
            logger.info(f"Loaded index for {key}")
            return vs
        except Exception as e:
            logger.error(f"Failed to load index for {key}: {e}")
            raise

    raise FileNotFoundError(f"No index found for hotel key: {key}")

def call_groq(messages):
    """Helper to query the Groq Cloud Chat completions API"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "temperature": 0.2
    }
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=30
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def classify_category(query):
    """Classify query into predefined categories using Groq"""
    categories = ["Rates", "Transfers", "Policies", "Offers", "General", "Amenities", "Location"]
    try:
        prompt = f"""Classify this hotel-related query into one of these categories:
Categories: {', '.join(categories)}

Query: {query}

Return only the most appropriate category name."""
        resp_content = call_groq([{"role": "user", "content": prompt}])
        category = resp_content.strip().split()[0]
        return category if category in categories else "General"
    except Exception as e:
        logger.warning(f"Failed to classify query: {e}")
        return "General"

def rag_answer(key, query):
    """Generate RAG-based answer for a query using Groq"""
    try:
        vs = load_index(key)
        retriever = vs.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)

        if not docs:
            return {
                "answer": "I couldn't find relevant information in the hotel documents for your query.",
                "citations": [],
                "category": "General"
            }

        # Build context from retrieved documents
        context_parts = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            chunk = doc.metadata.get('chunk', 0)
            content = doc.page_content
            context_parts.append(f"Source: {source} [Chunk {chunk}]\n{content}")

        context = "\n\n---\n\n".join(context_parts)
        category = classify_category(query)

        prompt = f"""You are a helpful hotel information assistant specializing in {category}. 
Answer the customer's question using ONLY the information provided in the context below.

Context from hotel documents:
{context}

Customer Question: {query}

Instructions:
- Provide a clear, helpful answer based on the context
- If the information isn't in the context, say so politely
- Be specific and reference relevant details
- Keep the response concise but informative

Answer:"""

        resp_content = call_groq([{"role": "user", "content": prompt}])

        return {
            "answer": resp_content.strip(),
            "citations": list(set([doc.metadata.get('source', 'Unknown') for doc in docs])),
            "category": category,
            "chunks_used": len(docs)
        }

    except Exception as e:
        logger.error(f"Failed to generate RAG answer: {e}")
        return {
            "answer": "I'm sorry, I encountered an error while processing your question. Please try again.",
            "citations": [],
            "category": "General"
        }

# --------- Routes ----------
@app.route("/")
def home():
    """Home page - Serve React index.html"""
    return send_from_directory(app.static_folder, "index.html")

@app.route("/chat")
def chat():
    """Chat page - Redirect to React SPA home"""
    return send_from_directory(app.static_folder, "index.html")

@app.route("/browse/<key>")
def browse(key):
    """Browse page - Redirect to React SPA home"""
    return send_from_directory(app.static_folder, "index.html")

@app.route("/chunks/<key>")
def get_chunks(key):
    """JSON API to fetch all indexed text passages for browser tab"""
    try:
        vs = load_index(key)
        queries = ["hotel information", "rates prices", "amenities facilities", "policies rules"]
        all_docs = []
        for query in queries:
            docs = vs.as_retriever(search_kwargs={"k": 10}).invoke(query)
            all_docs.extend(docs)

        seen_chunks = set()
        chunks = []
        for doc in all_docs:
            chunk_id = f"{doc.metadata.get('source', 'Unknown')}-{doc.metadata.get('chunk', 0)}"
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                chunks.append({
                    "chunk": doc.metadata.get("chunk", 0),
                    "source": doc.metadata.get("source", "Unknown"),
                    "text": doc.page_content,
                    "indexed_at": doc.metadata.get("indexed_at", "Unknown")
                })
        
        # Sort by source name and chunk number
        chunks.sort(key=lambda x: (x["source"], x["chunk"]))
        return jsonify({"chunks": chunks})
    except Exception as e:
        logger.error(f"Failed to fetch chunks for {key}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload():
    """Handle PDF upload and indexing"""
    try:
        files = request.files.getlist("files[]")
        if not files or all(f.filename == '' for f in files):
            return jsonify({"error": "No files selected"}), 400

        results = []
        errors = []

        for file in files:
            if file and file.filename and allowed_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    if not filename:
                        errors.append(f"Invalid filename: {file.filename}")
                        continue

                    # Check if file already exists
                    save_path = os.path.join(UPLOAD_FOLDER, filename)
                    if os.path.exists(save_path):
                        errors.append(f"File already exists: {filename}")
                        continue

                    file.save(save_path)

                    # Generate unique key
                    base_key = os.path.splitext(filename)[0]
                    base_key = re.sub(r'[^a-zA-Z0-9_-]', '_', base_key)
                    key = base_key
                    counter = 1
                    while key in HOTELS:
                        key = f"{base_key}_{counter}"
                        counter += 1

                    # Index the PDF
                    chunk_count = index_pdf(save_path, key)

                    # Update metadata
                    HOTELS[key] = {
                        "name": filename,
                        "files": [filename],
                        "uploaded_at": datetime.now().isoformat(),
                        "chunk_count": chunk_count,
                        "key": key
                    }
                    save_meta(HOTELS)

                    results.append({
                        "key": key,
                        "name": filename,
                        "chunks": chunk_count
                    })

                    logger.info(f"Successfully uploaded and indexed {filename} as {key}")

                except Exception as e:
                    logger.error(f"Failed to process file {file.filename}: {e}")
                    errors.append(f"Failed to process {file.filename}: {str(e)}")

                    # Cleanup on error
                    save_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
                    if os.path.exists(save_path):
                        os.remove(save_path)
            else:
                errors.append(f"Invalid file type: {file.filename}")

        response = {"indexed": results}
        if errors:
            response["errors"] = errors

        return jsonify(response)

    except Exception as e:
        logger.error(f"Upload endpoint error: {e}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route("/ask", methods=["POST"])
def ask():
    """Handle chat queries using Groq Cloud RAG"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        hotel_key = data.get("hotel")
        query = data.get("query")

        if not hotel_key or not query:
            return jsonify({"error": "Hotel key and query are required"}), 400

        if hotel_key not in HOTELS:
            return jsonify({"error": "Hotel not found"}), 404

        result = rag_answer(hotel_key, query)

        # Log the interaction
        logger.info(f"Query for {hotel_key}: {query[:100]}...")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Ask endpoint error: {e}")
        return jsonify({"error": f"Failed to process query: {str(e)}"}), 500

@app.route("/hotels")
def hotels():
    """Get list of available hotels"""
    try:
        hotel_list = [
            {
                "key": key,
                "name": info["name"],
                "uploaded_at": info.get("uploaded_at", "Unknown"),
                "chunk_count": info.get("chunk_count", 0)
            }
            for key, info in HOTELS.items()
        ]
        # Sort by upload date (newest first)
        hotel_list.sort(key=lambda x: x.get("uploaded_at", ""), reverse=True)
        return jsonify({"hotels": hotel_list})
    except Exception as e:
        logger.error(f"Hotels endpoint error: {e}")
        return jsonify({"error": "Failed to load hotels"}), 500

@app.route("/delete/<key>", methods=["DELETE"])
def delete_hotel(key):
    """Delete a hotel and its associated files"""
    try:
        if key not in HOTELS:
            return jsonify({"error": "Hotel not found"}), 404

        hotel_info = HOTELS[key]

        # Delete files
        for filename in hotel_info.get("files", []):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.exists(file_path):
                os.remove(file_path)

        # Delete index
        index_path = os.path.join(INDEX_FOLDER, key)
        if os.path.exists(index_path):
            shutil.rmtree(index_path)

        # Remove from memory and metadata
        if key in INDICES:
            del INDICES[key]

        del HOTELS[key]
        save_meta(HOTELS)

        logger.info(f"Successfully deleted hotel {key}")
        return jsonify({"message": "Hotel deleted successfully"})

    except Exception as e:
        logger.error(f"Delete endpoint error: {e}")
        return jsonify({"error": f"Failed to delete hotel: {str(e)}"}), 500

@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "hotels_count": len(HOTELS),
        "indices_loaded": len(INDICES)
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)}MB"}), 413

@app.errorhandler(Exception)
def handle_error(e):
    logger.error(f"Unhandled error: {e}")
    return jsonify({"error": "An internal error occurred"}), 500

if __name__ == "__main__":
    logger.info("Starting Hotel RAG System with Groq")
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=True, host='0.0.0.0', port=port)