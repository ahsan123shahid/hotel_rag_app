import os
import json
import re
import shutil
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_core.messages import HumanMessage

# ----------------- Configuration & Directories -----------------
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
INDEX_FOLDER = os.path.join(BASE_DIR, "indices")
META_FILE = os.path.join(BASE_DIR, "hotels.json")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)

# Set Page Config
st.set_page_config(
    page_title="Hotel Document AI",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styles (Oceanic Navy & Glassmorphism Theme)
st.markdown(
    """
    <style>
        /* General layout & background */
        .stApp {
            background-color: #0A1128;
            color: #E2E8F0;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: rgba(16, 23, 42, 0.95) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
            padding-top: 2rem;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #00F2FE !important;
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
        }
        
        /* Gradient header text */
        .gradient-text {
            background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
        }
        
        /* Buttons styling */
        .stButton>button {
            background: linear-gradient(135deg, #4FACFE 0%, #00F2FE 100%) !important;
            color: #0A1128 !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            padding: 0.6rem 1.2rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 14px rgba(0, 242, 254, 0.2);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 242, 254, 0.4);
            color: #0A1128 !important;
        }
        
        /* Card container widgets */
        .hotel-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        .hotel-card:hover {
            border-color: rgba(0, 242, 254, 0.3);
            box-shadow: 0 8px 30px rgba(0, 242, 254, 0.05);
            background: rgba(255, 255, 255, 0.05);
        }
        
        /* Badges */
        .badge {
            background: rgba(0, 242, 254, 0.1);
            color: #00F2FE;
            border: 1px solid rgba(0, 242, 254, 0.2);
            padding: 0.2rem 0.6rem;
            border-radius: 50px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        
        /* Chat bubble styling overrides */
        .stChatMessage {
            background-color: rgba(255, 255, 255, 0.02) !important;
            border: 1px solid rgba(255, 255, 255, 0.03) !important;
            border-radius: 12px !important;
            padding: 1rem !important;
            margin-bottom: 0.8rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- Helper Functions -----------------
def load_meta():
    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_meta(data):
    try:
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Failed to save metadata: {e}")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "pdf"

def extract_text_from_pdf(path):
    try:
        reader = PdfReader(path)
        text_parts = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                text_parts.append(text)
        if not text_parts:
            raise ValueError("No text could be extracted from the PDF")
        return "\n".join(text_parts)
    except Exception as e:
        st.error(f"Failed to extract text: {e}")
        raise

def index_pdf(file_path, key, cohere_key):
    try:
        text = extract_text_from_pdf(file_path)
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
                    "source": os.path.basename(file_path),
                    "chunk": i,
                    "indexed_at": datetime.now().isoformat()
                }
            )
            for i, chunk in enumerate(valid_chunks)
        ]

        embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=cohere_key)
        vs = FAISS.from_documents(docs, embeddings)
        save_path = os.path.join(INDEX_FOLDER, key)
        vs.save_local(save_path)
        return len(docs)
    except Exception as e:
        st.error(f"Indexing failed: {e}")
        raise

def rag_answer(key, query, cohere_key):
    try:
        path = os.path.join(INDEX_FOLDER, key)
        embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=cohere_key)
        vs = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        retriever = vs.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)

        if not docs:
            return {
                "answer": "I couldn't find relevant information in the hotel documents for your query.",
                "citations": [],
                "category": "General"
            }

        context_parts = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            chunk = doc.metadata.get('chunk', 0)
            content = doc.page_content
            context_parts.append(f"Source: {source} [Chunk {chunk}]\n{content}")

        context = "\n\n---\n\n".join(context_parts)

        # Classify query
        llm = ChatCohere(model="command-r-plus-08-2024", cohere_api_key=cohere_key)
        categories = ["Rates", "Transfers", "Policies", "Offers", "General", "Amenities", "Location"]
        try:
            classify_prompt = f"Classify this hotel-related query into one of these categories: {', '.join(categories)}\n\nQuery: {query}\n\nReturn only the category name."
            resp = llm.invoke([HumanMessage(content=classify_prompt)])
            category = resp.content.strip().split()[0]
            if category not in categories:
                category = "General"
        except Exception:
            category = "General"

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

        resp = llm.invoke([HumanMessage(content=prompt)])
        return {
            "answer": resp.content.strip(),
            "citations": list(set([doc.metadata.get('source', 'Unknown') for doc in docs])),
            "category": category,
            "chunks_used": len(docs),
            "retrieved_docs": docs
        }
    except Exception as e:
        return {
            "answer": f"I'm sorry, I encountered an error: {e}. Please double-check your API key.",
            "citations": [],
            "category": "General"
        }

# ----------------- API Key Resolution -----------------
# 1. Environment Variable, 2. Streamlit Secrets, 3. Sidebar Input
cohere_key = os.getenv("COHERE_API_KEY")
if not cohere_key:
    try:
        cohere_key = st.secrets.get("COHERE_API_KEY")
    except Exception:
        pass

# Sidebar API Key configuration
with st.sidebar:
    st.markdown('<div class="gradient-text">🏨 Hotel RAG</div>', unsafe_allow_html=True)
    st.caption("AI-Powered Brochure Assistant")
    st.divider()
    
    if not cohere_key:
        st.warning("⚠️ COHERE_API_KEY is not set.")
        key_input = st.text_input("Enter Cohere API Key:", type="password")
        if key_input:
            cohere_key = key_input
            st.success("API Key applied locally!")
    else:
        st.success("🔒 Cohere API Key is active")
        
    st.divider()
    st.markdown("### Quick Navigation")
    tab_selection = st.radio("Go to:", ["💬 AI Chat Assistant", "📁 Document Manager"])

HOTELS = load_meta()

# ----------------- Tab 1: AI Chat Assistant -----------------
if tab_selection == "💬 AI Chat Assistant":
    st.title("💬 Chat with Hotel Documents")
    st.write("Select a brochure and ask questions about amenities, rates, transfers, policies, and more.")
    
    if not HOTELS:
        st.info("No documents indexed yet. Go to 'Document Manager' to upload brochures!")
    else:
        # Dropdown to select a hotel brochure
        hotel_options = {HOTELS[key]["name"]: key for key in HOTELS}
        selected_hotel_name = st.selectbox("Choose Hotel Document:", list(hotel_options.keys()))
        selected_hotel_key = hotel_options[selected_hotel_name]
        
        # Reset chat history if switching hotels
        if "last_hotel" not in st.session_state or st.session_state.last_hotel != selected_hotel_key:
            st.session_state.chat_history = []
            st.session_state.last_hotel = selected_hotel_key
            
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("citations"):
                    st.markdown(f"**Sources:** {', '.join(msg['citations'])} | **Category:** `{msg.get('category')}`")
        
        # Chat input box
        user_query = st.chat_input("Ask a question about the hotel...")
        
        if user_query:
            # Display user message
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
                
            # Perform RAG Query
            if not cohere_key:
                st.error("Please configure a Cohere API Key in the sidebar to chat.")
            else:
                with st.spinner("Analyzing document and generating answer..."):
                    result = rag_answer(selected_hotel_key, user_query, cohere_key)
                    
                # Display Assistant Answer
                with st.chat_message("assistant"):
                    st.markdown(result["answer"])
                    if result["citations"]:
                        st.markdown(f"**Sources:** {', '.join(result['citations'])} | **Category:** `{result['category']}`")
                        
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "citations": result["citations"],
                    "category": result["category"]
                })

# ----------------- Tab 2: Document Manager -----------------
elif tab_selection == "📁 Document Manager":
    st.title("📁 Document Manager")
    
    upload_tab, manage_tab = st.tabs(["📤 Upload PDF", "📋 Manage Current Documents"])
    
    # Sub-tab: Upload PDF
    with upload_tab:
        st.subheader("Upload Hotel Brochures")
        uploaded_files = st.file_uploader(
            "Select one or more hotel brochure PDFs", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        
        if st.button("Upload & Index Documents"):
            if not uploaded_files:
                st.warning("Please select at least one PDF file first.")
            elif not cohere_key:
                st.error("Cohere API Key is required to create document embeddings.")
            else:
                for uploaded_file in uploaded_files:
                    with st.spinner(f"Processing and indexing: {uploaded_file.name}..."):
                        # Clean filename
                        filename = re.sub(r'[^a-zA-Z0-9._-]', '_', uploaded_file.name)
                        save_path = os.path.join(UPLOAD_FOLDER, filename)
                        
                        # Write file locally
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                            
                        # Generate hotel key
                        base_key = os.path.splitext(filename)[0]
                        base_key = re.sub(r'[^a-zA-Z0-9_-]', '_', base_key)
                        key = base_key
                        counter = 1
                        while key in HOTELS:
                            key = f"{base_key}_{counter}"
                            counter += 1
                            
                        try:
                            # Index the file
                            chunk_count = index_pdf(save_path, key, cohere_key)
                            
                            # Save to metadata
                            HOTELS[key] = {
                                "name": uploaded_file.name,
                                "files": [filename],
                                "uploaded_at": datetime.now().isoformat(),
                                "chunk_count": chunk_count,
                                "key": key
                            }
                            save_meta(HOTELS)
                            st.success(f"Successfully indexed '{uploaded_file.name}' into {chunk_count} text chunks!")
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {e}")
                            if os.path.exists(save_path):
                                os.remove(save_path)
                
                # Refresh metadata
                st.rerun()
                
    # Sub-tab: Manage Current Documents
    with manage_tab:
        st.subheader("Your Indexed Hotel Brochures")
        
        if not HOTELS:
            st.info("No documents are currently indexed.")
        else:
            for key, info in list(HOTELS.items()):
                # Create HTML layout for cards using custom styling
                st.markdown(
                    f"""
                    <div class="hotel-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="font-size: 1.2rem; font-weight: bold; color: #00F2FE;">🏨 {info['name']}</span><br>
                                <span style="color: #94A3B8; font-size: 0.9rem;">
                                    Uploaded: {datetime.fromisoformat(info.get('uploaded_at', '')).strftime('%Y-%m-%d %H:%M:%S')}
                                </span>
                            </div>
                            <div>
                                <span class="badge">{info.get('chunk_count', 0)} chunks</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Action Buttons
                c1, c2, _ = st.columns([1, 1, 6])
                
                with c1:
                    # View details
                    if st.button("Browse Chunks", key=f"browse_{key}"):
                        st.session_state.browsing_key = key
                with c2:
                    # Delete document
                    if st.button("Delete Document", key=f"del_{key}"):
                        # Remove files
                        for filename in info.get("files", []):
                            file_path = os.path.join(UPLOAD_FOLDER, filename)
                            if os.path.exists(file_path):
                                os.remove(file_path)
                        
                        # Remove indices
                        index_path = os.path.join(INDEX_FOLDER, key)
                        if os.path.exists(index_path):
                            shutil.rmtree(index_path)
                            
                        # Delete metadata
                        del HOTELS[key]
                        save_meta(HOTELS)
                        st.warning(f"Deleted index for {info['name']}.")
                        st.rerun()
            
            # Browser Drawer Panel
            if "browsing_key" in st.session_state and st.session_state.browsing_key in HOTELS:
                br_key = st.session_state.browsing_key
                br_info = HOTELS[br_key]
                st.divider()
                st.subheader(f"🔍 Browsing Content for: {br_info['name']}")
                
                # Retrieve chunks of the document
                try:
                    embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=cohere_key)
                    vs = FAISS.load_local(os.path.join(INDEX_FOLDER, br_key), embeddings, allow_dangerous_deserialization=True)
                    
                    # Search general chunks
                    docs = vs.as_retriever(search_kwargs={"k": 6}).invoke("hotel info")
                    
                    if not docs:
                        st.write("No text chunks found in vector index.")
                    else:
                        for idx, doc in enumerate(docs):
                            with st.expander(f"Chunk {doc.metadata.get('chunk', idx)} (Source: {doc.metadata.get('source')})"):
                                st.write(doc.page_content)
                except Exception as e:
                    st.error(f"Could not load index to browse: {e}. Check Cohere API key.")
                
                if st.button("Close Browser"):
                    del st.session_state.browsing_key
                    st.rerun()
