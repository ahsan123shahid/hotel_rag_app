import { useState, useEffect, useRef } from 'react';
import { 
  MessageSquare, 
  FolderOpen, 
  BookOpen, 
  UploadCloud, 
  Trash2, 
  Send, 
  Loader2, 
  Bot, 
  User, 
  CheckCircle, 
  AlertCircle, 
  Plus, 
  HelpCircle,
  FileText,
  Activity
} from 'lucide-react';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('chat'); // 'chat', 'manager', 'browser'
  const [hotels, setHotels] = useState([]);
  const [selectedHotel, setSelectedHotel] = useState('');
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [chunks, setChunks] = useState([]);
  const [chunksLoading, setChunksLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState(null); // { type: 'success'|'error', text: '' }

  const messagesEndRef = useRef(null);

  // Fetch hotels metadata from backend
  const fetchHotels = async () => {
    try {
      const res = await fetch('/hotels');
      const data = await res.json();
      if (data.hotels) {
        setHotels(data.hotels);
        // Auto-select first hotel if none selected
        if (data.hotels.length > 0 && !selectedHotel) {
          setSelectedHotel(data.hotels[0].key);
        }
      }
    } catch (err) {
      showStatus('error', 'Failed to fetch indexed documents from backend.');
    }
  };

  useEffect(() => {
    fetchHotels();
  }, []);

  // Scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Fetch document chunks for the browser tab
  const fetchChunks = async (key) => {
    if (!key) return;
    setChunksLoading(true);
    try {
      const res = await fetch(`/chunks/${key}`);
      const data = await res.json();
      if (data.chunks) {
        setChunks(data.chunks);
      } else {
        setChunks([]);
        showStatus('error', data.error || 'Failed to load document content.');
      }
    } catch (err) {
      setChunks([]);
      showStatus('error', 'Failed to connect to backend for browsing.');
    } finally {
      setChunksLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'browser' && selectedHotel) {
      fetchChunks(selectedHotel);
    }
  }, [activeTab, selectedHotel]);

  const showStatus = (type, text) => {
    setStatusMsg({ type, text });
    setTimeout(() => {
      setStatusMsg(null);
    }, 6000);
  };

  // Send a message to Groq RAG model
  const handleSendMessage = async (e) => {
    if (e) e.preventDefault();
    if (!inputText.trim() || !selectedHotel || loading) return;

    const userText = inputText;
    setInputText('');
    setMessages(prev => [...prev, { role: 'user', content: userText }]);
    setLoading(true);

    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hotel: selectedHotel, query: userText })
      });
      const data = await res.json();

      if (res.ok) {
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: data.answer,
          citations: data.citations || [],
          category: data.category || 'General'
        }]);
      } else {
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: `Error: ${data.error || 'Failed to get answer'}` 
        }]);
      }
    } catch (err) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Failed to connect to the assistant backend. Verify the server is running.' 
      }]);
    } finally {
      setLoading(false);
    }
  };

  // Upload and index a PDF brochure
  const handleFileUpload = async (files) => {
    if (!files || files.length === 0) return;
    
    // Check if files are PDFs
    const invalidFiles = Array.from(files).filter(f => !f.name.toLowerCase().endsWith('.pdf'));
    if (invalidFiles.length > 0) {
      showStatus('error', 'Only PDF files are supported.');
      return;
    }

    setUploading(true);
    const formData = new FormData();
    for (let file of files) {
      formData.append('files[]', file);
    }

    try {
      const res = await fetch('/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();

      if (res.ok) {
        let msg = `Successfully uploaded and indexed ${data.indexed.length} document(s).`;
        if (data.errors && data.errors.length > 0) {
          msg += ` Warnings: ${data.errors.join(', ')}`;
        }
        showStatus('success', msg);
        fetchHotels();
      } else {
        showStatus('error', data.error || 'Failed to upload and index files.');
      }
    } catch (err) {
      showStatus('error', 'Failed to connect to file upload server.');
    } finally {
      setUploading(false);
    }
  };

  // Delete a brochure
  const handleDeleteHotel = async (key, name) => {
    if (!confirm(`Are you sure you want to delete "${name}"? This action cannot be undone.`)) {
      return;
    }

    try {
      const res = await fetch(`/delete/${key}`, {
        method: 'DELETE'
      });
      if (res.ok) {
        showStatus('success', 'Document deleted successfully.');
        if (selectedHotel === key) {
          setSelectedHotel('');
        }
        fetchHotels();
      } else {
        const data = await res.json();
        showStatus('error', data.error || 'Failed to delete document.');
      }
    } catch (err) {
      showStatus('error', 'Failed to delete document. Server error.');
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files);
    }
  };

  const selectedHotelName = hotels.find(h => h.key === selectedHotel)?.name || 'Select a document';

  return (
    <div className="app-container">
      {/* Sidebar navigation and list */}
      <div className="sidebar">
        <div className="logo-section">
          <Bot className="logo-icon" size={32} />
          <h1 className="logo-title">Hotel Doc AI</h1>
        </div>
        <div className="logo-subtitle">RAG Document System (Groq Cloud)</div>

        <div className="nav-menu">
          <button 
            className={`nav-item ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
          >
            <MessageSquare size={18} />
            AI Chat Assistant
          </button>
          <button 
            className={`nav-item ${activeTab === 'manager' ? 'active' : ''}`}
            onClick={() => setActiveTab('manager')}
          >
            <FolderOpen size={18} />
            Document Manager
          </button>
          <button 
            className={`nav-item ${activeTab === 'browser' ? 'active' : ''}`}
            onClick={() => setActiveTab('browser')}
          >
            <BookOpen size={18} />
            Content Browser
          </button>
        </div>

        <div className="sidebar-section-title">Hotel Documents</div>
        
        {/* Document items list */}
        <div className="doc-list">
          {hotels.length === 0 ? (
            <div style={{ padding: '1rem', textAlign: 'center', color: '#64748B', fontSize: '0.85rem' }}>
              No documents uploaded yet.
            </div>
          ) : (
            hotels.map((hotel) => (
              <div 
                key={hotel.key}
                className={`doc-card ${selectedHotel === hotel.key ? 'selected' : ''}`}
                onClick={() => setSelectedHotel(hotel.key)}
              >
                <div className="doc-card-header">
                  <span className="doc-name">{hotel.name}</span>
                  <button 
                    className="delete-btn" 
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteHotel(hotel.key, hotel.name);
                    }}
                    title="Delete document"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
                <div className="doc-meta">
                  <span className="chunk-badge">{hotel.chunk_count || 0} chunks</span>
                  <span>{new Date(hotel.uploaded_at).toLocaleDateString()}</span>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Quick helper box */}
        <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid rgba(255,255,255,0.03)', borderRadius: '8px', padding: '0.75rem', fontSize: '0.75rem', color: '#64748B' }}>
          💡 Tip: Upload a PDF brochure in the Document Manager to index it, then choose it here to query specific details.
        </div>
      </div>

      {/* Main workspace container */}
      <div className="main-area">
        {/* Global status messages */}
        {statusMsg && (
          <div style={{ position: 'fixed', top: '1.5rem', right: '1.5rem', zIndex: 1000, width: '380px' }} className="animate-slideup">
            {statusMsg.type === 'success' ? (
              <div className="success-alert">
                <CheckCircle size={18} />
                <span>{statusMsg.text}</span>
              </div>
            ) : (
              <div className="error-alert">
                <AlertCircle size={18} />
                <span>{statusMsg.text}</span>
              </div>
            )}
          </div>
        )}

        {/* Tab 1: Chat Assistant */}
        {activeTab === 'chat' && (
          <div className="chat-container">
            <div className="chat-header">
              <h2 className="chat-header-title">
                <MessageSquare size={20} className="logo-icon" />
                Chatting with: <span style={{ color: '#00F2FE' }}>{selectedHotelName}</span>
              </h2>
              <div className="chat-header-status">
                <div className="status-dot active"></div>
                <span>Groq LLaMA-3.1 Active</span>
              </div>
            </div>

            {messages.length === 0 ? (
              <div className="welcome-screen">
                <div className="welcome-icon-wrapper">
                  <Bot size={40} />
                </div>
                <h1 className="welcome-title">Hotel Brochure RAG System</h1>
                <p className="welcome-desc">
                  Ask me anything about rates, services, transfer details, policies, and special packages. I extract real-time details from your uploaded PDF brochures.
                </p>
                {hotels.length > 0 && (
                  <div className="welcome-suggestions">
                    <div 
                      className="suggestion-card"
                      onClick={() => setInputText("What are the room rates and pricing options?")}
                    >
                      "What are the room rates and pricing options?"
                    </div>
                    <div 
                      className="suggestion-card"
                      onClick={() => setInputText("What transfers or shuttle services are available?")}
                    >
                      "What transfers or shuttle services are available?"
                    </div>
                    <div 
                      className="suggestion-card"
                      onClick={() => setInputText("What are the cancellation and booking policies?")}
                    >
                      "What are the cancellation and booking policies?"
                    </div>
                    <div 
                      className="suggestion-card"
                      onClick={() => setInputText("Are there any dining offers or packages?")}
                    >
                      "Are there any dining offers or packages?"
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="messages-list">
                {messages.map((msg, index) => (
                  <div key={index} className={`message-row ${msg.role} animate-slideup`}>
                    <div className="avatar">
                      {msg.role === 'user' ? <User size={18} /> : <Bot size={18} />}
                    </div>
                    <div className="bubble">
                      <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                      {msg.role === 'assistant' && (msg.citations?.length > 0 || msg.category) && (
                        <div className="bubble-meta">
                          {msg.category && (
                            <span className="category-tag">{msg.category}</span>
                          )}
                          {msg.citations?.map((cite, cIdx) => (
                            <span key={cIdx} className="citation-tag">📄 {cite}</span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                
                {/* Typing/Thinking indicator */}
                {loading && (
                  <div className="message-row assistant animate-slideup">
                    <div className="avatar">
                      <Bot size={18} />
                    </div>
                    <div className="bubble" style={{ display: 'flex', alignItems: 'center' }}>
                      <div className="typing-indicator">
                        <div className="typing-dot"></div>
                        <div className="typing-dot"></div>
                        <div className="typing-dot"></div>
                      </div>
                      <span style={{ fontSize: '0.8rem', color: '#64748B', marginLeft: '0.5rem' }}>Groq is searching vector chunks...</span>
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </div>
            )}

            {/* Chat inputs footer */}
            <form className="chat-input-bar" onSubmit={handleSendMessage}>
              <input 
                type="text" 
                className="chat-input"
                placeholder={selectedHotel ? "Ask about brochure details..." : "Please select or upload a document to start"}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                disabled={!selectedHotel || loading}
              />
              <button 
                type="submit" 
                className="send-btn"
                disabled={!selectedHotel || !inputText.trim() || loading}
              >
                <Send size={18} />
              </button>
            </form>
          </div>
        )}

        {/* Tab 2: Document Manager */}
        {activeTab === 'manager' && (
          <div className="manager-container">
            <div className="panel-header">
              <h2>📤 Upload and Index brochures</h2>
              <p style={{ color: '#94A3B8', margin: 0 }}>Add new hotel brochures in PDF format to build searchable vector indices.</p>
            </div>

            {/* Drag & Drop uploader */}
            <div 
              className={`uploader-card ${dragActive ? 'dragover' : ''}`}
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
            >
              <UploadCloud size={48} className="uploader-icon" />
              <h3>Drag and drop your PDF here</h3>
              <p style={{ color: '#64748B', margin: '0.5rem 0 1.5rem 0' }}>or click to choose files from your computer</p>
              
              <input 
                type="file" 
                className="uploader-input"
                multiple 
                accept=".pdf"
                onChange={(e) => handleFileUpload(e.target.files)}
                disabled={uploading}
              />
              
              <button className="stButton" disabled={uploading} style={{ pointerEvents: 'none' }}>
                {uploading ? 'Uploading & Indexing...' : 'Browse Files'}
              </button>

              {uploading && (
                <div className="progress-box">
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#00F2FE', marginBottom: '0.25rem' }}>
                    <span>Processing PDF chunks...</span>
                    <Loader2 className="animate-spin" size={12} style={{ animation: 'spin 1s linear infinite' }} />
                  </div>
                  <div className="progress-track">
                    <div className="progress-bar" style={{ width: '100%' }}></div>
                  </div>
                </div>
              )}
            </div>

            <div style={{ marginTop: '1rem' }}>
              <h3 style={{ fontSize: '1.1rem', color: '#E2E8F0', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '0.5rem', marginBottom: '1rem' }}>
                Active Document List
              </h3>
              {hotels.length === 0 ? (
                <p style={{ color: '#64748B', fontSize: '0.9rem' }}>No indexed documents found.</p>
              ) : (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '1rem' }}>
                  {hotels.map((hotel) => (
                    <div key={hotel.key} className="doc-card" style={{ cursor: 'default' }}>
                      <div className="doc-card-header">
                        <span className="doc-name" style={{ fontSize: '1rem' }}>📄 {hotel.name}</span>
                        <button 
                          className="delete-btn"
                          onClick={() => handleDeleteHotel(hotel.key, hotel.name)}
                          title="Delete document"
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>
                      <div className="doc-meta" style={{ marginTop: '0.75rem' }}>
                        <span className="chunk-badge">{hotel.chunk_count} passages</span>
                        <span>{new Date(hotel.uploaded_at).toLocaleString()}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tab 3: Content Browser */}
        {activeTab === 'browser' && (
          <div className="browser-container">
            <div className="browser-header">
              <h2 className="browser-title">
                <BookOpen size={20} className="logo-icon" style={{ verticalAlign: 'middle', marginRight: '0.5rem' }} />
                Document Content Explorer
              </h2>
              {hotels.length > 0 && (
                <select 
                  className="browser-selector"
                  value={selectedHotel}
                  onChange={(e) => setSelectedHotel(e.target.value)}
                >
                  {hotels.map(h => (
                    <option key={h.key} value={h.key}>{h.name}</option>
                  ))}
                </select>
              )}
            </div>

            {chunksLoading ? (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#94A3B8' }}>
                <Loader2 size={36} className="animate-spin" style={{ animation: 'spin 1.5s linear infinite', color: '#00F2FE', marginBottom: '1rem' }} />
                <span>Extracting vector database chunks...</span>
              </div>
            ) : !selectedHotel ? (
              <div className="welcome-screen">
                <BookOpen size={48} style={{ color: '#64748B', marginBottom: '1rem' }} />
                <p>No document selected or indexed. Please select a document in the selector above or upload one.</p>
              </div>
            ) : chunks.length === 0 ? (
              <div className="welcome-screen">
                <HelpCircle size={48} style={{ color: '#64748B', marginBottom: '1rem' }} />
                <p>No processed content found for this document. Verify it is indexed correctly.</p>
              </div>
            ) : (
              <div className="browser-content">
                {chunks.map((ch, idx) => (
                  <div key={idx} className="chunk-card animate-slideup">
                    <div className="chunk-card-header">
                      <span>Chunk #{ch.chunk}</span>
                      <span>Source: {ch.source}</span>
                    </div>
                    <div className="chunk-text">{ch.text}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
