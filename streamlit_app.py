import streamlit as st
import asyncio
import os
import time

# --- PARCHE PARA EL EVENT LOOP ---
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
# ---------------------------------

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from streamlit_agraph import agraph, Node, Edge, Config

# Configuración de página
st.set_page_config(page_title="Chat con Christian Silva", page_icon="⚡", layout="wide")

# --- ESTILOS CSS CORREGIDOS Y MEJORADOS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    /* FONDO GLOBAL */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(15, 23, 42) 0%, rgb(10, 15, 30) 90%);
        font-family: 'Poppins', sans-serif;
    }

    /* --- SCROLLBARS PERSONALIZADOS (ESTILO BLANCO/GRIS) --- */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
        background: #0f172a;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 5px;
        border: 1px solid #1e293b;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #f97316; /* Naranja al pasar el mouse */
    }

    /* --- ICONO DEL DASHBOARD (SIDEBAR) --- */
    /* Este filtro vuelve la imagen totalmente blanca */
    [data-testid="stSidebar"] img {
        filter: brightness(0) invert(1);
        opacity: 0.9;
    }

    /* --- TEXTOS GENERALES --- */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #f97316, #facc15);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
    }
    .stMarkdown p, .stMarkdown li, .stText, p, label {
        color: #e2e8f0 !important;
        font-size: 1.05rem;
        line-height: 1.7;
    }

    /* --- INPUTS Y TEXTAREAS (CORRECCIÓN DE COLOR) --- */
    /* Texto que escribe el usuario: BLANCO */
    .stTextInput input, .stChatInput textarea, .stTextArea textarea {
        background-color: rgba(30, 41, 59, 0.8) !important;
        color: #ffffff !important; /* Texto blanco puro */
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        caret-color: #f97316; /* Cursor naranja */
    }
    
    /* Placeholder (Texto de referencia/ejemplo): BLANCO TRANSLÚCIDO */
    .stTextInput input::placeholder, 
    .stChatInput textarea::placeholder, 
    .stTextArea textarea::placeholder {
        color: rgba(255, 255, 255, 0.8) !important; /* Blanco visible al 80% */
        opacity: 1; /* Forzar opacidad en algunos navegadores */
    }

    /* Focus (cuando haces click) */
    .stChatInput textarea:focus, .stTextArea textarea:focus {
        border-color: #f97316 !important;
        box-shadow: 0 0 0 1px rgba(249, 115, 22, 0.5);
    }

    /* --- BARRA LATERAL --- */
    [data-testid="stSidebar"] {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important; /* Gris muy claro */
    }
    [data-testid="stMetricValue"] {
        color: #f97316 !important; /* Naranja neón */
    }

    /* --- PESTAÑAS (TABS) --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        padding: 10px 0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        white-space: pre-wrap;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: #ffffff;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important;
        color: white !important;
        border: none;
        box-shadow: 0 4px 15px rgba(249, 115, 22, 0.3);
    }

    /* --- MENSAJES DE CHAT --- */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
    }

    /* --- BOTONES --- */
    .stButton button {
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
        color: white;
        font-weight: 700;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(249, 115, 22, 0.3);
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(249, 115, 22, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# --- INICIALIZACIÓN DE VARIABLES ---
if "api_calls" not in st.session_state:
    st.session_state.api_calls = 0
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "last_latency" not in st.session_state:
    st.session_state.last_latency = 0.0

st.title("⚡ Asistente IA de Christian Silva")
st.markdown("Potenciado por **Groq (Llama 3.3)** + **Embeddings Locales**.")

# --- API KEY ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
except FileNotFoundError:
    st.error("⚠️ No se encontró la GROQ_API_KEY.")
    st.stop()

# --- SIDEBAR DASHBOARD ---
with st.sidebar:
    # IMAGEN: Ahora se verá BLANCA gracias al filtro CSS
    st.image("https://cdn-icons-png.flaticon.com/512/10308/10308068.png", width=70)
    
    st.header("Métricas en Vivo")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    col1.metric("Llamadas", st.session_state.api_calls)
    col2.metric("Latencia", f"{st.session_state.last_latency:.2f}s")
    
    st.metric("Tokens Procesados", f"{st.session_state.total_tokens:,}")
    
    savings = (st.session_state.total_tokens / 1000) * 0.03 
    st.metric("Ahorro vs GPT-4", f"${savings:.4f}")
    
    st.markdown("---")
    st.caption("Stack: Python 3.11 | LangChain | Groq | Streamlit")

# --- FUNCIONES ---
@st.cache_resource
def load_and_process_pdf(pdf_path):
    text = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo PDF.")
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    if not chunks:
        return None

    try:
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error embeddings: {e}")
        return None

def get_conversation_chain(vectorstore):
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return chain

def update_metrics(start_time, prompt_len, response_len):
    end_time = time.time()
    st.session_state.last_latency = end_time - start_time
    st.session_state.api_calls += 1
    estimated_tokens = (prompt_len + response_len) // 4
    st.session_state.total_tokens += estimated_tokens

# --- CARGA INICIAL ---
if "conversation" not in st.session_state:
    with st.spinner("Inicializando sistemas..."):
        try:
            vectorstore = load_and_process_pdf("cv_csilva.pdf")
            if vectorstore:
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.process_complete = True
                st.toast("¡Sistema listo!", icon="✅")
        except Exception as e:
            st.error(f"Error: {e}")

# --- PESTAÑAS Y LÓGICA ---
if "process_complete" in st.session_state:
    
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🎯 Cartas", "🕸️ Habilidades"])

    # 1. CHAT
    with tab1:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input("Escribe tu pregunta aquí...", max_chars=1000):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    try:
                        start_time = time.time()
                        response = st.session_state.conversation({'question': prompt})
                        ai_response = response['answer']
                        update_metrics(start_time, len(prompt), len(ai_response))
                        st.write(ai_response)
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    # 2. GENERADOR
    with tab2:
        st.header("Generador de Cartas")
        st.markdown("Pega una oferta de trabajo para generar una carta personalizada.")
        
        job_description = st.text_area("Descripción de la Oferta:", height=250, placeholder="Ej: Buscamos Ingeniero de Datos con experiencia en Python...", max_chars=4000)
        
        if st.button("🚀 Generar Carta"):
            if job_description:
                with st.spinner("Redactando..."):
                    try:
                        start_time = time.time()
                        prompt_carta = f"Actúa como el candidato Christian Silva. Contexto: Mi CV. Tarea: Escribir carta de presentación para esta oferta: {job_description}. Tono: Profesional."
                        response = st.session_state.conversation({'question': prompt_carta})
                        ai_response = response['answer']
                        update_metrics(start_time, len(prompt_carta), len(ai_response))
                        
                        st.subheader("Carta Generada:")
                        st.markdown(ai_response)
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("⚠️ Pega la oferta primero.")

    # 3. GRAFO
    with tab3:
        st.header("Mapa de Habilidades")
        
        nodes = []
        edges = []
        
        # Nodos
        nodes.append(Node(id="Yo", label="Christian Silva", size=45, color="#f97316", shape="circularImage", image="https://cdn-icons-png.flaticon.com/512/3135/3135715.png"))
        nodes.append(Node(id="AI", label="Artificial Intelligence", color="#3b82f6", size=30))
        nodes.append(Node(id="ML", label="Machine Learning", color="#3b82f6"))
        nodes.append(Node(id="RAG", label="RAG Systems", color="#3b82f6"))
        nodes.append(Node(id="NLP", label="NLP & LLMs", color="#3b82f6"))
        nodes.append(Node(id="Py", label="Python Ecosystem", color="#10b981", size=30))
        nodes.append(Node(id="Data", label="Data Engineering", color="#10b981"))
        nodes.append(Node(id="SQL", label="SQL & Databases", color="#10b981"))
        nodes.append(Node(id="St", label="Streamlit / Web", color="#10b981"))
        nodes.append(Node(id="Cloud", label="Cloud Services", color="#8b5cf6", size=25))
        nodes.append(Node(id="Soft", label="Soft Skills", color="#f59e0b", size=25))
        nodes.append(Node(id="Com", label="Comunicación", color="#f59e0b"))
        nodes.append(Node(id="Prob", label="Resolución de Problemas", color="#f59e0b"))

        # Aristas
        edges.append(Edge(source="Yo", target="AI", label="Especialidad"))
        edges.append(Edge(source="Yo", target="Py", label="Core"))
        edges.append(Edge(source="AI", target="ML", label="Base"))
        edges.append(Edge(source="AI", target="RAG", label="Implementa"))
        edges.append(Edge(source="AI", target="NLP", label="Usa"))
        edges.append(Edge(source="Py", target="Data", label="Aplica"))
        edges.append(Edge(source="Py", target="St", label="Crea"))
        edges.append(Edge(source="Data", target="SQL", label="Consulta"))
        edges.append(Edge(source="Yo", target="Cloud", label="Despliega"))
        edges.append(Edge(source="Yo", target="Soft", label="Posee"))
        edges.append(Edge(source="Soft", target="Com", label="Clave"))
        edges.append(Edge(source="Soft", target="Prob", label="Enfoque"))

        config = Config(
            width=900,
            height=600,
            directed=True, 
            physics=True, 
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#f97316",
            collapsible=False,
            node={'labelProperty': 'label', 'renderLabel': True}
        )
        
        agraph(nodes=nodes, edges=edges, config=config)
