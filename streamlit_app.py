import streamlit as st
import asyncio
import os
import time  # <--- NUEVO: Para medir el tiempo de respuesta

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

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
<style>
    /* Fondo oscuro global */
    .stApp { background-color: #0f172a; }
    
    /* Títulos en naranja Groq */
    h1, h2, h3 { color: #f97316 !important; }
    
    /* TEXTO BLANCO Y LEGIBLE */
    .stMarkdown p, .stMarkdown li, .stText, p {
        color: #ffffff !important;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    /* Pestañas (Tabs) */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e293b;
        border-radius: 5px;
        color: #ffffff;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f97316 !important;
        color: white !important;
    }
    
    /* Mtricas en Sidebar */
    [data-testid="stMetricValue"] {
        color: #f97316 !important;
    }
    
    /* Cajitas de los mensajes */
    .stChatMessage { 
        background-color: #1e293b; 
        border: 1px solid #334155;
        border-radius: 10px;
    }
    
    /* INPUTS */
    .stTextInput input, .stChatInput textarea, .stTextArea textarea { 
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #334155;
        caret-color: #f97316;
    }
    
    /* Botones */
    .stButton button {
        background-color: #f97316;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .stButton button:hover {
        background-color: #ea580c;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- INICIALIZACIÓN DE VARIABLES DE ESTADO (MÉTRICAS) ---
if "api_calls" not in st.session_state:
    st.session_state.api_calls = 0
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "last_latency" not in st.session_state:
    st.session_state.last_latency = 0.0

st.title("⚡ Asistente IA de Christian Silva")
st.write("Potenciado por **Groq (Llama 3.3)** + **Embeddings Locales**.")

# --- GESTIÓN DE LA API KEY ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
except FileNotFoundError:
    st.error("⚠️ No se encontró la GROQ_API_KEY. Configura los 'Secrets' en Streamlit Cloud.")
    st.stop()

# --- SIDEBAR: DASHBOARD DE INGENIERÍA ---
with st.sidebar:
    st.header("📊 Métricas de Ingeniería")
    st.markdown("---")
    
    # Métricas en tiempo real
    col1, col2 = st.columns(2)
    col1.metric("Llamadas API", st.session_state.api_calls)
    col2.metric("Latencia (s)", f"{st.session_state.last_latency:.2f}s")
    
    st.metric("Tokens Procesados (Est.)", st.session_state.total_tokens)
    
    # Cálculo ficticio de ahorro vs GPT-4 ($30/millón tokens vs casi gratis)
    savings = (st.session_state.total_tokens / 1000) * 0.03 
    st.metric("Ahorro de Costos vs GPT-4", f"${savings:.4f}")
    
    st.markdown("---")
    st.markdown("**Stack Tecnológico:**")
    st.code("Python 3.11\nLangChain\nGroq LPU\nFAISS (Vector DB)\nStreamlit", language="text")
    st.caption("Este dashboard demuestra observabilidad en tiempo real.")

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
        st.error("❌ No se encontró el archivo PDF. Asegúrate de que 'cv_csilva.pdf' está en la carpeta raíz.")
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
        st.error(f"Error al crear embeddings locales: {e}")
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

# Función auxiliar para actualizar métricas
def update_metrics(start_time, prompt_len, response_len):
    end_time = time.time()
    st.session_state.last_latency = end_time - start_time
    st.session_state.api_calls += 1
    # Estimación simple: 1 token ~= 4 caracteres
    estimated_tokens = (prompt_len + response_len) // 4
    st.session_state.total_tokens += estimated_tokens

# --- INICIALIZACIÓN ---

if "conversation" not in st.session_state:
    with st.spinner("Cargando cerebro digital..."):
        try:
            vectorstore = load_and_process_pdf("cv_csilva.pdf")
            if vectorstore:
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.process_complete = True
                st.toast("¡Sistema listo!", icon="🚀")
        except Exception as e:
            st.error(f"Ocurrió un error al iniciar: {e}")

# --- INTERFAZ PRINCIPAL CON PESTAÑAS ---

if "process_complete" in st.session_state:
    
    tab1, tab2, tab3 = st.tabs(["💬 Chat Asistente", "📝 Generador de Cartas", "🕸️ Mapa de Habilidades"])

    # --- PESTAÑA 1: CHAT ---
    with tab1:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input("Pregúntame sobre mis proyectos o experiencia...", max_chars=1000):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Procesando..."):
                    try:
                        start_time = time.time() # Inicia cronómetro
                        
                        response = st.session_state.conversation({'question': prompt})
                        ai_response = response['answer']
                        
                        # Actualizar métricas
                        update_metrics(start_time, len(prompt), len(ai_response))
                        
                        st.write(ai_response)
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
                        
                        # Forzar actualización de la sidebar
                        st.rerun() 
                        
                    except Exception as e:
                        st.error(f"Error: {e}")

    # --- PESTAÑA 2: GENERADOR DE CANDIDATURAS ---
    with tab2:
        st.header("🎯 Generador de Cartas de Presentación")
        st.markdown("Pega aquí la descripción de la oferta y generaré una carta personalizada basada en mi CV.")
        
        job_description = st.text_area("Descripción de la Oferta:", height=200, placeholder="Pega aquí los requisitos del puesto...", max_chars=3000)
        
        if st.button("🚀 Generar Carta Personalizada"):
            if job_description:
                with st.spinner("Redactando..."):
                    try:
                        start_time = time.time() # Inicia cronómetro
                        
                        prompt_carta = f"Actúa como el candidato. Analiza esta oferta: {job_description}. Basado en mi CV (contexto), escribe una carta de presentación persuasiva."
                        response = st.session_state.conversation({'question': prompt_carta})
                        ai_response = response['answer']
                        
                        # Actualizar métricas
                        update_metrics(start_time, len(prompt_carta), len(ai_response))
                        
                        st.subheader("Tu Carta Generada:")
                        st.markdown(ai_response)
                        st.balloons()
                        
                        # Nota: Aquí no hacemos st.rerun() completo para no borrar la carta generada, 
                        # pero las métricas se actualizarán en la siguiente interacción.
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("⚠️ Pega la descripción de la oferta primero.")

    # --- PESTAÑA 3: GRAFO DE CONOCIMIENTO ---
    with tab3:
        st.header("🕸️ Mapa de Habilidades Interactivo")
        st.markdown("Explora mis conexiones técnicas. ¡Puedes arrastrar los nodos!")
        
        nodes = []
        edges = []
        
        # Nodos
        nodes.append(Node(id="Yo", label="Christian Silva", size=40, color="#f97316"))
        nodes.append(Node(id="AI", label="Artificial Intelligence", color="#3b82f6"))
        nodes.append(Node(id="ML", label="Machine Learning", color="#3b82f6"))
        nodes.append(Node(id="RAG", label="RAG Systems", color="#3b82f6"))
        nodes.append(Node(id="NLP", label="NLP", color="#3b82f6"))
        nodes.append(Node(id="Py", label="Python", color="#10b981"))
        nodes.append(Node(id="SQL", label="SQL", color="#10b981"))
        nodes.append(Node(id="St", label="Streamlit", color="#10b981"))
        nodes.append(Node(id="Git", label="Git/GitHub", color="#10b981"))
        nodes.append(Node(id="Com", label="Comunicación", color="#8b5cf6"))
        nodes.append(Node(id="Led", label="Liderazgo", color="#8b5cf6"))
        nodes.append(Node(id="Prob", label="Resolución Problemas", color="#8b5cf6"))
        
        # Aristas
        edges.append(Edge(source="Yo", target="AI", label="Especialidad"))
        edges.append(Edge(source="AI", target="ML", label="Core"))
        edges.append(Edge(source="AI", target="RAG", label="Implementación"))
        edges.append(Edge(source="AI", target="NLP", label="Uso"))
        edges.append(Edge(source="Yo", target="Py", label="Experto"))
        edges.append(Edge(source="Yo", target="SQL", label="Avanzado"))
        edges.append(Edge(source="Py", target="St", label="Framework"))
        edges.append(Edge(source="Py", target="AI", label="Base"))
        edges.append(Edge(source="Yo", target="Com", label="Soft Skill"))
        edges.append(Edge(source="Yo", target="Led", label="Soft Skill"))
        edges.append(Edge(source="Yo", target="Prob", label="Enfoque"))

        config = Config(
            width=800,
            height=500,
            directed=True, 
            physics=True, 
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#F7A7A6",
            collapsible=False
        )
        
        return_value = agraph(nodes=nodes, edges=edges, config=config)
