import streamlit as st
import asyncio
import os

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

# --- IMPORTACIÓN PARA EL GRAFO ---
from streamlit_agraph import agraph, Node, Edge, Config

# Configuración de página
st.set_page_config(page_title="Chat con Christian Silva", page_icon="⚡", layout="wide")

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
<style>
    /* Fondo oscuro global */
    .stApp { 
        background-color: #0f172a; 
    }
    
    /* Títulos en naranja Groq */
    h1, h2, h3 { 
        color: #f97316 !important; 
    }
    
    /* TEXTO BLANCO Y LEGIBLE */
    .stMarkdown p, .stMarkdown li, .stText, p {
        color: #ffffff !important;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    /* Pestañas (Tabs) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
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

st.title("⚡ Asistente IA de Christian Silva")
st.write("Potenciado por **Groq (Llama 3.3)** + **Embeddings Locales**.")

# --- GESTIÓN DE LA API KEY ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
except FileNotFoundError:
    st.error("⚠️ No se encontró la GROQ_API_KEY. Configura los 'Secrets' en Streamlit Cloud.")
    st.stop()

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
    
    # CREACIÓN DE 3 PESTAÑAS
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
                        response = st.session_state.conversation({'question': prompt})
                        ai_response = response['answer']
                        st.write(ai_response)
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
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
                        prompt_carta = f"Actúa como el candidato. Analiza esta oferta: {job_description}. Basado en mi CV (contexto), escribe una carta de presentación persuasiva."
                        response = st.session_state.conversation({'question': prompt_carta})
                        st.subheader("Tu Carta Generada:")
                        st.markdown(response['answer'])
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("⚠️ Pega la descripción de la oferta primero.")

    # --- PESTAÑA 3: GRAFO DE CONOCIMIENTO ---
    with tab3:
        st.header("🕸️ Mapa de Habilidades Interactivo")
        st.markdown("Explora mis conexiones técnicas. ¡Puedes arrastrar los nodos!")
        
        # Definición de Nodos (Skillset)
        # Puedes editar esto para que coincida exactamente con tus habilidades
        nodes = []
        edges = []
        
        # Nodo Central
        nodes.append(Node(id="Yo", label="Christian Silva", size=40, color="#f97316")) # Naranja Groq
        
        # Categoría: Data Science & AI (Azul)
        nodes.append(Node(id="AI", label="Artificial Intelligence", color="#3b82f6"))
        nodes.append(Node(id="ML", label="Machine Learning", color="#3b82f6"))
        nodes.append(Node(id="RAG", label="RAG Systems", color="#3b82f6"))
        nodes.append(Node(id="NLP", label="NLP", color="#3b82f6"))
        
        edges.append(Edge(source="Yo", target="AI", label="Especialidad"))
        edges.append(Edge(source="AI", target="ML", label="Core"))
        edges.append(Edge(source="AI", target="RAG", label="Implementación"))
        edges.append(Edge(source="AI", target="NLP", label="Uso"))

        # Categoría: Lenguajes & Tools (Verde)
        nodes.append(Node(id="Py", label="Python", color="#10b981"))
        nodes.append(Node(id="SQL", label="SQL", color="#10b981"))
        nodes.append(Node(id="St", label="Streamlit", color="#10b981"))
        nodes.append(Node(id="Git", label="Git/GitHub", color="#10b981"))
        
        edges.append(Edge(source="Yo", target="Py", label="Experto"))
        edges.append(Edge(source="Yo", target="SQL", label="Avanzado"))
        edges.append(Edge(source="Py", target="St", label="Framework"))
        edges.append(Edge(source="Py", target="AI", label="Base"))

        # Categoría: Soft Skills (Violeta)
        nodes.append(Node(id="Com", label="Comunicación", color="#8b5cf6"))
        nodes.append(Node(id="Led", label="Liderazgo", color="#8b5cf6"))
        nodes.append(Node(id="Prob", label="Resolución Problemas", color="#8b5cf6"))
        
        edges.append(Edge(source="Yo", target="Com", label="Soft Skill"))
        edges.append(Edge(source="Yo", target="Led", label="Soft Skill"))
        edges.append(Edge(source="Yo", target="Prob", label="Enfoque"))

        # Configuración del Grafo
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
        
        # Renderizar el grafo
        return_value = agraph(nodes=nodes, edges=edges, config=config)
