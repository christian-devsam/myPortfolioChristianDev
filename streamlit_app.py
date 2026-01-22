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

# --- IMPORTACIÓN DE GROQ ---
from langchain_groq import ChatGroq

# Configuración de página
st.set_page_config(page_title="Chat con Christian Silva", page_icon="⚡")

st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    h1 { color: #f97316 !important; } /* Naranja estilo Groq */
    .stChatMessage { background-color: #1e293b; border: 1px solid #334155; }
    .stTextInput input { color: #0f172a !important; }
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
        # Embeddings Locales (CPU)
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
    # --- INTEGRACIÓN CON GROQ ---
    # Usamos llama-3.3-70b-versatile que es rapidísimo y muy inteligente
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

# --- LÓGICA PRINCIPAL ---

if "conversation" not in st.session_state:
    with st.spinner("Cargando motor de ultra-velocidad..."):
        try:
            vectorstore = load_and_process_pdf("cv_csilva.pdf")
            
            if vectorstore:
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.process_complete = True
                st.toast("¡Groq Activo! 🚀", icon="⚡")
        except Exception as e:
            st.error(f"Ocurrió un error al iniciar: {e}")

# --- INTERFAZ DE CHAT ---

if "process_complete" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ej: ¿Qué experiencia tiene Christian?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Groq pensando..."):
                try:
                    response = st.session_state.conversation({'question': prompt})
                    ai_response = response['answer']
                    st.write(ai_response)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                except Exception as e:
                    st.error(f"Error generando respuesta: {e}")
