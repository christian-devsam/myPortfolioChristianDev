import streamlit as st
import time
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Configuración de página
st.set_page_config(page_title="Chat con Christian Silva", page_icon="✨")

# Estilos CSS
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    h1 { color: #38bdf8 !important; }
    .stChatMessage { background-color: #1e293b; border: 1px solid #334155; }
    .stTextInput input { color: #0f172a !important; }
</style>
""", unsafe_allow_html=True)

st.title("✨ Asistente IA de Christian Silva")
st.write("Potenciado por **Google Gemini**. Pregúntame sobre mi experiencia y proyectos.")

# --- GESTIÓN DE LA API KEY ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except FileNotFoundError:
    st.error("⚠️ No se encontró la API Key. Configura los 'Secrets' en Streamlit Cloud.")
    st.stop()

# --- FUNCIONES ---

@st.cache_resource
def load_and_process_pdf(pdf_path):
    # 1. Leer el PDF
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
    
    if not text:
        st.error("⚠️ El PDF parece estar vacío o es una imagen escaneada. Intenta subir un PDF con texto seleccionable.")
        return None
    
    # 2. Partir texto
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    
    total_chunks = len(chunks)
    if total_chunks == 0:
        return None

    # 3. Crear Vector Store POR LOTES (Para evitar Error 504)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    vectorstore = None
    
    # Barra de progreso
    progress_bar = st.progress(0, text="Iniciando análisis del perfil...")
    batch_size = 5 # Procesamos de 5 en 5 fragmentos
    
    try:
        for i in range(0, total_chunks, batch_size):
            # Tomamos un lote pequeño
            batch = chunks[i : i + batch_size]
            
            if vectorstore is None:
                # El primer lote crea la base de datos
                vectorstore = FAISS.from_texts(texts=batch, embedding=embeddings)
            else:
                # Los siguientes lotes se agregan
                vectorstore.add_texts(batch)
                time.sleep(0.5) # Pausa pequeña para no saturar la API
            
            # Actualizar barra
            progress = min((i + batch_size) / total_chunks, 1.0)
            progress_bar.progress(progress, text=f"Analizando fragmento {min(i + batch_size, total_chunks)} de {total_chunks}...")
            
        progress_bar.empty() # Limpiar barra al terminar
        return vectorstore
        
    except Exception as e:
        st.error(f"Error procesando embeddings: {e}")
        return None

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.3)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return chain

# --- PROCESAMIENTO INICIAL ---

if "conversation" not in st.session_state:
    vectorstore = load_and_process_pdf("cv_csilva.pdf")
    
    if vectorstore:
        st.session_state.conversation = get_conversation_chain(vectorstore)
        st.session_state.process_complete = True
        st.toast("¡Perfil cargado exitosamente!", icon="✅")

# --- INTERFAZ DE CHAT ---

if "process_complete" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ej: ¿Qué tecnologías domina Christian?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    response = st.session_state.conversation({'question': prompt})
                    ai_response = response['answer']
                    st.write(ai_response)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                except Exception as e:
                    st.error(f"Error al responder: {e}")
