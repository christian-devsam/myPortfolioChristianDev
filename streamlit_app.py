import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings  # <--- CAMBIO IMPORTANTE
from langchain_google_genai import ChatGoogleGenerativeAI

# Configuración de página
st.set_page_config(page_title="Chat con Christian Silva", page_icon="✨")

st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    h1 { color: #38bdf8 !important; }
    .stChatMessage { background-color: #1e293b; border: 1px solid #334155; }
    .stTextInput input { color: #0f172a !important; }
</style>
""", unsafe_allow_html=True)

st.title("✨ Asistente IA de Christian Silva")
st.write("Potenciado por **Google Gemini** + **Embeddings Locales**.")

# --- GESTIÓN DE LA API KEY (Solo para el Chat, no para Embeddings) ---
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
    
    # 2. Partir texto
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    if not chunks:
        return None

    # 3. Crear Vector Store (MODELO LOCAL - GRATIS Y SIN TIMEOUTS)
    # Usamos all-MiniLM-L6-v2 que es rápido y ligero
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error al crear embeddings locales: {e}")
        return None

def get_conversation_chain(vectorstore):
    # Usamos Gemini solo para generar la respuesta final (Chat)
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
    with st.spinner("Descargando modelo local y leyendo CV... (Esto puede tardar 30 seg la primera vez)"):
        try:
            vectorstore = load_and_process_pdf("cv_csilva.pdf")
            
            if vectorstore:
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.process_complete = True
                st.toast("¡IA Lista para responder!", icon="✅")
        except Exception as e:
            st.error(f"Ocurrió un error general: {e}")

# --- INTERFAZ DE CHAT ---

if "process_complete" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ej: ¿Qué experiencia tiene Christian en Data Engineering?"):
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
                    st.error(f"Error: {e}")
