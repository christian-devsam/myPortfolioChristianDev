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

    /* Focus (cuando haces
