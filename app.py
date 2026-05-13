import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
import os

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Asistente Legal Interno", page_icon="⚖️", layout="wide")

# --- 2. REGLAS ESTRICTAS DEL AGENTE (SYSTEM PROMPT) ---
PROMPT_TEMPLATE = """
Eres el Asistente Legal y de Procesos Internos de la empresa.
Tu único objetivo es responder las dudas de los empleados basándote ESTRICTAMENTE en el contexto proporcionado (los documentos subidos).
Si la respuesta no está en los documentos, NO la inventes. Responde exactamente esto: "Lo siento, esa información no se encuentra en los manuales actuales. Por favor, contacta directamente con el departamento legal."
Responde con un tono corporativo, amable, claro y estructurado (usa viñetas si es necesario).

Contexto:
{context}

Pregunta del empleado:
{question}

Respuesta:
"""

# --- 3. FUNCIONES DE PROCESAMIENTO ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1) 
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    # Sintaxis moderna de LangChain para procesar documentos
    chain = create_stuff_documents_chain(model, prompt)
    return chain

# --- 4. INTERFAZ VISUAL ---
st.title("⚖️ Asistente de Procesos y Legal")
st.markdown("Pregúntame sobre políticas de la empresa, procesos internos o plantillas legales.")

api_key = st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    st.error("Error: No se ha configurado la API Key de Google en los secretos de Streamlit (Settings -> Secrets).")
    st.stop()

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("📂 Panel de Administración")
    st.write("Sube aquí los PDFs con manuales y políticas.")
    pdf_docs = st.file_uploader("Carga tus PDFs y presiona 'Procesar'", accept_multiple_files=True, type=["pdf"])
    
    if st.button("Procesar Documentos"):
        if pdf_docs:
            with st.spinner("Leyendo documentos y entrenando al agente..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks, api_key)
                st.session_state["vector_store"] = vector_store
                st.success("¡Agente entrenado! Ya pueden hacer preguntas.")
        else:
            st.warning("Por favor, sube al menos un PDF primero.")

# --- ZONA DE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Escribe tu duda aquí (ej. ¿Cuántos días de vacaciones me corresponden?)")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        if "vector_store" in st.session_state:
            with st.spinner("Buscando en los documentos legales..."):
                docs = st.session_state["vector_store"].similarity_search(user_question)
                chain = get_conversational_chain()
                # Ejecutamos la cadena con la sintaxis moderna
                reply = chain.invoke({"context": docs, "question": user_question})
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            st.warning("El administrador aún no ha cargado los documentos legales. Por favor, procesa los PDFs primero en el menú lateral.")
