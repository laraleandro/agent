import os
import time
import base64
import re
import tempfile
import numpy as np
import streamlit as st
from openai import AzureOpenAI
import requests
import soundfile as sf
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit, ImageReader

# --- CONFIG INICIAL ---
AZURE_API_KEY = st.secrets["azure"]["api_key"]
AZURE_ENDPOINT = st.secrets["azure"]["endpoint"]
AZURE_API_VERSION = "2024-05-01-preview"
MODEL_DEPLOYMENT = "gpt-4o-mini"
OPENAI_KEY = st.secrets["openai"]["api_key"]

def load_instructions(file_path="assistant_role.txt"):
    with open(file_path, "r") as f:
        return f.read().strip()

ASSISTANT_ROLE = load_instructions()

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
)

# --- SESSION STATE DEFAULTS ---
if "language" not in st.session_state:
    st.session_state.language = "Português"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "audio_info" not in st.session_state:
    st.session_state.audio_info = {}
if "audio_recording" not in st.session_state:
    st.session_state.audio_recording = False
if "webrtc_ctx" not in st.session_state:
    st.session_state.webrtc_ctx = None
if "uploaded_file_ids" not in st.session_state:
    st.session_state.uploaded_file_ids = []

# --- STATIC ASSETS ---
LOGO_PATH = os.path.join(os.path.dirname(__file__), "fidelidade_logo.png")
ICON_PATH = os.path.join(os.path.dirname(__file__), "fidelidade_icon.png")
ICON_TAB_PATH = os.path.join(os.path.dirname(__file__), "fidelidade_icon_tab.png")

with open(LOGO_PATH, "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode()
with open(ICON_PATH, "rb") as f:
    icon_base64 = base64.b64encode(f.read()).decode()
with open(ICON_TAB_PATH, "rb") as f:
    icon_tab_base64 = base64.b64encode(f.read()).decode()

# --- TRANSLATIONS ---
LANG_STRINGS = {
    "Português": {
        "new_chat": "Novo Chat",
        "chat_input": "Escreva a sua pergunta…",
        "processing": "A processar…",
        "error_run_failed": "Erro: a execução falhou ou foi cancelada. Status:",
        "error_timeout": "Erro: tempo limite de execução excedido.",
        "reset_button": " Novo Chat",
        "agent_name": "Assistente Virtual",
        "login": "Iniciar Sessão",
        "login_user": "Email",
        "login_pass": "Palavra-passe",
        "login_submit": "Enviar",
        "login_success": "Sessão iniciada com sucesso",
        "sources": "Fontes",
        "start_speaking": "Começar a falar",
        "stop_speaking": "Parar de falar",
        "upload_label": "Carregar ficheiros para consulta",
        "export_pdf": "Exportar como PDF",
        "download_pdf": "Baixar PDF",
    },
    "English": {
        "new_chat": "New Chat",
        "chat_input": "Write your question…",
        "processing": "Processing…",
        "error_run_failed": "Error: run failed or was cancelled. Status:",
        "error_timeout": "Error: execution timeout exceeded.",
        "reset_button": " New Chat",
        "agent_name": "Agent Assistant",
        "login": "Login",
        "login_user": "Email",
        "login_pass": "Password",
        "login_submit": "Submit",
        "login_success": "Login submitted successfully",
        "sources": "Sources",
        "start_speaking": "Start speaking",
        "stop_speaking": "Stop speaking",
        "upload_label": "Upload files for assistant context",
        "export_pdf": "Export as PDF",
        "download_pdf": "Download PDF",
    },
}

# --- HELPER FUNCTIONS ---
def send_and_get_response_azure(user_input, file_ids=None, temperature=0.2, top_p=0.8):
    messages = [
        {"role": "system", "content": ASSISTANT_ROLE},
        {"role": "user", "content": user_input}
    ]
    response = client.chat.completions.create(
        model=MODEL_DEPLOYMENT,
        messages=messages,
        temperature=temperature,
        top_p=top_p
    )
    reply = response.choices[0].message.content
    return reply, []

def whisper_api_transcribe(audio_path, language="pt"):
    with open(audio_path, "rb") as f:
        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}"},
            files={"file": ("audio.wav", f, "audio/wav")},
            data={"model": "whisper-1", "language": language},
        )
    if response.ok and "text" in response.json():
        return response.json()["text"]
    return ""

def clean_markdown(text: str) -> str:
    text = re.sub(r'\\(.?)\\*', r'\1', text)    # bold
    text = re.sub(r'\(.?)\*', r'\1', text)        # italics
    text = re.sub(r'(.*?)', r'\1', text)          # underline/italics
    text = re.sub(r'#+\s+', '', text)               # headers
    text = re.sub(r'(.*?)', r'\1', text)          # code
    text = re.sub(r'\[\d+\]', '', text)             # [1] [2] citations
    return text.strip()

def export_chat_as_pdf() -> io.BytesIO:
    pdf_buffer = io.BytesIO()
    pdf_canvas = canvas.Canvas(pdf_buffer, pagesize=A4)
    width, height = A4
    margin = 50
    line_height = 18
    pair_spacing = 8
    logo_bytes = base64.b64decode(logo_base64)
    logo_reader = ImageReader(io.BytesIO(logo_bytes))
    logo_w, logo_h = 250, 28
    pdf_canvas.drawImage(
        logo_reader,
        margin,
        height - margin - logo_h,
        width=logo_w,
        height=logo_h,
        mask="auto"
    )
    y_position = height - margin - logo_h - line_height
    for role, msg in st.session_state.chat_history:
        text = clean_markdown(msg)
        font_name = "Helvetica-Bold" if role == "user" else "Helvetica"
        pdf_canvas.setFont(font_name, 12)
        wrapped = simpleSplit(text, font_name, 12, width - 2 * margin)
        for line in wrapped:
            if y_position < margin + line_height:
                pdf_canvas.showPage()
                pdf_canvas.setFont(font_name, 12)
                pdf_canvas.drawImage(
                    logo_reader,
                    margin,
                    height - margin - logo_h,
                    width=logo_w,
                    height=logo_h,
                    mask="auto"
                )
                y_position = height - margin - logo_h - line_height
            pdf_canvas.drawString(margin, y_position, line)
            y_position -= line_height
        if role == "assistant":
            y_position -= pair_spacing
    pdf_canvas.save()
    pdf_buffer.seek(0)
    return pdf_buffer

def upload_files_to_assistant(files):
    file_ids = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp:
            tmp.write(file.read())
            tmp.flush()
            uploaded_file = client.files.create(
                file=open(tmp.name, "rb"),
                purpose="assistants"
            )
            file_ids.append(uploaded_file.id)
    return file_ids

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []
    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        data = frame.to_ndarray()
        self.frames.append(data)
        return frame

# --- THEME / COLORS ---
primary_red = "#C80A1E"
light_grey = "#F5F5F5"
bg_color = "white"
text_color = "black"

st.set_page_config(
    page_title=LANG_STRINGS[st.session_state.language]["agent_name"],
    page_icon=ICON_TAB_PATH,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- GLOBAL CSS ---
st.markdown(
    f"""
    <style>
    .main {{ background-color: {bg_color}; color: {text_color}; }}
    div.block-container, section.main.block-container {{ padding-top: 0 !important; margin-top: 0 !important; }}
    header {{ margin-top: 0 !important; padding-top: 0 !important; }}
    .stChatMessage:nth-child(odd) div[class^='stChatMessage'] {{
        background: {primary_red}; color: white; border-radius: 8px;
    }}
    .stChatMessage:nth-child(even) div[class^='stChatMessage'] {{
        background: {light_grey}; color: {text_color}; border-radius: 8px;
    }}
    .stChatMessage {{ display: flex !important; align-items: flex-start !important; gap: 0.5rem; }}
    .stChatMessage > div.stAvatar {{ flex-shrink: 0; margin-top: 0.2rem; }}
    .stChatMessage > div[class^='stChatMessage'] {{ padding: 0.75rem 1rem !important; border-radius: 8px !important; flex-grow: 1; }}
    .stChatMessage > div {{ padding: 0 !important; }}
    h2 {{ margin: 0 !important; line-height: 1 !important; }}
    div.block-container, section.main.block-container {{ padding-top:4px !important; margin-top:0 !important; }}
    section[data-testid="stSidebar"] .block-container {{ padding-top: 0.5rem !important; }}
    .header-bar {{
        display:flex; align-items:center; justify-content:space-between;
        gap:1rem;
        padding:0.6rem 0;          
        margin: 30px 0 6px 0 !important;
    }}
    .header-left  {{ display:flex; align-items:center; gap:1.2rem; }}
    .header-left h2 {{ margin:0; line-height:1; }}
    .logo {{ width:300px; }}
    @media (max-width:600px) {{
        .header-bar {{ flex-direction:column; text-align:center; }}
        .logo {{ width:200px; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- HEADER ---
header_col1, header_col2 = st.columns([6, 1.3])
with header_col1:
    st.markdown(
        f"""
        <div class="header-bar">
            <div class="header-left">
                <img src="data:image/png;base64,{logo_base64}" class="logo" />
                <h2>{LANG_STRINGS[st.session_state.language]["agent_name"]}</h2>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_col2:
    st.markdown("<div style='margin-top: 3.7rem;'></div>", unsafe_allow_html=True)
    with st.expander(LANG_STRINGS[st.session_state.language]["login"], expanded=False):
        st.text_input(LANG_STRINGS[st.session_state.language]["login_user"], key="login_user")
        st.text_input(LANG_STRINGS[st.session_state.language]["login_pass"], type="password", key="login_pass")
        if st.button(LANG_STRINGS[st.session_state.language]["login_submit"], key="login_submit"):
            st.success(LANG_STRINGS[st.session_state.language]["login_success"])

# --- SIDEBAR ---
with st.sidebar:
    if st.button(LANG_STRINGS[st.session_state.language]["new_chat"]):
        for key in ("chat_history", "last_sources", "audio_info", "audio_recording", "webrtc_ctx", "uploaded_file_ids"):
            st.session_state.pop(key, None)
        st.session_state.chat_history = []
        st.session_state.last_sources = []
        st.session_state.audio_info = {}
        st.session_state.audio_recording = False
        st.session_state.webrtc_ctx = None
        st.session_state.uploaded_file_ids = []
        st.rerun()
    previous_language = st.session_state.language
    selected_language = st.selectbox(
        "Language / Idioma",
        options=["Português", "English"],
        index=0 if st.session_state.language == "Português" else 1,
        key="language_selector",
    )
    if selected_language != previous_language:
        st.session_state.language = selected_language
        st.rerun()

    # --- FILE UPLOAD BUTTON ---
    uploaded_files = st.file_uploader(
        LANG_STRINGS[st.session_state.language]["upload_label"],
        type=["pdf", "txt", "docx", "csv"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    if uploaded_files:
        with st.spinner("A carregar ficheiros..."):
            file_ids = upload_files_to_assistant(uploaded_files)
            st.session_state.uploaded_file_ids = file_ids

    # --- EXPORTAÇÃO DE HISTÓRICO NA BARRA LATERAL ---
    if st.button(LANG_STRINGS[st.session_state.language]["export_pdf"]):  
        chat_pdf = export_chat_as_pdf()  
        st.download_button(  
            label=LANG_STRINGS[st.session_state.language]["download_pdf"],  
            data=chat_pdf,  
            file_name="chat_history.pdf",  
            mime="application/pdf",  
        )

    # --- RENDER SOURCES ---
    if st.session_state.last_sources:
        st.markdown(f"### {LANG_STRINGS[st.session_state.language]['sources']}")
        for source in st.session_state.last_sources:
            st.markdown(f"[{source['n']}] {source['file']}")

# --- CHAT INPUT & AUDIO INPUT ---
user_input = st.chat_input(LANG_STRINGS[st.session_state.language]["chat_input"])

# Botão único: alterna entre gravar e parar
if st.session_state.audio_recording:
    btn_label = LANG_STRINGS[st.session_state.language]["stop_speaking"]
else:
    btn_label = LANG_STRINGS[st.session_state.language]["start_speaking"]

audio_toggled = st.button(btn_label, key="audio_toggle")
if audio_toggled:
    st.session_state.audio_recording = not st.session_state.audio_recording

if st.session_state.audio_recording:
    st.info("Clique em Start no widget abaixo para ativar o microfone e em Stop para terminar.")
    st.session_state.webrtc_ctx = webrtc_streamer(
        key="audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
        audio_processor_factory=AudioProcessor,
    )
else:
    ctx = st.session_state.webrtc_ctx
    if ctx and hasattr(ctx, "state") and not ctx.state.playing:
        if ctx.audio_processor and ctx.audio_processor.frames:
            audio_data = np.concatenate(ctx.audio_processor.frames, axis=1)
            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(temp_audio.name, audio_data.T, 48000, format="WAV")
            st.audio(temp_audio.name)
            st.info("A transcrever áudio via Whisper API...")
            transcript = whisper_api_transcribe(temp_audio.name, language="pt")
            if transcript:
                st.session_state.audio_info = {"transcript": transcript}
                st.success(f"Pergunta reconhecida: \"{transcript}\"")
            else:
                st.error("Falha ao transcrever áudio.")
        st.session_state.webrtc_ctx = None

if st.session_state.audio_info.get("transcript"):
    user_input = st.session_state.audio_info["transcript"]
    st.session_state.audio_info = {}

# --- CHAT ENGINE ---
if user_input:
    with st.spinner(LANG_STRINGS[st.session_state.language]["processing"]):
        reply, sources = send_and_get_response_azure(
            user_input,
            file_ids=st.session_state.uploaded_file_ids,
            temperature=0.2,
            top_p=0.8
        )
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", reply))
    st.session_state.last_sources = sources

# --- RENDER CHAT HISTORY ---
for role, msg in st.session_state.chat_history:
    if role == "user":
        with st.chat_message("user"):
            st.markdown(msg)
    else:
        avatar = f"data:image/png;base64,{icon_base64}"
        with st.chat_message("assistant", avatar=avatar):
            st.markdown(msg)