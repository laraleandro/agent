import os
import time
import base64
import re
from openai import AzureOpenAI
import streamlit as st

# --- CONFIG INICIAL ---
AZURE_API_KEY = st.secrets["azure"]["api_key"]
AZURE_ENDPOINT = st.secrets["azure"]["endpoint"]
AZURE_API_VERSION = "2024-05-01-preview"
MODEL_DEPLOYMENT = "gpt-4o-mini"

def load_instructions(file_path="assistant_role.txt"):
    with open(file_path, "r") as f:
        return f.read().strip()

ASSISTANT_FILE = "AssistantID.TXT"
ASSISTANT_NAME = "GroupF_Assistant"
ASSISTANT_ROLE = load_instructions()

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
)

def load_or_create_assistant():
    if os.path.exists(ASSISTANT_FILE):
        with open(ASSISTANT_FILE, "r") as f:
            assistant_id = f.read().strip()
        assistant = client.beta.assistants.retrieve(assistant_id)
    else:
        assistant = client.beta.assistants.create(
            name=ASSISTANT_NAME,
            instructions=ASSISTANT_ROLE,
            model=MODEL_DEPLOYMENT,
            tools=[{"type": "file_search"}]
        )
        with open(ASSISTANT_FILE, "w") as f:
            f.write(assistant.id)
    return assistant

def create_thread():
    return client.beta.threads.create()

def get_file_info(file_id):
    try:
        file_obj = client.files.retrieve(file_id)
        file_name = getattr(file_obj, "filename", None)
        if not file_name:
            file_name = getattr(file_obj, "name", None)
        if not file_name and isinstance(file_obj, dict):
            file_name = file_obj.get("filename") or file_obj.get("name")
        if not file_name:
            file_name = str(file_obj)
        return file_name
    except Exception:
        return file_id

def send_and_get_response(assistant_id, thread_id, message):
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message
    )
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )
    max_wait = 30
    start_time = time.time()

    while True:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
        if run_status.status == "completed":
            break
        elif run_status.status in ["failed", "cancelled", "expired"]:
            return "Erro: execução falhou.", []
        if time.time() - start_time > max_wait:
            return "Erro: tempo limite de execução excedido.", []
        time.sleep(1)

    messages = client.beta.threads.messages.list(thread_id=thread_id)
    last_message_obj = messages.data[0]
    content_block = last_message_obj.content[0]
    value = getattr(content_block.text, "value", "")
    annotations = getattr(content_block.text, "annotations", [])

    sources = []
    citation_map = {}
    citation_counter = 1
    for annotation in annotations:
        if getattr(annotation, "type", "") == "file_citation":
            file_id = getattr(annotation.file_citation, "file_id", None)
            marker = getattr(annotation, "text", "")
            if file_id and marker not in citation_map:
                citation_map[marker] = citation_counter
                file_name = get_file_info(file_id)
                sources.append({"n": citation_counter, "file": file_name})
                citation_counter += 1

    def replace_marker(match):
        marker = match.group(0)
        if marker in citation_map:
            return f"[{citation_map[marker]}]"
        return ""
    clean_message = re.sub(r"【\d+:\d+†source】", replace_marker, value)
    return clean_message, sources

# ───────────────────────────────  RESOURCES  ────────────────────────────────
import os
import base64
import streamlit as st

# --- UI, LOGOS, LOCALE ---
LOGO_PATH = os.path.join(os.path.dirname(__file__), "fidelidade_logo.png")
ICON_PATH = os.path.join(os.path.dirname(__file__), "fidelidade_icon.png")

with open(LOGO_PATH, "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode()
with open(ICON_PATH, "rb") as f:
    icon_base64 = base64.b64encode(f.read()).decode()

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
        "sources": "Fontes"
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
        "login_success": "Login submitted successful",
        "sources": "Sources"
    }
}

# --- SESSION STATE ---
if "language" not in st.session_state:
    st.session_state.language = "Português"
if "assistant" not in st.session_state:
    st.session_state.assistant = load_or_create_assistant()
if "thread" not in st.session_state:
    st.session_state.thread = create_thread()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# --- THEME / COLORS ---
primary_red = "#C80A1E"
light_grey = "#F5F5F5"
bg_color = "white"
text_color = "black"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title=LANG_STRINGS[st.session_state.language]["agent_name"],
    page_icon=ICON_PATH,
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
        for key in ("assistant", "thread", "chat_history", "last_sources"):
            st.session_state.pop(key, None)
        st.session_state.assistant = load_or_create_assistant()
        st.session_state.thread = create_thread()
        st.session_state.chat_history = []
        st.session_state.last_sources = []
        st.rerun()

    selected_language = st.selectbox(
        "Language / Idioma",
        options=["Português", "English"],
        index=0 if st.session_state.language == "Português" else 1,
        key="language_selector"
    )
    st.session_state.language = selected_language

# --- CHAT INPUT & UPDATE STATE ---
user_input = st.chat_input(LANG_STRINGS[st.session_state.language]["chat_input"])
if user_input:
    with st.spinner(LANG_STRINGS[st.session_state.language]["processing"]):
        reply, sources = send_and_get_response(
            st.session_state.assistant.id,
            st.session_state.thread.id,
            user_input,
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

# --- RENDER SOURCES ---
if st.session_state.last_sources:
    st.sidebar.markdown(f"### {LANG_STRINGS[st.session_state.language]['sources']}")
    for source in st.session_state.last_sources:
        st.sidebar.markdown(f'[{source["n"]}] {source["file"]}')
