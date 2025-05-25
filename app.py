import os
import time
import base64
from openai import AzureOpenAI
import streamlit as st

# Constants
AZURE_API_KEY = st.secrets["azure"]["api_key"]
AZURE_ENDPOINT = st.secrets["azure"]["endpoint"]
AZURE_API_VERSION = "2024-05-01-preview"
MODEL_DEPLOYMENT = "gpt-4o-mini"  # Adjust if your deployment name is different

# Load assistant role instructions
def load_instructions(file_path="assistant_role.txt"):
    with open(file_path, "r") as f:
        return f.read().strip()

ASSISTANT_FILE = "AssistantID.TXT"
ASSISTANT_NAME = "GroupF_Assistant"
ASSISTANT_ROLE = load_instructions()

# Initialize OpenAI client
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
)

# Load or create assistant
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

# Create new thread
def create_thread():
    return client.beta.threads.create()

# Send message and get response
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
            return f"{LANG_STRINGS[st.session_state.language]['error_run_failed']} {run_status.status}"
        if time.time() - start_time > max_wait:
            return LANG_STRINGS[st.session_state.language]["error_timeout"]
        time.sleep(1)

    messages = client.beta.threads.messages.list(thread_id=thread_id)
    last_message = messages.data[0].content[0].text.value
    return last_message

# Paths for logos/icons
LOGO_PATH = os.path.join(os.path.dirname(__file__), "fidelidade_logo.png")
ICON_PATH = os.path.join(os.path.dirname(__file__), "fidelidade_icon.png")

with open(LOGO_PATH, "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode()

with open(ICON_PATH, "rb") as f:
    icon_base64 = base64.b64encode(f.read()).decode()

# Localization
LANG_STRINGS = {
    "English": {
        "new_chat": "New Chat",
        "chat_input": "Write your question…",
        "processing": "Processing…",
        "error_run_failed": "Error: run failed or was cancelled. Status:",
        "error_timeout": "Error: execution timeout exceeded.",
        "reset_button": "🔄 New Chat",
        "agent_name": "Agent Assistant",
    },
    "Português": {
        "new_chat": "Novo Chat",
        "chat_input": "Escreva a sua pergunta…",
        "processing": "A processar…",
        "error_run_failed": "Erro: a execução falhou ou foi cancelada. Status:",
        "error_timeout": "Erro: tempo limite de execução excedido.",
        "reset_button": "🔄 Novo Chat",
        "agent_name": "Assistente Virtual",
    }
}

# Session state initialization
if "language" not in st.session_state:
    st.session_state.language = "Português"

if "assistant" not in st.session_state:
    st.session_state.assistant = load_or_create_assistant()

if "thread" not in st.session_state:
    st.session_state.thread = create_thread()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Colors (fixed theme)
primary_red = "#C80A1E"
light_grey = "#F5F5F5"
bg_color = "white"
text_color = "black"

# Page configuration
st.set_page_config(
    page_title=LANG_STRINGS[st.session_state.language]["agent_name"],
    page_icon=ICON_PATH,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar UI
with st.sidebar:
    if st.button(LANG_STRINGS[st.session_state.language]["new_chat"]):
        for key in ("assistant", "thread", "chat_history"):
            st.session_state.pop(key, None)
        st.session_state.assistant = load_or_create_assistant()
        st.session_state.thread = create_thread()
        st.session_state.chat_history = []

    selected_language = st.selectbox(
        "Language / Idioma",
        options=["Português", "English"],
        index=0 if st.session_state.language == "Português" else 1,
        key="language_selector"
    )
    st.session_state.language = selected_language

# Global CSS styling
st.markdown(
    f"""
    <style>
    .main {{
        background-color: {bg_color};
        color: {text_color};
    }}
    div.block-container, section.main.block-container {{
        padding-top: 0 !important;
        margin-top: 0 !important;
    }}
    header {{
        margin-top: 0 !important;
        padding-top: 0 !important;
    }}
    .stChatMessage:nth-child(odd) div[class^='stChatMessage'] {{
        background: {primary_red};
        color: white;
        border-radius: 8px;
    }}
    .stChatMessage:nth-child(even) div[class^='stChatMessage'] {{
        background: {light_grey};
        color: {text_color};
        border-radius: 8px;
    }}
    .stChatMessage {{
        display: flex !important;
        align-items: flex-start !important;
        gap: 0.5rem;
    }}
    .stChatMessage > div.stAvatar {{
        flex-shrink: 0;
        margin-top: 0.2rem;
    }}
    .stChatMessage > div[class^='stChatMessage'] {{
        padding: 0.75rem 1rem !important;
        border-radius: 8px !important;
        flex-grow: 1;
    }}
    .stChatMessage > div {{
        padding: 0 !important;
    }}
    h2 {{
        margin: 0 !important;
        line-height: 1 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header with logo and assistant name
st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 1.2rem; padding: 1rem 0; margin-top: 12px;">
        <img src="data:image/png;base64,{logo_base64}" width="300" style="margin: 0;" />
        <h2 style="margin: 0; line-height: 1;">{LANG_STRINGS[st.session_state.language]["agent_name"]}</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

# Chat input and processing
user_input = st.chat_input(LANG_STRINGS[st.session_state.language]["chat_input"])
if user_input:
    with st.spinner(LANG_STRINGS[st.session_state.language]["processing"]):
        reply = send_and_get_response(
            st.session_state.assistant.id,
            st.session_state.thread.id,
            user_input,
        )
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", reply))

# Render chat history
for role, msg in st.session_state.chat_history:
    if role == "user":
        with st.chat_message("user"):
            st.markdown(msg)
    else:
        avatar = f"data:image/png;base64,{icon_base64}"
        with st.chat_message("assistant", avatar=avatar):
            st.markdown(msg)
