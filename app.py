import streamlit as st  
import os  
import time  
from openai import AzureOpenAI  
import pickle  
  
# Constants  

AZURE_API_KEY = st.secrets["azure"]["api_key"]
AZURE_ENDPOINT = st.secrets["azure"]["endpoint"]
AZURE_API_VERSION = "2024-05-01-preview"
MODEL_DEPLOYMENT = "gpt-4o-mini"  # Your deployment name


def load_instructions(file_path="assistant_role.txt"):
    with open(file_path, "r") as f:
        return f.read().strip()


# Assistant settings
ASSISTANT_FILE = "AssistantID.TXT"
ASSISTANT_NAME = "GroupF_Assistant"
ASSISTANT_ROLE = load_instructions() 



 # Initialize client
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
    max_wait = 30  # segundos máximo de espera
    start_time = time.time()

    while True:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
        if run_status.status == "completed":
            break
        elif run_status.status in ["failed", "cancelled", "expired"]:
            print("Run failed with details:", run_status)
            return f"Erro: a execução falhou ou foi cancelada. Status: {run_status.status}"
        if time.time() - start_time > max_wait:
            return "Erro: tempo limite de execução excedido."
        time.sleep(1)

    messages = client.beta.threads.messages.list(thread_id=thread_id)
    last_message = messages.data[0].content[0].text.value
    return last_message

# --- Streamlit UI ---
from pathlib import Path
import base64


# ----------  CONFIG & BRAND ASSETS  ----------
PRIMARY_RED = "#C80A1E"            # Fidelidade red
LIGHT_GREY  = "#F5F5F5"
LOGO_PATH = os.path.join(os.path.dirname(__file__), "fidelidade_logo.png")
ICON_PATH = os.path.join(os.path.dirname(__file__), "fidelidade_icon.png")
import os

st.set_page_config(
    page_title="Assistente Virtual",
    page_icon=ICON_PATH
)

# ----------  GLOBAL STYLE OVERRIDES ----------
st.markdown(
    f"""
    <style>
    /* Remove padding and margin from main container */
    div.block-container, section.main.block-container {{
        padding-top: 0 !important;
        margin-top: 0 !important;
    }}

    /* Remove margin and padding from header */
    header {{
        margin-top: 0 !important;
        padding-top: 0 !important;
    }}

    /* Style chat bubbles */
    .stChatMessage:nth-child(odd) div[class^='stChatMessage'] {{
        background: {PRIMARY_RED};
        color: white;
        border-radius: 8px;
    }}
    .stChatMessage:nth-child(even) div[class^='stChatMessage'] {{
        background: {LIGHT_GREY};
        color: black;
        border-radius: 8px;
    }}

    /* Align avatar and text vertically */
    .stChatMessage {{
        display: flex !important;
        align-items: flex-start !important; /* Align avatar and text at top */
        gap: 0.5rem;
    }}

    .stChatMessage > div.stAvatar {{
        flex-shrink: 0;
        margin-top: 0.2rem; /* small spacing from top */
    }}

    .stChatMessage > div[class^='stChatMessage'] {{
        padding: 0.75rem 1rem !important;
        border-radius: 8px !important;
        /* ensure text block takes remaining space */
        flex-grow: 1;
    }}

    /* Remove padding from message container div to avoid double padding */
    .stChatMessage > div {{
        padding: 0 !important;
    }}

    /* Remove margins from h2 headings */
    h2 {{
        margin: 0 !important;
        line-height: 1 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)



# ----------  PAGE HEADER ----------
import base64
import os

# Paths
  # smaller/different image

# Encode big logo for header
with open(LOGO_PATH, "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode()

# Encode smaller icon for chat avatar
with open(ICON_PATH, "rb") as f:
    icon_base64 = base64.b64encode(f.read()).decode()




# Render image + text properly aligned
st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 1.2rem; padding: 1rem 0; margin-top: 12px;">
        <img src="data:image/png;base64,{logo_base64}" width="300" style="margin: 0;" />
        <h2 style="margin: 0; line-height: 1;">Agent Assistant</h2>
    </div>
    """,
    unsafe_allow_html=True,
)






# ----------  RESET BUTTON ----------
if st.button("🔄 Reiniciar Chat"):
    for key in ("assistant", "thread", "chat_history"):
        st.session_state.pop(key, None)
    st.rerun()

# ----------  INITIALISE STATE ----------
if "assistant" not in st.session_state:
    st.session_state.assistant = load_or_create_assistant()
if "thread" not in st.session_state:
    st.session_state.thread = create_thread()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------  CHAT INPUT ----------
user_input = st.chat_input("Escreva a sua pergunta…")
if user_input:
    with st.spinner("A processar…"):
        reply = send_and_get_response(
            st.session_state.assistant.id,
            st.session_state.thread.id,
            user_input,
        )
    # store history
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", reply))

# ----------  RENDER CHAT HISTORY ----------
for role, msg in st.session_state.chat_history:
    if role == "user":
        # User messages with default user icon (no avatar specified)
        with st.chat_message("user"):
            st.markdown(msg)
    else:
        # Assistant messages with your photo avatar
        avatar = f"data:image/png;base64,{icon_base64}"
        with st.chat_message("assistant", avatar=avatar):
            st.markdown(msg)







