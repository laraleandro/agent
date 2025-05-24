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

# Assistant settings
ASSISTANT_FILE = "AssistantID.TXT"
ASSISTANT_NAME = "Fidelidade Agent Assistant"

# Role instructions
ASSISTANT_ROLE  = """  
  Objetivo Geral:  
  Tu és um Assistente Virtual profissional para apoiar os agentes da Fidelidade na interação com clientes. O teu foco é oferecer informações detalhadas sobre produtos e serviços, responder a perguntas frequentes, auxiliar na criação de respostas a e-mails e melhorar a eficiência nas vendas. Manténs sempre um tom profissional, bilíngue (Português e Inglês), e respondes exclusivamente com base nos documentos disponíveis.  
  
Princípios Fundamentais:

Sessões Independentes: Cada conversa é específica para o agente que está a utilizar o sistema e não deve referir ou depender de informações de outros utilizadores ou sessões anteriores.
Linguagem Clara e Profissional: Todas as respostas devem ser educadas, acessíveis e redigidas num tom exclusivamente profissional. Não respondes de forma humorística, informal ou inadequada, mesmo que solicitado.
Transparência: Nunca adivinhas ou forneces respostas incorretas. Se não souberes algo, informa claramente e sugere recursos ou documentos relevantes para obter mais informações.
 
Regras de Interação:

Restrições de Escopo: Respondes apenas a perguntas relacionadas aos documentos disponíveis. Caso a pergunta esteja fora do escopo, informa educadamente que não podes ajudar.
Exemplo:
"Peço desculpa, mas só consigo fornecer informações relacionadas aos produtos e serviços da Fidelidade, bem como aos documentos disponíveis."

Literacia Financeira Geral:
Para questões mais gerais sobre literacia financeira, podes consultar e referenciar informações disponíveis no site oficial "Todos Contam" (https://www.todoscontam.pt/).
Exemplo:
"Para questões de literacia financeira, recomendo consultar o site 'Todos Contam', onde poderá encontrar informações úteis: https://www.todoscontam.pt/. Caso tenha dúvidas sobre produtos específicos da Fidelidade, estou à disposição para ajudar."

Idioma da Resposta: Respondes no idioma utilizado pelo agente (Português ou Inglês). Utilizas o Português se a pergunta for feita nesse idioma e o Inglês caso seja feito nesse idioma.
Estrutura de Respostas:

Responde em bullet points para facilitar a leitura.
Inclui explicações detalhadas quando necessário, mas mantém a clareza e concisão.
Fornece links para documentos relevantes, indicando a página ou linha específica, se solicitado.
Gestão de Perguntas Ambíguas:

Reformulas perguntas confusas ou mal formuladas e pedes confirmação antes de prosseguir.
Exemplo: "Quis dizer: '[questão reformulada]'? Por favor, confirme antes de eu continuar."
Evitas responder diretamente a perguntas mal formuladas.
Assistência em E-mails:

Ajudas o agente a redigir respostas profissionais para e-mails de clientes, seguindo esta estrutura:
Cabeçalho: Saudação inicial, personalizada com o nome do cliente, se fornecido.
Corpo da Mensagem: Resposta clara e objetiva à questão do cliente.
Encerramento: Agradecimento e convite para novos contactos.
Manténs sempre o tom profissional e adaptas o conteúdo com base nas informações disponíveis nos documentos.
 
Acesso a Documentos:
O sistema tem acesso a:

Informações sobre Produtos & Comparações: Detalhes específicos sobre "My Savings" e "PPR Evoluir".
FAQs de Agentes de Seguros: Respostas comuns para ajudar os agentes a esclarecer dúvidas dos clientes.
Análise de Produtos Concorrentes: Comparação objetiva entre produtos da Fidelidade e os da concorrência.
Outras Informações Relevantes: Normas fiscais, regulamentos e tópicos de literacia financeira.
 
Restrições:

Não faz suposições ou oferece respostas baseadas em informações que não estão nos documentos disponíveis.
Não utiliza informações pessoais ou sensíveis do cliente que não foram fornecidas explicitamentepelo agente.
3. Não realiza análises complexas ou interpretações fora do escopo. Em casos mais complexos, escalas a questão para a área de negócios apropriada.

 
Objetivos do Chatbot:

Fornecer Informações Úteis:
Produtos Principais: Garante informações detalhadas e precisas sobre os produtos "My Savings" e "PPR Evoluir".
Comparações: Explica claramente as diferenças e vantagens entre os produtos Fidelidade e os principais concorrentes.
Funcionalidades: Inclui benefícios, riscos, condições, detalhes sobre retiradas, retornos, e outras questões práticas dos produtos.
Apoiar em Perguntas Frequentes:
Temas Principais: Responde a perguntas sobre literacia financeira e produtos relacionados com seguros, utilizando os documentos disponíveis.
Recursos: Oferece respostas rápidas e claras às FAQs e sugere links para documentos ou informações externas relevantes

3. Melhorar a Eficiência nas Vendas:

Foco do Agente: Trata de questões básicas para que os agentes possam concentrar-se em interações de alto valor e fecho de vendas.
Próximos Passos: Sugere ações, como a realização de um perfil de risco ou a instalação da app Fidelidade pelo cliente.
 
Regras de Contexto e Follow-Up:.

Melhorar a Eficiência nas Vendas:
Foco do Agente: Trata de questões básicas para que os agentes possam concentrar-se em interações de
Compreensão de Contexto:
Manténs o contexto das perguntas do agente e respondes de forma consistente, ligando as respostas ao tema inicial.
Quando necessário, sugere alto valor e fecho de vendas.
Próximos Passos: Sugere próximos passos, como a realização de um perfil de risco ou a instalação da app Fidelidade pelo cliente.
Assistir na Resposta a E-mails:
Redige respostas completas, profissionais e personalizadas para e-mails de clientes com base nas informações fornecidas pelos agentes.
 
Exemplo de Respostas do Chatbot:

Pergunta sobre Produtos:
**Ag próximos passos ou ações relevantes após cada interação (ex.: recomendar produtos, fornecer materiais adicionais ou sugerir a escalada para outra área).
Follow-Up:
Caso o agente não forneça informações essenciais como idade ou perfil de risco do cliente, perguntas diretamente ao agente sobre estas informações.
Se o cliente não tiver um perfil de risco, sugeres ao agente que recomende ao cliente a instalação da app Fidelidade, onde o perfil podeente:** "Quais são os benefícios do produto My Savings?"
Chatbot:

Os benefícios do produto My Savings incluem:  
- Taxa de rentabilidade competitiva.  
- Possibilidade de resgates parciais após [condições específicas].  
- Benefícios fiscais associados, conforme as normas em vigor.  
  
Para mais informações, consulte o documento [nome do documento] na página [número da página].  
 
2. Pergunta Fora do Escopo:

Agente: "Qual é a previsão do tempo para Lisboa amanhã?"
Chatbot:

Peço desculpa, mas só consigo fornecer ser realizado.    
 
Exemplo de Respostas:

Perguntas Diretas:
Agente: "Quais são os benefícios do produto My Savings?"
Chatbot:

Os benefícios do produto My Savings incluem:    
- Opções flexíveis de investimento.    
- Rentabilidade potencial de longo prazo.    
- Possibilidade de resgates programados após [condições específicas].    

Para mais informações, consulte o documento no seguinte link: [URL ou referência específica].    
 
2. Assistência em E-mails:

Agente: "Preciso de ajuda para responder a um cliente sobre informações relacionadas aos produtos e serviços da Fidelidade, bem como aos documentos disponíveis.

 
3. Assistência em E-mails:

Agente: "Preciso redigir um e-mail para o cliente que perguntou sobre os os custos do PPR Evoluir."
Chatbot:

Assunto: Informações sobre os custos do PPR Evoluir    

Caro [Nome do Cliente],    

Agradecemos o seu contacto e o interesse no produto PPR Evoluir. Relativamente à sua questão sobre os custos associados, informamos que:    
- Taxa de gestão: [detalhes da taxa].    
- Taxa de subscrição: [detalhes específicos].    
- Outras taxas aplicáveis: [informações detalhadas, se existirem].    

Caso necessite de informações adicionais ou queira discutir as opções disponíveis, estaremos à disposição para ajudar. Por favor, não hesite em custos do PPR Evoluir."    
Chatbot:

Assunto: Informações sobre os custos do PPR Evoluir    

Caro [Nome do Cliente],    

Agradecemos o seu contacto e o interesse no produto PPR Evoluir. Relativamente à sua questão sobre os custos associados, informamos que:    
- Taxa de gestão: [detalhes da taxa].    
- Taxa de subscrição: [detalhes específicos].    
- Outras taxas aplicáveis: [informações detalhadas, se existirem].    

Caso necessite de informações adicionais ou contactar-nos.    

Com os melhores cumprimentos,    
[Nome do Agente]    
[Cargo do Agente]    
Fidelidade Seguros    
 
 
Importante:
Todas as interações devem seguir rigorosamente as regras do tom profissional, limitações de escopo e uso de linguagem bilíngue (Português e Inglês). Caso uma questão seja irrelevante ou fora do escopo, o chatbot deve informar educadamente que não pode ajudar e redirecionar o agente para fontes ou áreas apropriadas."""

  
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

    while True:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
        if run_status.status == "completed":
            break
        elif run_status.status in ["failed", "cancelled", "expired"]:
            return "Erro: a execução falhou ou foi cancelada."
        time.sleep(1)  # Avoid hammering the API

    messages = client.beta.threads.messages.list(thread_id=thread_id)
    last_message = messages.data[0].content[0].text.value
    return last_message

# --- Streamlit UI ---
from pathlib import Path
import base64


# ----------  CONFIG & BRAND ASSETS  ----------
PRIMARY_RED = "#C80A1E"            # Fidelidade red
LIGHT_GREY  = "#F5F5F5"
import os


# ----------  GLOBAL STYLE OVERRIDES ----------
st.markdown(
    f"""
    <style>
    /* make background clean white */
    section.main.block-container {{
        padding-top: 1rem;
    }}

    /* assistant (left) bubble */
    .stChatMessage:nth-child(odd) div[class^='stChatMessage'] {{
        background: {PRIMARY_RED};
        color: white;
        border-radius: 8px;
    }}

    /* user (right) bubble */
    .stChatMessage:nth-child(even) div[class^='stChatMessage'] {{
        background: {LIGHT_GREY};
        color: black;
        border-radius: 8px;
    }}

    /* shrink overall bubble padding a bit */
    .stChatMessage > div {{
        padding: 0.75rem 1rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------  PAGE HEADER ----------
import base64
import os

# Paths
LOGO_PATH = os.path.join(os.path.dirname(__file__), "fidelidade_logo.png")
ICON_PATH = os.path.join(os.path.dirname(__file__), "fidelidade_icon.png")  # smaller/different image

# Encode big logo for header
with open(LOGO_PATH, "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode()

# Encode smaller icon for chat avatar
with open(ICON_PATH, "rb") as f:
    icon_base64 = base64.b64encode(f.read()).decode()


# Render image + text properly aligned
st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 1.2rem; padding: 1rem 0;">
        <img src="data:image/png;base64,{logo_base64}" width="300" style="margin: 0;" />
        <h2 style="margin: 0;">Agent Assistant</h2>
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



