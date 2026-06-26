import streamlit as st
import os
import tempfile
import time
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from google import genai
from groq import Groq

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Transcriptor", page_icon="🎥", layout="wide")

MODELE_EMBED     = "BAAI/bge-m3"
DOSSIER_DB       = "./chroma_video_db"
FICHIER_ANALYSE  = "./analyse_video.md"

# Mots-clés qui signifient "donne-moi tout le texte"
MOTS_CLES_FULL = [
    "transcription complète", "transcription complete", "transcription entière",
    "transcription entiere", "tout ce qui est dit", "tout le texte",
    "tout le contenu", "résumé complet", "resume complet",
    "donne-moi tout", "donne moi tout", "montre tout",
    "l'intégralité", "l integralite", "texte complet",
]

# ─── STYLE ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

.stApp { background: #0f0f13; color: #e0e0f0; }
.block-container { padding: 2rem 2rem 6rem 2rem; } /* padding-bottom pour la barre fixe */

/* Sidebar */
[data-testid="stSidebar"] { background: #16161f; border-right: 1px solid #252535; }
[data-testid="stSidebar"] .stButton > button {
    background: #3333aa; color: white; border: none;
    border-radius: 8px; font-weight: 600; width: 100%;
    padding: 0.55rem 1rem; margin-top: 0.5rem;
}
[data-testid="stSidebar"] .stButton > button:hover { background: #4444cc; color: white; }
[data-testid="stFileUploader"] { border: 2px dashed #2a2a45; border-radius: 10px; background: #13131c; }

/* Titres */
h1 { color: #ffffff; font-size: 1.6rem; font-weight: 700; margin-bottom: 0.2rem; }
h3 { color: #aaaacc; font-size: 1rem; font-weight: 400; margin-top: 0; }

/* Bulles */
.bubble-user {
    background: #2d2d8a; color: #fff;
    border-radius: 14px 14px 4px 14px;
    padding: 0.65rem 1rem; margin: 0.5rem 0 0.5rem 20%;
    font-size: 0.92rem; line-height: 1.5; word-wrap: break-word;
}
.bubble-ai {
    background: #1e1e2e; border: 1px solid #2a2a40; color: #e0e0f0;
    border-radius: 14px 14px 14px 4px;
    padding: 0.65rem 1rem; margin: 0.5rem 20% 0.5rem 0;
    font-size: 0.92rem; line-height: 1.6; word-wrap: break-word;
}

/* Barre de saisie fixée en bas */
.input-bar-fixed {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: #16161f;
    border-top: 1px solid #252535;
    padding: 0.8rem 1.5rem;
    z-index: 999;
    display: flex;
    gap: 0.6rem;
    align-items: center;
}

/* Expander */
[data-testid="stExpander"] { border: 1px solid #252535 !important; border-radius: 8px !important; }
details summary { color: #8888aa !important; font-size: 0.82rem !important; }
hr { border-color: #1e1e2e; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "db_prete" not in st.session_state:
    st.session_state.db_prete = os.path.exists(DOSSIER_DB)

# ─── CLIENTS ──────────────────────────────────────────────────────────────────
@st.cache_resource
def charger_embeddings():
    return HuggingFaceEmbeddings(model_name=MODELE_EMBED, model_kwargs={"device": "cpu"})

@st.cache_resource
def charger_vectordb():
    return Chroma(persist_directory=DOSSIER_DB, embedding_function=charger_embeddings())

@st.cache_resource
def get_gemini():
    key = os.getenv("API_KEY")
    if not key:
        st.error("❌ GEMINI_API_KEY manquante dans ton fichier .env")
        st.stop()
    return genai.Client(api_key=key)

@st.cache_resource
def get_groq():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        st.error("❌ GROQ_API_KEY manquante dans ton fichier .env")
        st.stop()
    return Groq(api_key=key)

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def est_question_full(query: str) -> bool:
    """Détecte si l'utilisateur veut la totalité du texte analysé."""
    q = query.lower()
    return any(mot in q for mot in MOTS_CLES_FULL)

def lire_analyse_complete() -> str | None:
    """Lit le fichier markdown sauvegardé lors de l'ingestion."""
    if os.path.exists(FICHIER_ANALYSE):
        with open(FICHIER_ANALYSE, "r", encoding="utf-8") as f:
            return f.read()
    return None

def repondre(query: str) -> tuple[str, list]:
    """
    Retourne (réponse, docs_sources).
    - Si question full → retourne le texte complet directement.
    - Sinon → RAG ChromaDB + Groq.
    """
    # CAS 1 : l'utilisateur veut tout le texte
    if est_question_full(query):
        texte = lire_analyse_complete()
        if texte:
            return texte, []
        else:
            return "⚠️ Aucune analyse trouvée. Assure-toi d'avoir analysé une vidéo.", []

    # CAS 2 : question normale → RAG
    vectordb        = charger_vectordb()
    docs_pertinents = vectordb.similarity_search(query, k=4)
    contexte        = "\n\n".join([d.page_content for d in docs_pertinents])

    messages = [{"role": "system", "content": f"""Tu es un assistant qui répond aux questions sur une vidéo.
Réponds UNIQUEMENT à partir du contexte ci-dessous.
Si la réponse n'y figure pas, dis-le poliment.

CONTEXTE :
{contexte}"""}]

    for msg in st.session_state.chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": query})

    response = get_groq().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.4,
        max_tokens=800,
    )
    return response.choices[0].message.content, docs_pertinents

# ─── INGESTION ────────────────────────────────────────────────────────────────
def ingerer_video(chemin_tmp: str, nom_fichier: str, progress_bar):
    client_gemini = get_gemini()

    progress_bar.progress(10, text=" Upload vers Gemini…")
    video_file = client_gemini.files.upload(file=chemin_tmp)

    progress_bar.progress(20, text=" Traitement Gemini (1-2 min)…")
    while video_file.state.name == "PROCESSING":
        time.sleep(5)
        video_file = client_gemini.files.get(name=video_file.name)

    if video_file.state.name != "ACTIVE":
        raise RuntimeError(f"Gemini n'a pas pu traiter le fichier (statut : {video_file.state.name})")

    progress_bar.progress(50, text=" Génération de la transcription…")
    prompt = """
Tu es un expert en analyse vidéo. Visionne cette vidéo et fournis en Markdown :

1. **Transcription audio** — ce qui est dit, avec horodatages (ex : [00:15]).
2. **Description visuelle** — les actions clés à l'écran avec horodatages.
3. **Résumé synthétique** — un paragraphe sur l'objectif et le contenu.
4. **Points clés** — 3 à 5 idées principales.
"""
    response      = client_gemini.models.generate_content(model="gemini-2.5-flash", contents=[video_file, prompt])
    texte_analyse = response.text
    client_gemini.files.delete(name=video_file.name)

    # Sauvegarde du texte complet (utilisé pour les questions "full")
    with open(FICHIER_ANALYSE, "w", encoding="utf-8") as f:
        f.write(texte_analyse)

    progress_bar.progress(70, text=" Découpage du texte…")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks   = splitter.split_text(texte_analyse)
    docs     = [Document(page_content=c, metadata={"source": nom_fichier}) for c in chunks]

    progress_bar.progress(85, text=" Vectorisation (BGE-M3)…")
    Chroma.from_documents(documents=docs, embedding=charger_embeddings(), persist_directory=DOSSIER_DB)

    progress_bar.progress(100, text=f"✅ {nom_fichier} — {len(chunks)} fragments indexés.")
    st.session_state.db_prete = True
    st.session_state.chat_history = []
    charger_vectordb.clear()

# ─── SIDEBAR : ÉTAPE 1 ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 1. Upload ta vidéo")
    uploaded = st.file_uploader("MP4, AVI, MOV, MKV", type=["mp4", "avi", "mov", "mkv"], label_visibility="collapsed")

    if uploaded:
        st.video(uploaded)
        if st.button(" Analyser & Indexer"):
            ext = "." + uploaded.name.rsplit(".", 1)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(uploaded.read())
                chemin_tmp = tmp.name
            bar = st.progress(0, text="Démarrage…")
            try:
                ingerer_video(chemin_tmp, uploaded.name, bar)
                st.success("Vidéo prête ! Pose tes questions →")
            except Exception as e:
                st.error(f"❌ {e}")
            finally:
                os.unlink(chemin_tmp)

    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑️ Effacer la conversation"):
            st.session_state.chat_history = []
            st.rerun()

# ─── ZONE PRINCIPALE : ÉTAPE 2 ────────────────────────────────────────────────
st.markdown("# 🎥 Transcriptor")
st.markdown("### 2. Pose des questions sur ta vidéo")
st.markdown("---")

if not st.session_state.db_prete:
    st.info("👈 Upload et analyse une vidéo depuis le menu de gauche pour commencer.")
    st.stop()

# Conteneur scrollable pour les messages
chat_container = st.container()

with chat_container:
    for msg in st.session_state.chat_history:
        css = "bubble-user" if msg["role"] == "user" else "bubble-ai"
        st.markdown(f'<div class="{css}">{msg["content"]}</div>', unsafe_allow_html=True)

# ─── BARRE DE SAISIE FIXÉE EN BAS ────────────────────────────────────────────
with st.form("chat_form", clear_on_submit=True):
    col_input, col_btn = st.columns([6, 1])
    with col_input:
        query = st.text_input(
            "question",
            placeholder="Ex : Donne-moi la transcription complète / De quoi parle la vidéo ?",
            label_visibility="collapsed",
        )
    with col_btn:
        envoyer = st.form_submit_button("Envoyer")

# JS : scroll automatique vers le bas après chaque message
st.markdown("""
<script>
    function scrollToBottom() {
        const chatContainer = document.querySelector('[data-testid="stAppViewContainer"]');
        if (chatContainer) chatContainer.scrollTop = chatContainer.scrollHeight;
        window.scrollTo(0, document.body.scrollHeight);
    }
    setTimeout(scrollToBottom, 300);
</script>
""", unsafe_allow_html=True)

# ─── TRAITEMENT DE LA QUESTION ────────────────────────────────────────────────
if envoyer and query.strip():
    # Bulle utilisateur immédiate
    with chat_container:
        st.markdown(f'<div class="bubble-user">{query}</div>', unsafe_allow_html=True)

    with st.spinner("…"):
        reponse, docs_pertinents = repondre(query)

    # Bulle réponse
    with chat_container:
        st.markdown(f'<div class="bubble-ai">{reponse}</div>', unsafe_allow_html=True)

    # Sources (uniquement pour les réponses RAG, pas pour les full)
    if docs_pertinents:
        with st.expander("🔍 Fragments utilisés pour cette réponse"):
            for i, doc in enumerate(docs_pertinents):
                st.markdown(f"**Fragment {i+1}** — `{doc.metadata.get('source', 'Inconnue')}`")
                st.info(doc.page_content)

    # Sauvegarde historique
    st.session_state.chat_history.append({"role": "user",      "content": query})
    st.session_state.chat_history.append({"role": "assistant", "content": reponse})

    # Scroll vers le bas
    st.markdown("""
    <script>
        setTimeout(function() { window.scrollTo(0, document.body.scrollHeight); }, 200);
    </script>
    """, unsafe_allow_html=True)
