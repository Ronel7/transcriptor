import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from google import genai


load_dotenv()
# Configuration de la page Streamlit
st.set_page_config(page_title="Video RAG Assistant", page_icon="🎥", layout="wide")
st.title(" Assistant Intelligent - Analyse & Chat Vidéo")

# Configuration des clés et modèles 
CLE_API = os.getenv("API_KEY")
NOM_MODELE_EMBEDDING = "BAAI/bge-m3"
DOSSIER_DB = "C:/Users/Elitebook 840 G6/Downloads/Transcriptor/chroma_video_db"

# Initialisation du client Gemini pour la phase de réponse
client = genai.Client(api_key=CLE_API)

# Chargement de la base vectorielle 
@st.cache_resource
def charger_base_vectorielle():
    if not os.path.exists(DOSSIER_DB):
        return None
    embeddings = HuggingFaceEmbeddings(
        model_name=NOM_MODELE_EMBEDDING,
        model_kwargs={'device': 'cpu'}
    )
    return Chroma(persist_directory=DOSSIER_DB, embedding_function=embeddings)

vectordb = charger_base_vectorielle()

# --- INTERFACE ---
if vectordb is None:
    st.warning(" La base vectorielle n'existe pas encore. Lance d'abord ton script `modelvid.py` pour ingérer une vidéo.")
else:
    # Création de deux onglets : un pour le Chat, un pour Inspecter la base
    onglet_chat, onglet_inspecter = st.tabs(["💬 Discuter avec la vidéo", "🔍 Inspecter les données stockées"])

    # --- ONGLET 1 : CHAT (RECHERCHE RAG) ---
    with onglet_chat:
        st.subheader("Pose une question sur le contenu de tes vidéos")
        query = st.text_input("Exemple : Que se passe-t-il à la 15ème seconde ? / De quoi parle la vidéo ?")

        if query:
            with st.spinner("Recherche des passages pertinents et génération de la réponse..."):
                # 1. Recherche de similarité dans ChromaDB (On récupère les 3 fragments les plus proches)
                docs = vectordb.similarity_search(query, k=3)
                
                # On assemble les fragments trouvés pour donner du contexte à Gemini
                contexte_extrait = "\n\n".join([d.page_content for d in docs])
                
                # 2. Construction du prompt pour Gemini
                prompt_rag = f"""
                Tu es un assistant virtuel. Tu dois répondre à la question de l'utilisateur en te basant UNIQUEMENT sur le contexte extrait d'une analyse vidéo ci-dessous.
                Si la réponse n'est pas dans le contexte, dis poliment que tu ne sais pas.

                CONTEXTE EXTRAIT :
                {contexte_extrait}

                QUESTION DE L'UTILISATEUR :
                {query}
                """
                
                # 3. Génération de la réponse avec Gemini 2.5 Flash
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt_rag
                )
                
                # Affichage de la réponse
                st.markdown("### 🤖 Réponse de l'assistant :")
                st.write(response.text)
                
                # Affichage des sources utilisées pour la transparence
                with st.expander(" Voir les fragments exacts extraits de ChromaDB"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Fragment {i+1} (Source: {doc.metadata.get('source', 'Inconnue')}) :**")
                        st.info(doc.page_content)

    # --- ONGLET 2 : INSPECTER LA BASE ---
    with onglet_inspecter:
        st.subheader("Contenu brut mémorisé dans ChromaDB")
        
        # Astuce pour voir ce qu'il y a dans ChromaDB : on récupère toutes les données
        donnees = vectordb.get()
        
        if donnees and 'documents' in donnees:
            st.write(f"La base contient actuellement **{len(donnees['documents'])} fragments** de texte.")
            
            for idx, text in enumerate(donnees['documents']):
                source = donnees['metadatas'][idx].get('source', 'Inconnue') if donnees['metadatas'] else 'Inconnue'
                with st.expander(f"Fragment #{idx + 1} | Source : {os.path.basename(source)}"):
                    st.text(text)
        else:
            st.info("La base est vide ou inaccessible.")