import time
import os
from google import genai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()
# 1. Initialisation du client (ATTENTION : Utilise une variable d'environnement en production)
CLE_API = os.getenv("API_KEY")
MODELE_EMBEDDING = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

client = genai.Client(api_key=CLE_API)

def analyser_video(chemin_video):
    print(f"\n--- ÉTAPE 1 : EXTRACTION ---")
    print(f"Upload de la vidéo {chemin_video} en cours...")
    
    video_file = client.files.upload(file=chemin_video)
    print(f"Fichier uploadé avec succès. Nom distant : {video_file.name}")
    
    print("Traitement de la vidéo par l'API de Google...")
    while video_file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(10)
        video_file = client.files.get(name=video_file.name)
        
    if video_file.state.name != "ACTIVE":
        raise ValueError(f"Le traitement a échoué. Statut : {video_file.state.name}")
        
    print("\nVidéo prête ! Génération de l'analyse avec Gemini 2.5 Flash...")

    prompt = """
    Tu es un assistant expert en analyse vidéo. Visionne cette vidéo attentivement et fournis-moi les éléments suivants structurés en Markdown :
    
    1. **Transcription Audio :** Ce qui est dit, avec les horodatages (ex: [00:15] Texte...).
    2. **Description Visuelle :** Les actions clés qui se passent à l'écran, avec leurs horodatages.
    3. **Résumé Synthétique :** Un résumé global en un paragraphe de l'objectif et du contenu de la vidéo.
    4. **Points Clés :** Les 3 à 5 idées principales à retenir.
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[video_file, prompt]
    )
    
    print("Nettoyage du fichier distant...")
    client.files.delete(name=video_file.name)
    
    return response.text

def ingerer_dans_chroma(texte_analyse, chemin_source):
    print(f"\n--- ÉTAPE 2 : INGESTION RAG ---")
    print("Découpage de l'analyse en fragments (Chunking)...")
    
    # 1. On découpe le long résumé en petits blocs de 1000 caractères
    # L'overlap (chevauchement) de 200 permet de ne pas couper une idée en plein milieu
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(texte_analyse)
    
    # 2. On crée des "Documents" LangChain en y associant la source (les métadonnées)
    documents = [Document(page_content=chunk, metadata={"source": chemin_source}) for chunk in chunks]
    
    print("Génération des embeddings et sauvegarde dans ChromaDB...")
    
    # 3. On utilise le modèle d'embedding de Google (très rapide et adapté au texte français)
    print(f"Chargement du modèle d'embedding local : {MODELE_EMBEDDING}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODELE_EMBEDDING,
        model_kwargs={'device': 'cpu'} )

    
    # 4. On sauvegarde la base de données dans un dossier local
    dossier_db = "C:/Users/Elitebook 840 G6/Downloads/chroma_video_db"
    
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=dossier_db
    )
    
    print(f"Succès ! {len(chunks)} fragments ont été vectorisés et sauvegardés dans '{dossier_db}'.")
    return vectordb

# --- Exécution ---
if __name__ == "__main__":
    try: 
        chemin_video = "chemin_video"
        
        # 1. On extrait le texte de la vidéo
        resultat_texte = analyser_video(chemin_video)
        
        # (Optionnel) On sauvegarde aussi le texte brut pour garder une trace lisible
        with open("analyse_video.md", "w", encoding="utf-8") as f:
            f.write(resultat_texte)
            
        # 2. On ingère ce texte dans la base vectorielle
        ingerer_dans_chroma(resultat_texte, chemin_video)
        
    except Exception as e:
        print(f"\nErreur rencontrée : {e}")