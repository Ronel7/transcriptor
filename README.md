#  Transcripteur Automatique de Vidéo — Assistant Intelligent d'Analyse

Ce projet met en place un **transcripteur automatique de vidéo** couplé à un pipeline de **RAG (Retrieval-Augmented Generation) Multimodal**. Il permet de transcrire automatiquement le contenu audio/visuel d'une vidéo, d'ingérer cette transcription dans une base de données vectorielle locale

##  Fonctionnalités
- **Transcription Multimodale :** Utilisation de `Gemini 2.5 Flash` pour transcrire l'audio et décrire les actions visuelles d'une vidéo.
- **Ingestion Vectorielle Locale :** Découpage du texte avec `LangChain` et vectorisation avec le modèle open-source multilingue `BAAI/bge-m3` (Hugging Face) stocké à 100% en local.
- **Stockage Persistant :** Sauvegarde des embeddings dans une base vectorielle `ChromaDB`.
- **Interface Chat & Inspection :** Application `Streamlit` pour interroger la transcription et inspecter les fragments mémorisés.

---

##  Structure du Projet

```text
├── ingestion.py          # Script de transcription et d'ingestion (Backend)
├── app.py               # Interface utilisateur Streamlit (Frontend)
├── .env                 # Variables d'environnement (Clés API - Caché)
├── .gitignore           # Fichiers à ignorer par Git
└── README.md            # Documentation du projet
```

## 🛠️ Installation et Prérequis

### 1. Cloner le projet et installer les dépendances
Assurez-vous d'avoir Python 3 installé, puis lancez la commande suivante pour installer tous les packages requis :

```bash
pip install google-genai streamlit python-dotenv langchain langchain-text-splitters langchain-community langchain-core chromadb sentence-transformers
```

### 2. Configuration des variables d'environnement
Créez un fichier `.env` à la racine du projet et ajoutez votre clé API Google AI Studio :

## 💻 Utilisation

Le projet fonctionne en deux étapes distinctes :

### Étape 1 : Transcription et ingestion d'une vidéo
Ouvrez le fichier `ingestion.py`, configurez le chemin de votre fichier vidéo local, puis exécutez le script pour lancer la transcription et créer la base vectorielle :

```bash
python ingestion.py
```

> **Note :** Au premier lancement, le modèle d'embedding `bge-m3` sera téléchargé automatiquement (environ 1.2 Go). Les lancements suivants seront instantanés.

### Étape 2 : Lancement de l'interface de Chat
Une fois la vidéo transcrite et ingérée dans `chroma_video_db/`, lancez l'application Streamlit pour interagir avec la transcription :

```bash
streamlit run app.py
```

L'interface s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`. Vous pourrez alors utiliser l'onglet **Discuter** pour poser vos questions sur le contenu de la vidéo, ou l'onglet **Inspecter** pour voir les fragments de transcription bruts.