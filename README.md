## 🎥 Transcriptor - Assistant RAG Vidéo Multimodal

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white)
![Gemini](https://img.shields.io/badge/Google_Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F00?style=for-the-badge&logo=database&logoColor=white)

**Transcriptor** est une application web interactive propulsée par l'Intelligence Artificielle qui permet d'analyser des fichiers vidéo et de dialoguer avec leur contenu. En combinant la puissance de l'analyse multimodale et la vitesse de l'inférence via Groq, l'application transforme n'importe quelle vidéo en une base de connaissances interrogeable en temps réel.


---

## ✨ Fonctionnalités Principales

* **🧠 Extraction Multimodale Complète :** Transcrit l'audio, décrit les actions visuelles clés, et génère un résumé structuré avec horodatages grâce à Gemini 2.5 Flash.
* **⚡ Chatbot RAG Ultra-Rapide :** Posez des questions en langage naturel sur la vidéo. Les réponses sont générées par **Llama-3.3-70b** via l'API Groq, offrant une latence quasi-nulle.
* **🔎 Traçabilité des Sources :** Chaque réponse générée s'accompagne d'un menu déroulant permettant de visualiser les fragments exacts de la transcription (chunks) utilisés par le modèle.
* **🎯 Détection d'Intention :** L'application comprend quand l'utilisateur demande "la transcription complète" et court-circuite le RAG pour livrer le document intégral instantanément.
* **🎨 UI/UX Moderne :** Interface fluide conçue avec Streamlit, intégrant un historique de chat persistant façon messagerie, un autoscroll intelligent, et des composants visuels épurés.

---

## 🏗️ Architecture Technique (Pipeline RAG)

Ce projet implémente une architecture hybride Cloud/Local pour optimiser à la fois les coûts, la vitesse et la précision :

1. **Phase d'Ingestion (Google GenAI) :** La vidéo uploadée est envoyée à l'API Gemini. Le modèle visionne le fichier et génère un document Markdown exhaustif (transcription, descriptions visuelles, points clés).
2. **Phase d'Indexation (LangChain & ChromaDB) :** Le document est découpé en fragments (`RecursiveCharacterTextSplitter` avec overlap) pour conserver le contexte. Ces fragments sont vectorisés localement par le modèle **HuggingFace `BAAI/bge-m3`** (optimisé CPU) et stockés dans une base de données ChromaDB.
3. **Phase de Génération (Groq) :** Lorsqu'une question est posée, ChromaDB effectue une recherche de similarité pour isoler le contexte pertinent. Ce contexte est envoyé au modèle open-source **Llama-3.3-70b-versatile** exécuté sur les LPU (Language Processing Units) de Groq pour une inférence extrêmement rapide.

---

## 🚀 Installation & Déploiement Local

### Prérequis
* Python 3.9+
* Un compte [Google AI Studio](https://aistudio.google.com/) (pour la clé API Gemini)
* Un compte [Groq Console](https://console.groq.com/) (pour la clé API Groq)

### Étapes d'installation

1. **Cloner le dépôt :**
   ```bash
   git clone [https://github.com/Ronel7/transcriptor.git](https://github.com/Ronel7/transcriptor.git)
   cd transcriptor
Créer un environnement virtuel (Recommandé) :
Installer les dépendances :Bashpip install -r requirements.txt
Configuration des variables d'environnement :Créez un fichier .env à la racine du projet et ajoutez vos clés :Extrait de codeAPI_KEY=votre_cle_api_google_gemini
GROQ_API_KEY=votre_cle_api_groq
Lancer l'application :Bashstreamlit run main.py


💻 Exemples d'utilisation
Une fois une vidéo indexée (ex: une conférence, un cours, un tutoriel), essayez les prompts suivants :"Donne-moi la transcription complète.""Fais-moi un résumé en 3 points de cette vidéo.""Que se passe-t-il à la minute 02:15 ?""Quels sont les concepts techniques abordés par le présentateur ?"🛠️ Stack Technologique DétailléeComposantTechnologie UtiliséeFrontend UIStreamlitFramework LLMLangChainModèle Vision / ExtractionGoogle Gemini 2.5 FlashModèle Chat / RaisonnementLlama 3.3 70B (via Groq)Modèle d'EmbeddingHuggingFace BAAI/bge-m3Base de Données VectorielleChromaDB (Local persistant)
