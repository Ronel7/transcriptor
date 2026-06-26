---
title: Transcriptor
emoji: 🎥
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.35.0"
app_file: main.py
pinned: false
---

# 🎥 Transcriptor — Assistant Intelligent d'Analyse Vidéo

Dépose une vidéo, obtiens instantanément sa transcription complète, son résumé, et discute avec son contenu grâce à l'IA.

## Fonctionnalités

- **Transcription multimodale** via Gemini 2.5 Flash (audio + description visuelle avec horodatages)
- **Résumé automatique** : points clés et synthèse extraits à l'ingestion
- **Chat RAG** : pose des questions libres sur le contenu, l'IA répond en se basant uniquement sur ta vidéo
- **Transcription complète à la demande** : demande "donne-moi la transcription complète" pour tout récupérer
- **Stockage persistant** : la base vectorielle survit aux redémarrages (dossier `/data` sur HuggingFace Spaces)

## Stack technique

| Rôle | Outil |
|---|---|
| Transcription & analyse vidéo | Gemini 2.5 Flash |
| Embeddings vectoriels | BAAI/bge-m3 (local, CPU) |
| Base vectorielle | ChromaDB |
| Chat RAG | Groq — Llama 3.3 70B |
| Interface | Streamlit |

---

## Déploiement sur HuggingFace Spaces

### 1. Créer le Space

1. Va sur [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choisis **Streamlit** comme SDK
3. Choisis **Persistent Storage** (obligatoire pour que ChromaDB survive aux redémarrages)

### 2. Configurer les secrets

Dans ton Space → **Settings → Variables and Secrets**, ajoute :

| Nom | Valeur |
|---|---|
| `GEMINI_API_KEY` | Ta clé Google AI Studio |
| `GROQ_API_KEY` | Ta clé Groq Console |

### 3. Pousser le code

```bash
# Clone ton Space HuggingFace
git clone https://huggingface.co/spaces/TON_USERNAME/TON_SPACE
cd TON_SPACE

# Copie les fichiers du projet
cp /chemin/vers/main.py .
cp /chemin/vers/requirements.txt .
cp /chemin/vers/README.md .

# Push
git add .
git commit -m "initial deploy"
git push
```

Le Space se construit automatiquement. Attends ~3-5 min (téléchargement du modèle bge-m3 ~1.2 Go au premier démarrage).

---

## Installation locale

```bash
git clone https://huggingface.co/spaces/TON_USERNAME/TON_SPACE
cd TON_SPACE
pip install -r requirements.txt
```

Crée un fichier `.env` à la racine :

```
GEMINI_API_KEY=AIza...
GROQ_API_KEY=gsk_...
```

Lance l'app :

```bash
streamlit run main.py
```

> Au premier lancement, le modèle d'embedding `bge-m3` (~1.2 Go) est téléchargé automatiquement. Les lancements suivants sont instantanés.

## Limite de taille vidéo

Gemini accepte les fichiers jusqu'à **2 Go**. Pour les vidéos très longues, extrais l'audio en MP3 d'abord pour réduire la taille.
