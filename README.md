# 🛡️ SafeGuard AI — Toxic Content Detector

**A production-ready NLP application that detects toxic content in text using a fine-tuned DistilBERT transformer model — trained on the Jigsaw Toxic Comment Classification dataset.**

</div>

---

## 📌 What is this project?

SafeGuard AI is an end-to-end NLP pipeline that:

1. **Trains** a DistilBERT transformer on 159K real-world Wikipedia comments
2. **Classifies** any text as `SAFE` or `TOXIC` with a confidence score
3. **Serves** predictions through a clean Streamlit web interface
4. **Deploys** publicly via Streamlit Cloud — no setup required for users

Whether you're moderating a forum, filtering user-generated content, or exploring NLP — this project covers the complete lifecycle from raw data to deployed app.

---

## 🏗️ Architecture & Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING  (Kaggle)                       │
│                                                              │
│  train.csv  ──►  Tokenizer  ──►  DistilBERT  ──►  Fine-tune │
│  (159K rows)     (max 128 tokens)  (66M params)  (3 epochs) │
│                                                              │
│                          ▼                                   │
│                   saved_model/                               │
│                   ├── model/         (weights)               │
│                   ├── tokenizer/     (vocab)                 │
│                   └── config.json   (metadata)               │
└──────────────────────────┬──────────────────────────────────┘
                           │  download & commit
┌──────────────────────────▼──────────────────────────────────┐
│                   INFERENCE  (Streamlit)                      │
│                                                              │
│  User Input  ──►  analyser.py  ──►  Softmax  ──►  Label     │
│  (any text)       (tokenize +       (probs)     SAFE/TOXIC   │
│                    forward pass)                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Dataset

**Jigsaw Toxic Comment Classification Challenge** — [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

| Property | Details |
|---|---|
| **Source** | Wikipedia Talk Pages (real user comments) |
| **Size** | 159,571 training samples |
| **Labels** | `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` |
| **Task** | Binary classification → `SAFE` (0) or `TOXIC` (1) |
| **Class balance** | ~90.4% Safe · ~9.6% Toxic |
| **Language** | English |

The binary label is derived by taking the **max across all 6 categories** — if any category is positive, the comment is considered toxic.

### Label Distribution

```
toxic          ████████████████░░░░░░░░░░░░░░  15,294
obscene        ████████░░░░░░░░░░░░░░░░░░░░░░   8,449
insult         ███████░░░░░░░░░░░░░░░░░░░░░░░   7,877
severe_toxic   ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░   1,595
identity_hate  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░   1,405
threat         █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░     478
```

---

## 🤖 Technology Stack

| Layer | Technology | Why we chose it |
|---|---|---|
| **Model** | DistilBERT (distilbert-base-uncased) | 40% smaller than BERT, 60% faster, retains 97% of BERT's performance |
| **Framework** | HuggingFace Transformers | Industry standard, easy fine-tuning API, model hub integration |
| **Training** | PyTorch + HuggingFace Trainer | Handles fp16, gradient accumulation, early stopping out of the box |
| **Tokenizer** | WordPiece (128 max tokens) | Handles OOV words, subword tokenization |
| **UI** | Streamlit | Fastest path from model to interactive web app in Python |
| **Training Platform** | Kaggle (T4 GPU) | Free GPU, direct dataset access, no setup |
| **Deployment** | Streamlit Cloud | Free hosting, GitHub integration, one-click deploy |

### Why DistilBERT over other models?

```
Model           ROC-AUC    Speed      Size
─────────────────────────────────────────
TF-IDF + LR      ~0.960    ⚡⚡⚡⚡    ~50MB
DistilBERT       ~0.985    ⚡⚡⚡      ~250MB   ← We use this
BERT-base        ~0.987    ⚡⚡        ~450MB
RoBERTa-base     ~0.990    ⚡⚡        ~500MB
```

DistilBERT hits the sweet spot — near-BERT accuracy at a fraction of the compute cost, making it practical for deployment on free-tier servers.

---

## 📊 Model Performance

Trained for **3 epochs** on Kaggle T4 GPU (~55 minutes):

| Epoch | Train Loss | Val Loss | ROC-AUC | Accuracy |
|---|---|---|---|---|
| 1 | 0.1634 | 0.2049 | 0.9835 | 96.62% |
| 2 | 0.1384 | 0.1983 | **0.9857** | 96.28% |
| 3 | 0.0958 | 0.1995 | 0.9844 | 96.64% |

> ✅ Best model saved at **Epoch 2** (highest ROC-AUC = **0.9857**)

### What ROC-AUC 0.985 means

- A random classifier scores **0.50**
- A good model scores **> 0.90**
- Our model scores **0.985** — it correctly ranks a toxic comment above a safe one 98.5% of the time

---

## 📁 Project Structure

```
jigsaw-analyser/
│
├── app.py                        # Streamlit web application
├── analyser.py                   # Model loading & inference logic
├── requirements.txt              # Python dependencies
├── .gitignore                    # Excludes saved_model/ (too large for GitHub)
│
├── notebook/                     # 📓 Training reference
│   └── jigsaw_training.ipynb     # Full Kaggle training notebook
│
├── saved_model/                  # ⚠️ NOT in GitHub — download separately
│   ├── config.json               # Model metadata & thresholds
│   ├── model/                    # Fine-tuned DistilBERT weights
│   └── tokenizer/                # Tokenizer vocab & config
│
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/jigsaw-analyser.git
cd jigsaw-analyser
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get the trained model

The full training notebook is at `notebook/jigsaw_training.ipynb`. Run it on Kaggle (free T4 GPU):

1. Upload the notebook to [kaggle.com/code](https://kaggle.com/code)
2. Add the [Jigsaw dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) as input
3. Run all cells — training takes ~55 minutes
4. Download `saved_model.zip` from the Output panel
5. Extract it into the project root as `saved_model/`


### 5. Run the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🚀 Deployment

This app is deployed on **Streamlit Cloud**:

1. Push this repo to GitHub (without `saved_model/`)
2. Upload model weights to HuggingFace Hub
3. Go to [share.streamlit.io](https://share.streamlit.io) → New App
4. Select your repo → `app.py` → Deploy

---


Made with ❤️ · If this helped you, consider giving it a ⭐

</div>