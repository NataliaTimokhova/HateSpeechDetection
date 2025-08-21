# Hate Speech Detection  

A research project to automatically recognize hate speech in short social media posts using **machine learning** and **natural language processing (NLP)**.  
We build on the **HASOC 2019 dataset (Task 1)** and focus on **binary classification**:  
- **HOF** → Hate/Offensive content  
- **NOT** → Non-offensive content  

This repository provides modular experiments with **transformer-based models** and **custom classifier heads**, aiming for transparency, reproducibility, and robust comparison.  

---

## 🚀 Goals
- Automate detection of hate and offensive speech in short texts.  
- Explore pooling strategies and classifier architectures beyond the default `[CLS]` token.  
- Evaluate modern transformer models (**BERT, RoBERTa, XLNet**).  
- Use **Weights & Biases (WandB)** for transparent tracking and reproducibility.  

---

## 🏗️ Architecture Overview  

Our pipeline is modular, consisting of **three stages**:  

### 1. Pretrained Transformer Models  
- `bert-base-uncased`  
- `roberta-base`  
- `roberta-large`  
- `cardiffnlp/twitter-roberta-base-sentiment-latest`  
- `xlnet-base-cased`  

### 2. Pooling Mechanisms  
- **CLS Token** → Standard `[CLS]` embedding  
- **Mean Pooling** → Average embeddings across valid tokens  
- **Max Pooling** → Max value per feature dimension  
- **Attention Pooling** → Learned token-level attention with masking and temperature scaling  

### 3. Custom Classifier Heads  
- **Shallow ANN** → Linear + dropout  
- **Deep ANN** → Multi-layer with dropout  
- **CNN** → 1D convolutions over token embeddings to capture local n-gram patterns  

Fine-tuning strategies:  
- Fully frozen encoders  
- Partially unfrozen top layers  
- Full fine-tuning (for select models)  

---

## ⚙️ Setup  

Clone this repository:  
```bash
git clone https://github.com/juliusatgit/HateSpeechDetection.git
cd HateSpeechDetection
```

Install dependencies:  
```bash
pip install -r requirements.txt
```

---

## 📊 Training & Experimentation  

We use **WandB** for experiment tracking:  
- Logs training/validation accuracy, loss, and F1  
- Supports hyperparameter sweeps  
- Enables side-by-side comparison of models  

Training script example:  
```bash
python train.py --model roberta-base --pooling attention --head deep --epochs 10
```

---

## ✅ Results  

| Model            | Config          | F1   | Accuracy |
|------------------|----------------|------|----------|
| **RoBERTa (base, fine-tuned)** | Attention + Deep ANN | **0.89** | 0.83 |
| BERT (fine-tuned) | CLS + Shallow ANN | 0.72 | 0.77 |
| XLNet (partial FT, top 2 layers) | CNN Head | 0.65 | 0.69 |
| XLNet (full FT)  | CNN Head | 0.53 | 0.53 |  

**Key Findings**  
- Fine-tuned **RoBERTa** consistently outperforms other models.  
- **Attention pooling + deep ANN** → strongest performance (macro-F1 = 0.89).  
- **Partial fine-tuning** reduces overfitting compared to fully frozen encoders.  
- **XLNet** underperforms on short, social-media style texts.  

---

## 📂 Dataset  

- **HASOC 2019** [link](https://hasocfire.github.io/hasoc/2019/)  
- Task 1: **HOF vs. NOT** classification (binary)  
- Preprocessing:  
  - Removed usernames, URLs, hashtags, special characters  
  - Lowercased text  
  - Balanced classes via class-weighting  

---

## 🔬 References  

- Mandl et al. *HASOC 2019: Hate Speech and Offensive Content Identification*  
- Liu et al. *RoBERTa: A Robustly Optimized BERT Pretraining Approach*  
- Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers*  
- Yang et al. *XLNet: Generalized Autoregressive Pretraining*  
- Lin et al. *A Structured Self-Attentive Sentence Embedding* (ICLR 2017)  

---

## 📌 Conclusion  

- **RoBERTa** is the most robust and efficient model for **HASOC Task 1**.  
- Fine-tuning improves adaptability to hate speech detection.  
- Future work: multilingual extensions, adversarial robustness, and deployment in moderation pipelines.  
