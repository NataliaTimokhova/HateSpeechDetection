# HateSpeechDetection

A project to recognise hate speech in text messages using machine learning and NLP. This project uses the public dataset from HASOC (Task 1) and focuses on binary classification of offensive vs. non-offensive content in social media posts.

## Goals
- Automated detection of hate speech in short social media texts
- Exploration of different pooling strategies and custom classification heads
- Use of modern transformer-based models (BERT, RoBERTa)
- Transparent and reproducible experimentation using Weights & Biases (WandB)


## Architecture Overview

Our model pipeline consists of three modular stages:

1. **Pretrained Transformer Models**
   - `bert-base-uncased`
   - `roberta-base`
   - `roberta-large`
   - `cardiffnlp/twitter-roberta-base-sentiment-latest`

2. **Pooling Layers**  
   We compare multiple strategies for aggregating token embeddings:
   - **CLS Token**: Using the embedding at the [CLS] position
   - **Mean Pooling**: Averaging embeddings across tokens
   - **Max Pooling**: Selecting maximum per feature dimension
   - **Attention Pooling**: Learned attention weights with softmax temperature and masking

3. **Custom Classifier Heads**
   - **Shallow ANN**: Single hidden layer
   - **Deeper ANN**: Two or more hidden layers with dropout
   - **CNN**: 1D convolution over the token sequence for local pattern extraction and dimensionality reduction

Fine-tuning is applied selectively (e.g. first one or two transformer layers unfrozen) to improve generalization and better integrate with the custom pooling and classifier architecture.


## Setup
```bash
pip install -r requirements.txt
```

## Training & Experimentation

All experiments are tracked via Weights & Biases, enabling:
  - Comparison of pooling strategies
  - Evaluation of different classifier heads
  - Analysis of transformer variant performance

## Results Summary
  - Attention pooling combined with a deeper ANN showed the best F1 performance
  - Partial fine-tuning of transformer layers reduced overfitting
  - CNN heads effectively compressed high-dimensional embeddings and captured local patterns

## References
### Dataset & Shared Task
HASOC 2019 - Hate Speech and Offensive Content Identification
Organizers:
- Thomas Mandl: University of Hildesheim, Germany
- Sandip Modha: DA-IICT, Gandhinagar, India
- Chintak Mandlia: infoAnalytica Consulting Pvt. Ltd.
- Daksh Patel: Dalhousie University, Halifax, Canada
- Aditya Patel: Dalhousie University, Halifax, Canada
- Mohana Dave: LDRP-ITR, Gandhinagar, India

### Key Architecture Reference
Lin et al., A Structured Self-attentive Sentence Embedding, ICLR 2017.
