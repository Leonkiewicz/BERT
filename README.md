
# BERT from Scratch on MovieDialogs Dataset 🎬🧠  

## Overview  
This project aims to implement **BERT (Bidirectional Encoder Representations from Transformers)** from scratch, using conversational data from the **Cornell Movie-Dialogs Corpus**.  

The primary goal is to train BERT on real dialogue, starting with building a custom tokenizer and gradually moving towards full pretraining tasks like **Masked Language Modeling (MLM)** and **Next Sentence Prediction (NSP)**.  

---

## Current Progress 🚀  

- ✅ Entry data preparation
- ✅ **WordPiece Tokenizer** trained on the MovieDialogs dataset (`bert_tokenizer/`)  
- ⏳ MLM & NSP not yet implemented   
- ❌ Building BERT architecture   
- ❌ Preparing training routines 

---

## Project Structure  

```
BERT/
│
├── bert_tokenizer/
│   └── bert-movie_corp-vocab.txt        # Custom tokenizer vocabulary
│
├── data/
│   ├── dataset.py
│   ├── get_data.py
│   └── preprocessing.py
│
├── datasets/
│   ├── aclImdb/                         # IMDB dataset (optional use)
│   └── cornell movie-dialogs corpus/    # Main conversational dataset
│
├── model/
│   ├── bert.py                          # BERT model implementation
│   └── layers.py                        # Transformer layers
│
├── training/
│   ├── train.py                         # Training loop
│   └── optimizer.py                     # Optimizer setup
│
├── utils/
│   └── helpers.py                       # Utility functions
│
├── main.py                              # Entry point
└── early_development.ipynb              # Experimentation notebook
```

---

## How to Use 🚀  

### 1. Download the datasets  
Uncomment the `download_files()` line inside `main.py` to automatically download the **Cornell Movie-Dialogs Corpus** and the **IMDB dataset**.  
Alternatively, you can download them manually and place them inside the `datasets/` folder.  

### 2. Preprocess the data  
Run the `main.py` file to preprocess the corpus and train the tokenizer:  
```bash  
python main.py  
```  
This will:  
- Extract question-answer pairs from the movie dialogues.  
- Chunk the data into manageable text files in `datasets/MovieDialogsChunks/`.  
- Train a custom **WordPiece tokenizer** on the movie dialogue data.  

### 3. Train BERT (coming soon)  
Once **MLM** and **NSP** are implemented, you will be able to kick off the BERT pretraining like this:  
```bash  
python training/train.py  
```  

---

## Upcoming Work 📝  

- [ ] Implement **Masked Language Modeling (MLM)**  
- [ ] Implement **Next Sentence Prediction (NSP)**  
- [ ] Integrate pretraining loop  
- [ ] Evaluate embeddings  
- [ ] Fine-tune on downstream tasks (e.g., sentiment analysis on IMDB)  

---

## References  

- [BERT Paper](https://arxiv.org/abs/1810.04805)  
- [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)  
- [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)  
- [Mastering BERT Model: Building it from Scratch with Pytorch](https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891)

---

