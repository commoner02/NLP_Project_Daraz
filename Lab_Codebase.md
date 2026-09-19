# Natural Language Processing (NLP) Lab Repository
### Comprehensive Curriculum, Code Walkthroughs, Mathematical Formulations, and Architectures

---

## 📌 Table of Contents
1. [Curriculum Overview](#-curriculum-overview)
2. [Repository Directory Tree](#-repository-directory-tree)
3. [Lab 01: Text Preprocessing & Spelling Correction](#-lab-01-text-preprocessing--spelling-correction)
   - [1.1 Regular Expressions (Regex) for Text Sanitization](#11-regular-expressions-regex-for-text-sanitization)
   - [1.2 Tokenization and Stop Word Removal](#12-tokenization-and-stop-word-removal)
   - [1.3 Stemming (Porter's Algorithm) vs. Lemmatization (WordNet)](#13-stemming-porters-algorithm-vs-lemmatization-wordnet)
   - [1.4 Levenshtein Edit Distance & Spell Checking Engine](#14-levenshtein-edit-distance--spell-checking-engine)
4. [Lab 02: Text Representation, N-Gram Modeling & Shannon's Game](#-lab-02-text-representation-n-gram-modeling--shannons-game)
   - [2.1 Vector Space Models: Bag of Words (BoW) & TF-IDF](#21-vector-space-models-bag-of-words-bow--tf-idf)
   - [2.2 N-Gram Language Modeling & Markov Assumption](#22-n-gram-language-modeling--markov-assumption)
   - [2.3 Maximum Likelihood Estimation (MLE) & Probability Tables](#23-maximum-likelihood-estimation-mle--probability-tables)
   - [2.4 Shannon's Guessing Game (1951) & Autoregressive Text Generation](#24-shannons-guessing-game-1951--autoregressive-text-generation)
5. [Lab 03: Word Embeddings & Foundational Classifiers](#-lab-03-word-embeddings--foundational-classifiers)
   - [3.1 Continuous Word Embeddings (Word2Vec Skip-Gram from Scratch)](#31-continuous-word-embeddings-word2vec-skip-gram-from-scratch)
   - [3.2 Generative Text Classification (Multinomial Naive Bayes)](#32-generative-text-classification-multinomial-naive-bayes)
   - [3.3 Discriminative Text Classification (Logistic Regression with SGD)](#33-discriminative-text-classification-logistic-regression-with-sgd)
   - [3.4 Embedding Aggregation & The "Anagram Dilemma"](#34-embedding-aggregation--the-anagram-dilemma)
   - [3.5 Foundational Paradigm Comparison Matrix](#35-foundational-paradigm-comparison-matrix)
6. [Lab 04: PyTorch Sequence Models (RNN, LSTM, BiLSTM, Seq2Seq)](#-lab-04-pytorch-sequence-models-rnn-lstm-bilstm-seq2seq)
   - [4.1 Pretrained Google News Word2Vec Embedding Matrix](#41-pretrained-google-news-word2vec-embedding-matrix)
   - [4.2 Sequence Classification: Vanilla RNN vs. Stacked BiLSTM vs. Mean-Pooled BiLSTM](#42-sequence-classification-vanilla-rnn-vs-stacked-bilstm-vs-mean-pooled-bilstm)
   - [4.3 Sequence Labeling: POS Tagging with Bidirectional RNN & Loss Masking](#43-sequence-labeling-pos-tagging-with-bidirectional-rnn--loss-masking)
   - [4.4 Language Modeling & Autoregressive Text Generation with Temperature](#44-language-modeling--autoregressive-text-generation-with-temperature)
   - [4.5 Seq2Seq Machine Translation (English to Bengali) with Teacher Forcing](#45-seq2seq-machine-translation-english-to-bengali-with-teacher-forcing)
   - [4.6 PyTorch Sequence Tensor Shapes & Dimension Reference](#46-pytorch-sequence-tensor-shapes--dimension-reference)
7. [Lab 05: Transformer Architecture (Encoder-Only Classifier)](#-lab-05-transformer-architecture-encoder-only-classifier)
   - [5.1 Encoder-Only Transformer Architecture Overview](#51-encoder-only-transformer-architecture-overview)
   - [5.2 Sinusoidal Positional Encoding & Buffer Registration](#52-sinusoidal-positional-encoding--buffer-registration)
   - [5.3 Multi-Head Attention, Feed-Forward Sublayers & Padding Masking](#53-multi-head-attention-feed-forward-sublayers--padding-masking)
   - [5.4 Sequence Mean Pooling & Classification Head](#54-sequence-mean-pooling--classification-head)
   - [5.5 Transformer Tensor Shape Lifecycle & Data Flow Matrix](#55-transformer-tensor-shape-lifecycle--data-flow-matrix)
8. [Environment Setup & Installation](#-environment-setup--installation)

---

## 🔬 Curriculum Overview

This laboratory curriculum covers the progression of modern Natural Language Processing (NLP) — starting from deterministic string manipulation and heuristic preprocessing, progressing through statistical and count-based representations, foundational machine learning classifiers, recurrent deep learning sequence models, and culminating in attention-based Transformer architectures.

```
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│         LAB 01          │     │         LAB 02          │     │         LAB 03          │
│ • Regex & Cleaning      │ ──► │ • Bag of Words & TF-IDF │ ──► │ • Word2Vec (Skip-Gram)  │
│ • Tokenization & Stops  │     │ • N-Gram Language Model │     │ • Naive Bayes (Laplace) │
│ • Stemming vs. Lemmas   │     │ • MLE Probability Norm  │     │ • Logistic Regression   │
│ • Levenshtein Distance  │     │ • Shannon's Game & Gen  │     │ • Vector Aggregation    │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
                                                                             │
                                ┌─────────────────────────┐                  ▼
                                │         LAB 05          │     ┌─────────────────────────┐
                                │ • Transformer Encoder   │ ◄── │         LAB 04          │
                                │ • Positional Encoding   │     │ • Pretrained Word2Vec   │
                                │ • Multi-Head Attention  │     │ • RNN / Stacked BiLSTM  │
                                │ • Padding Mask & Pool   │     │ • POS Sequence Labeling │
                                │ • Binary Classification │     │ • Seq2Seq (En -> Bn)    │
                                └─────────────────────────┘     └─────────────────────────┘
```

---

## 📂 Repository Directory Tree

```
_NLP Lab/
├── README.md                                                         # Repository Master Documentation
├── .nlp_venv/                                                        # Python Virtual Environment
├── Lab01/                                                            # LAB 01: Preprocessing & Edit Distance
│   ├── Lab1.ipynb                                                    # Jupyter Notebook: Preprocessing & Spell Checker
│   └── NLP Lab 1_ Text Preprocessing and Spelling Correction.docx    # Reference Manual: Regex, Porter Rules & DP
├── Lab02/                                                            # LAB 02: Text Representation & N-Gram Modeling
│   ├── Lab2.ipynb                                                    # Jupyter Notebook: BoW, TF-IDF, N-Grams, Shannon
│   ├── test.ipynb                                                    # Scratch & Validation Notebook
│   ├── assignment116.py                                              # Standalone Script: Preprocessing & N-Gram Gen
│   ├── NLP Lab 2 Guide_ Text Representation and N-Gram...docx        # Reference Manual: Count Models & Markov Chains
│   ├── AnimeQuotes.csv                                               # Dataset: Anime Dialogue & Transcripts
│   ├── Sherlock Holmes.txt                                           # Corpus: The Adventures of Sherlock Holmes (Full)
│   ├── author-quote.txt                                              # Corpus: Large Multi-Author Quote Dataset
│   ├── ch1.txt                                                       # Corpus: Novel Chapter Sample Text
│   ├── sherlock.txt                                                  # Corpus: Short Sherlock Holmes Excerpt
│   └── SenGen/                                                       # Subpackage: Sentence Generation Sandbox
│       ├── assignment116.py                                          # N-Gram Sentence Generator Implementation
│       └── ch1.txt                                                   # Benchmark Chapter Corpus
├── Lab03/                                                            # LAB 03: Embeddings & Foundational Classifiers
│   ├── Lab3.ipynb                                                    # Jupyter Notebook: Word2Vec, NB, LogReg, Pooling
│   └── NLP Lab 3_ Word Embeddings _ Foundational Classifiers.docx    # Reference Manual: Gradient Math & Anagram Proof
├── Lab04/                                                            # LAB 04: PyTorch Recurrent Sequence Models
│   ├── Lab4.ipynb                                                    # Jupyter Notebook: RNN, BiLSTM, LM, Seq2Seq
│   ├── Lab4TaskB2.ipynb                                              # Lab Assignment: Mean-Pooled BiLSTM Classifier
│   └── NLP Lab 4 Reference Manual_ PyTorch Sequence Models...docx    # Reference Manual: Recurrent Math & Tensor Shapes
└── Lab05/                                                            # LAB 05: Transformer Sequence Classification
    ├── Lab5.ipynb                                                    # Jupyter Notebook: PyTorch Transformer Encoder
    └── NLP Lab_5 Transformer_Code_Walkthrough.docx                   # Reference Manual: Attention Math & Tensor Shapes
```

---

## 🧹 Lab 01: Text Preprocessing & Spelling Correction

Lab 01 establishes the foundational data hygiene pipeline required before feeding raw, noisy textual data into machine learning algorithms.

### 1.1 Regular Expressions (Regex) for Text Sanitization
Text scraped from web sources, chat logs, or OCR outputs contains HTML tags, non-alphanumeric noise, and inconsistent whitespace.

```python
import re

raw_text = "Hello!!! Welcome to the NLP lab in 2026. This text has <br> HTML tags, numbers like 123, and symbols #$%^."

# 1. Strip HTML tags
no_html = re.sub(r'<[^>]+>', '', raw_text)

# 2. Retain only alphabetic letters and whitespace
cleaned_text = re.sub(r'[^A-Za-z\s]', '', no_html)

# 3. Collapse redundant whitespace sequences
cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
```

#### Regex Pattern Reference Table
| Category | Pattern | Description & Meaning |
| :--- | :--- | :--- |
| **Metacharacters** | `.` | Matches any single character except newline (`\n`). |
| **Metacharacters** | `^` | Asserts the position at the start of the string. |
| **Metacharacters** | `$` | Asserts the position at the end of the string. |
| **Quantifiers** | `*` | Matches 0 or more occurrences of the preceding token. |
| **Quantifiers** | `+` | Matches 1 or more occurrences of the preceding token. |
| **Quantifiers** | `?` | Matches 0 or 1 occurrence of the preceding token (optional match). |
| **Quantifiers** | `{n,m}` | Matches between $n$ and $m$ repetitions of the preceding pattern. |
| **Character Classes** | `[A-Za-z]` | Matches any ASCII uppercase or lowercase alphabetical character. |
| **Character Classes** | `\d` | Matches any decimal digit (equivalent to `[0-9]`). |
| **Character Classes** | `\s` | Matches any whitespace character (space, tab, carriage return, newline). |
| **Character Classes** | `[^...]` | Inverted set: matches any character *not* enclosed inside brackets. |

---

### 1.2 Tokenization and Stop Word Removal
- **Tokenization:** Decomposing continuous character sequences into atomic tokens (words or punctuation) using linguistic boundary heuristics.
- **Stop Words:** High-frequency syntactic function words (e.g., `"the"`, `"is"`, `"at"`, `"which"`) carrying near-zero task-specific discriminative entropy.

```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

tokens = word_tokenize(cleaned_text.lower())
stop_words = set(stopwords.words('english'))
filtered_tokens = [w for w in tokens if w not in stop_words]
```

---

### 1.3 Stemming (Porter's Algorithm) vs. Lemmatization (WordNet)

```
Inflected Word: "running"
   ├── Stemmer  (Rule-based truncation) ──► "run"
   └── Lemmatizer (Morphological Lexicon) ─► "run" (POS='v')

Inflected Word: "better"
   ├── Stemmer  (Rule-based truncation) ──► "better"
   └── Lemmatizer (Morphological Lexicon) ─► "good" (POS='a')
```

#### Stemming vs. Lemmatization Comparison
| Characteristic | Stemming (Porter Stemmer) | Lemmatization (WordNet Lemmatizer) |
| :--- | :--- | :--- |
| **Core Method** | Rule-based heuristic suffix chopping | Morphological dictionary lookup & POS analysis |
| **Execution Cost**| Extremely low $O(1)$ string pattern rules | Higher computational cost $O(\log |V|)$ dictionary traversal |
| **Linguistic Validity** | Produces non-real word stems (e.g., `poni`, `gener`) | Always produces valid grammatical dictionary lemmas |
| **Context Sensitivity** | None | High (requires Part-of-Speech tag, e.g., `pos='v'`, `pos='n'`) |

#### Porter's Algorithm 5-Step Execution Rules
The algorithm measures the complexity of a word root via measure $m$, where a word is generalized as $[C](VC)^m[V]$ ($C$ = consonant sequence, $V$ = vowel sequence):
1. **Step 1a & 1b:** Handles plurals and participle endings:
   - `sses` $\to$ `ss` (*caresses* $\to$ *caress*)
   - `ies` $\to$ `i` (*ponies* $\to$ *poni*)
   - `eed` $\to$ `ee` if $m > 0$ (*agreed* $\to$ *agree*)
   - `ed` / `ing` are removed if preceded by a vowel (*walking* $\to$ *walk*)
2. **Step 1c:** Terminal `y` $\to$ `i` if preceded by another letter (*happy* $\to$ *happi*).
3. **Step 2:** Reduces compound suffixes if $m > 0$ (`ational` $\to$ `ate`, `izer` $\to$ `ize`, `alli` $\to$ `al`).
4. **Step 3:** Handles derivational suffixes if $m > 0$ (`icate` $\to$ `ic`, `ative` $\to$ $\emptyset$, `alize` $\to$ `al`).
5. **Step 4:** Strips standardized derivational endings if $m > 1$ (`-able`, `-ance`, `-ence`, `-ment`, `-ic`).
6. **Step 5a & 5b:** Clean-up phase: strips trailing `e` if $m > 1$ (*probate* $\to$ *probat*) and reduces double consonants (`ll` $\to$ `l` if $m > 1$).

---

### 1.4 Levenshtein Edit Distance & Spell Checking Engine
Quantifies the minimum character transformation operations (Insertion, Deletion, Substitution) required to transform string $A$ of length $m$ to string $B$ of length $n$.

#### Mathematical Recurrence Relation
$$D(i, j) = \begin{cases} 
\max(i, j) & \text{if } \min(i, j) = 0 \\
\min \begin{cases} 
D(i-1, j) + 1 & \text{(Deletion of } A[i]\text{)} \\
D(i, j-1) + 1 & \text{(Insertion of } B[j]\text{)} \\
D(i-1, j-1) + \text{cost} & \text{(Substitution)}
\end{cases} & \text{otherwise}
\end{cases}$$

$$\text{where } \text{cost} = \begin{cases} 0 & \text{if } A[i] = B[j] \\ 1 & \text{if } A[i] \neq B[j] \end{cases}$$

#### Dynamic Programming Implementation & Spell Checking
```python
import numpy as np

def calculate_edit_distance(word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if word1[i - 1] == word2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # Deletion
                dp[i][j - 1] + 1,        # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )
    return dp[m][n]

# Minimum Edit Distance Spell Correction Query
vocabulary = ["apple", "banana", "orange", "grape", "mango"]
misspelled = "bananna"

distances = {word: calculate_edit_distance(misspelled, word) for word in vocabulary}
suggested_word = min(distances, key=distances.get)
print(f"Query: '{misspelled}' -> Correction: '{suggested_word}' (Distance: {distances[suggested_word]})")
```

---

## 📊 Lab 02: Text Representation, N-Gram Modeling & Shannon's Game

Lab 02 transitions from character-level operations to numerical vector representations and statistical language modeling.

### 2.1 Vector Space Models: Bag of Words (BoW) & TF-IDF

#### 1. Bag of Words (BoW)
Represents a document as a sparse vector of word frequencies, discarding grammatical syntax and word order.
$$\mathbf{x}_d = [C(w_1, d), C(w_2, d), \dots, C(w_{|V|}, d)]$$

#### 2. Term Frequency-Inverse Document Frequency (TF-IDF)
Penalizes omnipresent words across the corpus while amplifying domain-specific keywords.
$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t', d}}$$

$$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = [
    "the quick brown fox jumps over the lazy dog",
    "never jump over lazy dogs",
    "the dog barked at the brown fox"
]

bow_vec = CountVectorizer()
X_bow = bow_vec.fit_transform(corpus)

tfidf_vec = TfidfVectorizer()
X_tfidf = tfidf_vec.fit_transform(corpus)
```

---

### 2.2 N-Gram Language Modeling & Markov Assumption

An **N-gram** is a contiguous sequence of $n$ tokens:
- $n=1$ (Unigram): `["fox"]`
- $n=2$ (Bigram): `["fox", "jumps"]`
- $n=3$ (Trigram): `["fox", "jumps", "over"]`

#### Joint Probability via the Chain Rule
$$P(w_1, w_2, \dots, w_m) = \prod_{k=1}^{m} P(w_k \mid w_1, w_2, \dots, w_{k-1})$$

#### $(N-1)^{\text{th}}$-Order Markov Approximation
Instead of conditioning on the infinite past history, an N-gram model approximates the probability by looking back only $n-1$ tokens:
$$P(w_k \mid w_1, \dots, w_{k-1}) \approx P(w_k \mid w_{k-n+1}, \dots, w_{k-1})$$

---

### 2.3 Maximum Likelihood Estimation (MLE) & Probability Tables

The MLE parameters are computed directly via count ratios:
$$P_{\text{MLE}}(w_k \mid w_{k-n+1}^{k-1}) = \frac{C(w_{k-n+1}^{k-1}, w_k)}{C(w_{k-n+1}^{k-1})} = \frac{C(w_{k-n+1}^{k})}{\sum_{w} C(w_{k-n+1}^{k-1}, w)}$$

#### Example Corpus:
1. `<s> I am Sam </s>`
2. `<s> Sam I am </s>`
3. `<s> I am free </s>`

#### Trigram MLE Probability Distribution Table
| History $(w_{i-2}, w_{i-1})$ | Target Word ($w_i$) | Trigram Count $C(w_{i-2}, w_{i-1}, w_i)$ | Bigram History Count $C(w_{i-2}, w_{i-1})$ | MLE Probability $P(w_i \mid w_{i-2}, w_{i-1})$ |
| :--- | :--- | :--- | :--- | :--- |
| `<s>, I` | `am` | 2 | 2 | $2/2 = 1.00$ |
| `<s>, Sam` | `I` | 1 | 1 | $1/1 = 1.00$ |
| `I, am` | `Sam` | 1 | 3 | $1/3 \approx 0.33$ |
| `I, am` | `free` | 1 | 3 | $1/3 \approx 0.33$ |
| `I, am` | `</s>` | 1 | 3 | $1/3 \approx 0.33$ |
| `am, Sam` | `</s>` | 1 | 1 | $1/1 = 1.00$ |
| `Sam, I` | `am` | 1 | 1 | $1/1 = 1.00$ |
| `am, free` | `</s>` | 1 | 1 | $1/1 = 1.00$ |

---

### 2.4 Shannon's Guessing Game (1951) & Autoregressive Text Generation

Claude Shannon formulated the guessing game to estimate the entropy of printed English by predicting subsequent tokens given antecedent context.

```python
from collections import defaultdict, Counter
import numpy as np

def build_ngram_model(tokens: list, n: int = 2) -> dict:
    raw_counts = defaultdict(Counter)
    for i in range(len(tokens) - n + 1):
        history = tuple(tokens[i : i + n - 1])
        target = tokens[i + n - 1]
        raw_counts[history][target] += 1
    
    # Compute normalized conditional probabilities
    prob_model = defaultdict(dict)
    for history, targets in raw_counts.items():
        total = sum(targets.values())
        for target, count in targets.items():
            prob_model[history][target] = count / total
    return prob_model

def shannons_predictor(history_phrase: str, model: dict, n: int = 3) -> list:
    tokens = history_phrase.lower().split()
    history_key = tuple(tokens[-(n - 1):])
    if history_key not in model:
        return []
    candidates = model[history_key]
    return sorted(candidates.items(), key=lambda x: x[1], reverse=True)

def generate_text(model: dict, seed_phrase: str, max_words: int = 20, n: int = 3) -> str:
    output_tokens = seed_phrase.lower().split()
    for _ in range(max_words):
        history_key = tuple(output_tokens[-(n - 1):])
        if history_key not in model:
            break
        candidates = list(model[history_key].keys())
        probs = list(model[history_key].values())
        next_word = np.random.choice(candidates, p=probs)
        output_tokens.append(next_word)
        if next_word == "</s>":
            break
    return " ".join(output_tokens)
```

---

## 🧠 Lab 03: Word Embeddings & Foundational Classifiers

Lab 03 covers dense representation learning (Word2Vec) and compares generative versus discriminative foundational text classification.

### 3.1 Continuous Word Embeddings (Word2Vec Skip-Gram from Scratch)

```
        One-Hot Vector x_c (1 x V)
                  │
                  ▼
       ┌─────────────────────┐
       │   Matrix W1 (V x D) │  ◄── Center Word Embedding Lookup: h = W1[c]
       └─────────────────────┘
                  │
                  ▼  Hidden Representation h (1 x D)
       ┌─────────────────────┐
       │   Matrix W2 (D x V) │  ◄── Projection to Vocabulary Scores: z = h * W2
       └─────────────────────┘
                  │
                  ▼
         Softmax Normalization  ──► Predicted Probabilities y_hat (1 x V)
                  │
                  ▼
        Cross-Entropy Loss with True Context One-Hot Vectors y_o
```

#### Forward Pass Mathematical Formulation
1. **Hidden Lookup:**
   $$\mathbf{h} = \mathbf{x}_c^T W_1 = W_1[c, :] \in \mathbb{R}^{1 \times D}$$
2. **Logit Projection:**
   $$\mathbf{z} = \mathbf{h} W_2 \in \mathbb{R}^{1 \times V}$$
3. **Numerically Stable Softmax:**
   $$\hat{y}_i = \frac{\exp(z_i - \max(\mathbf{z}))}{\sum_{j=1}^{V} \exp(z_j - \max(\mathbf{z}))}$$
4. **Cross-Entropy Loss (for true context word index $o$):**
   $$\mathcal{L} = -\log \hat{y}_o = -z_o + \log \sum_{j=1}^{V} \exp(z_j)$$

#### Gradient Derivations & SGD Weight Updates
$$\mathbf{e} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \hat{\mathbf{y}} - \mathbf{y}_{\text{true}} \in \mathbb{R}^{1 \times V}$$

$$\frac{\partial \mathcal{L}}{\partial W_2} = \mathbf{h}^T \mathbf{e} \in \mathbb{R}^{D \times V} \implies W_2^{(t+1)} = W_2^{(t)} - \eta (\mathbf{h}^T \mathbf{e})$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}} = \mathbf{e} W_2^T \in \mathbb{R}^{1 \times D} \implies W_1[c, :]^{(t+1)} = W_1[c, :]^{(t)} - \eta (\mathbf{e} W_2^T)$$

```python
# NumPy Vectorized Skip-Gram Step
def skip_gram_step(center_id, context_id, W1, W2, lr=0.01):
    # 1. Hidden lookup
    h = W1[center_id:center_id+1] # (1, D)
    
    # 2. Output logits & stable softmax
    z = np.dot(h, W2) # (1, V)
    z_stable = z - np.max(z)
    exp_z = np.exp(z_stable)
    y_pred = exp_z / np.sum(exp_z)
    
    # 3. Error vector
    e = y_pred.copy()
    e[0, context_id] -= 1.0 # (y_pred - y_true)
    
    # 4. Gradients
    dW2 = np.dot(h.T, e)         # (D, V)
    dW1_center = np.dot(e, W2.T) # (1, D)
    
    # 5. Weight updates
    W2 -= lr * dW2
    W1[center_id] -= lr * dW1_center.squeeze()
```

---

### 3.2 Generative Text Classification (Multinomial Naive Bayes)

Generative classifiers model the joint probability $P(d, c) = P(c) P(d \mid c)$ and apply Bayes' Rule:
$$\hat{c} = \arg\max_{c \in C} P(c \mid d) = \arg\max_{c \in C} P(c) \prod_{i=1}^{|d|} P(w_i \mid c)$$

#### Laplace (Add-1) Smoothing & Log-Space Arithmetic
To prevent zero-probability annihilation from out-of-vocabulary terms and numerical underflow:
$$\hat{P}(w \mid c) = \frac{C(w, c) + 1}{\sum_{w' \in V} C(w', c) + |V|}$$

$$\hat{c} = \arg\max_{c \in C} \left[ \ln P(c) + \sum_{i=1}^{|d|} \ln \hat{P}(w_i \mid c) \right]$$

---

### 3.3 Discriminative Text Classification (Logistic Regression with SGD)

Logistic Regression directly models the posterior $P(y=1 \mid \mathbf{x})$:
$$z = \mathbf{w}^T \mathbf{x} + b, \quad \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

#### Binary Cross-Entropy (Log Loss) & Vectorized Gradients
$$J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \ln \hat{y}^{(i)} + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right]$$

$$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m} X^T (\hat{\mathbf{y}} - \mathbf{y}), \quad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

---

### 3.4 Embedding Aggregation & The "Anagram Dilemma"

Converting variable-length documents into fixed-size feature vectors for linear classifiers:
1. **Unweighted Mean Pooling:**
   $$\mathbf{d}_{\text{mean}} = \frac{1}{|d|} \sum_{w \in d} \mathbf{v}_w$$
2. **TF-IDF Weighted Pooling:**
   $$\mathbf{d}_{\text{tfidf}} = \sum_{w \in d} \text{TF-IDF}(w, d) \cdot \mathbf{v}_w$$

#### Mathematical Proof of the Anagram Dilemma
Consider two sentences with inverted semantics:
- $S_A$: `"the dog bit the man"`
- $S_B$: `"the man bit the dog"`

Both sentences share the exact same multiset of tokens: $V_d = \{\text{"the"}: 2, \text{"dog"}: 1, \text{"bit"}: 1, \text{"man"}: 1\}$.
1. **Term Frequencies & IDF:** $\text{TF}(w, S_A) = \text{TF}(w, S_B)$ and $\text{IDF}(w, D)$ is identical for both.
2. **Vector Addition Commutativity:**
   $$\mathbf{d}_A = \mathbf{v}_{\text{the}} + \mathbf{v}_{\text{dog}} + \mathbf{v}_{\text{bit}} + \mathbf{v}_{\text{the}} + \mathbf{v}_{\text{man}} = \mathbf{v}_{\text{the}} + \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{bit}} + \mathbf{v}_{\text{the}} + \mathbf{v}_{\text{dog}} = \mathbf{d}_B$$

$$\therefore f_{\text{classifier}}(\mathbf{d}_A) \equiv f_{\text{classifier}}(\mathbf{d}_B)$$

**Conclusion:** Neither Mean nor TF-IDF Weighted Pooling can distinguish between sentences with identical word counts but opposing sequential syntax. This necessitates sequence-aware recurrent or attention-based architectures.

---

### 3.5 Foundational Paradigm Comparison Matrix

| Property | Multinomial Naive Bayes | Logistic Regression | Word2Vec (Skip-Gram) |
| :--- | :--- | :--- | :--- |
| **Learning Paradigm** | Generative | Discriminative | Self-Supervised / Representation |
| **Loss Function** | Closed-form joint likelihood | Binary Cross-Entropy (Log Loss) | Categorical Cross-Entropy / NCE |
| **Primary Input Format** | Sparse integer count vectors | Sparse BoW or Dense Aggregated Vectors | Categorical token indices (One-Hot) |
| **Computational Complexity**| $O(N \cdot |V|)$ single pass | $O(\text{epochs} \cdot m \cdot D)$ iterative SGD | $O(\text{epochs} \cdot T \cdot C \cdot D \cdot |V|)$ |
| **Handling Unseen Words** | Laplace (Add-1) smoothing | Handled via zero feature values | OOV vector mapping / Random init |
| **Preserves Word Order?** | ❌ No | ❌ No (under BoW/Averaging) | ⚠️ Local context window only ($c \pm k$) |

---

## 🔁 Lab 04: PyTorch Sequence Models (RNN, LSTM, BiLSTM, Seq2Seq)

Lab 04 implements PyTorch neural sequence architectures that preserve word order and temporal dependencies.

### 4.1 Pretrained Google News Word2Vec Embedding Matrix
Instead of randomly initializing token vectors, weights are extracted from Google's official 3-million word 300-dimensional Word2Vec model:

```python
import gensim.downloader as api
import torch
import torch.nn as nn

w2v_google = api.load('word2vec-google-news-300')

def build_pretrained_embedding_matrix(corpus, w2v_model):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for sent in corpus:
        for word in sent:
            if word not in vocab:
                vocab[word] = len(vocab)
                
    weights = torch.zeros((len(vocab), 300), dtype=torch.float32)
    for word, idx in vocab.items():
        if word in w2v_model:
            weights[idx] = torch.tensor(w2v_model[word])
        elif word != "<PAD>":
            weights[idx] = torch.randn(300) * 0.1
            
    embedding_layer = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=0)
    return vocab, embedding_layer
```

---

### 4.2 Sequence Classification: Vanilla RNN vs. Stacked BiLSTM vs. Mean-Pooled BiLSTM

#### 1. Stacked Bidirectional LSTM Classifier
Captures forward context $\overrightarrow{h}_t$ and backward context $\overleftarrow{h}_t$ across 2 stacked recurrent layers.

```python
class StackedBiLSTMClassifier(nn.Module):
    def __init__(self, embed_matrix, hidden_dim, num_classes=1):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embed_matrix, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_matrix.shape[1],
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embeds = self.embedding(x)                 # (B, T, D)
        _, (h_n, _) = self.lstm(embeds)            # h_n: (num_layers * 2, B, hidden_dim)
        # Extract forward and backward hidden states from the top layer
        h_forward = h_n[-2, :, :]                  # (B, hidden_dim)
        h_backward = h_n[-1, :, :]                 # (B, hidden_dim)
        h_cat = torch.cat((h_forward, h_backward), dim=1) # (B, hidden_dim * 2)
        logits = self.fc(h_cat).squeeze(-1)        # (B,)
        return logits
```

#### 2. Mean-Pooled BiLSTM Classifier (Task B2)
Instead of extracting only the final boundary hidden slice ($h_n$), Mean-Pooling averages all token-level representations across the sequence dimension $T$, capturing information uniformly from all words.

```python
class MeanPooledBiLSTMClassifier(nn.Module):
    def __init__(self, embed_matrix, hidden_dim, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embed_matrix, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_matrix.shape[1],
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embeds = self.embedding(x)                 # (B, T, D)
        lstm_out, _ = self.lstm(embeds)            # (B, T, hidden_dim * 2)
        # Average across the sequence length dimension (T)
        pooled = torch.mean(lstm_out, dim=1)       # (B, hidden_dim * 2)
        logits = self.fc(pooled)                   # (B, num_classes)
        return logits
```

---

### 4.3 Sequence Labeling: POS Tagging with Bidirectional RNN & Loss Masking
In sequence labeling (Part-of-Speech tagging), the model outputs predictions for every token in the sequence. To handle batch padding without corrupting gradients, `ignore_index=0` is applied in `CrossEntropyLoss`.

```python
class BiRNNSequenceLabeler(nn.Module):
    def __init__(self, embed_matrix, hidden_dim, tagset_size):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embed_matrix, freeze=False, padding_idx=0)
        self.rnn = nn.RNN(embed_matrix.shape[1], hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)
        
    def forward(self, x):
        embeds = self.embedding(x)      # (B, T, D)
        out, _ = self.rnn(embeds)       # (B, T, hidden_dim * 2)
        logits = self.fc(out)           # (B, T, tagset_size)
        return logits

# Training Loss with Padding Masking
criterion = nn.CrossEntropyLoss(ignore_index=0)
# Flattening (B, T, C) -> (B*T, C) and target (B, T) -> (B*T)
loss = criterion(logits.view(-1, tagset_size), targets.view(-1))
```

---

### 4.4 Language Modeling & Autoregressive Text Generation with Temperature
Trained on the Tiny Shakespeare corpus using sliding-window token pairs $(w_t, \dots, w_{t+K-1}) \to (w_{t+1}, \dots, w_{t+K})$.

#### Temperature-Scaled Softmax Sampling Formula
$$P(w_i) = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$
- $\tau \to 0$: Argmax / Greedy selection (conservative, repetitive).
- $\tau = 1.0$: Standard categorical distribution sampling.
- $\tau > 1.0$: Flattened distribution (diverse, creative, higher entropy).

```python
def generate_shakespeare(model, seed_idx, idx2word, length=50, temperature=0.8):
    model.eval()
    input_tensor = torch.tensor([[seed_idx]], dtype=torch.long).to(device)
    hidden = None
    generated = [idx2word[seed_idx]]
    
    with torch.no_grad():
        for _ in range(length):
            logits, hidden = model(input_tensor, hidden)
            # Scale last token logits by temperature
            scaled_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            generated.append(idx2word.get(next_idx, "<UNK>"))
            input_tensor = torch.tensor([[next_idx]], dtype=torch.long).to(device)
    return " ".join(generated)
```

---

### 4.5 Seq2Seq Machine Translation (English to Bengali) with Teacher Forcing

```
 ENCODER (English)                           DECODER (Bengali)
 "i love my country"                         <SOS> -> "আমি" -> "আমার" -> "দেশকে" -> "ভালোবাসি" -> <EOS>
         │                                     ▲        │        │         │            │
 ┌───────────────┐                             │        │        │         │            │
 │ Encoder LSTM  │ ──► [Hidden, Cell] ─────────┴────────┴────────┴─────────┴────────────┘
 └───────────────┘     Context Vectors
```

#### Teacher Forcing Mechanism
During training, instead of feeding the model's own (potentially flawed) output $\hat{y}_{t-1}$ to step $t$, the ground-truth target token $y_{t-1}$ is fed directly:

```python
class Seq2SeqTranslation(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)
        
        # 1. Encode source sentence into final context vectors
        hidden, cell = self.encoder(src)
        
        # 2. First decoder input is <SOS> token
        decoder_input = trg[:, 0].unsqueeze(1)
        
        # 3. Teacher forcing loop
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t, :] = output.squeeze(1)
            # Feed ground-truth token at step t
            decoder_input = trg[:, t].unsqueeze(1)
            
        return outputs
```

---

### 4.6 PyTorch Sequence Tensor Shapes & Dimension Reference

| Component / Task | Operation / Layer | Input Tensor Shape | Output Tensor Shape | Mathematical Transformation |
| :--- | :--- | :--- | :--- | :--- |
| **Embedding** | `nn.Embedding` | $(B, T)$ | $(B, T, D)$ | Integer ID lookup $\to$ dense vector |
| **Vanilla RNN** | `nn.RNN(batch_first=True)` | $(B, T, D)$ | $(B, T, H)$, $h_n: (1, B, H)$ | $h_t = \tanh(W_{ih} x_t + W_{hh} h_{t-1} + b)$ |
| **Stacked BiLSTM** | `nn.LSTM(num_layers=2, bidir=True)`| $(B, T, D)$ | $(B, T, 2H)$, $h_n: (4, B, H)$ | Dual-directional 2-layer recurrence |
| **Classification Head**| `torch.cat([h_fwd, h_bwd])` | $2 \times (B, H)$ | $(B, 2H) \to (B, 1)$ | Linear mapping to binary logit |
| **POS Tagging** | `BiRNN` + Linear | $(B, T, D)$ | $(B, T, C)$ | Per-token tagset score projection |
| **Language Model** | `StackedLSTM` + Linear | $(B, T, D)$ | $(B, T, |V|)$ | Vocabulary projection across sequence |
| **Seq2Seq Encoder** | `Encoder.LSTM` | $(B, T_{\text{src}}, D)$ | $(h_n, c_n): (1, B, H)$ | Full sequence compressed into context vectors |
| **Seq2Seq Decoder** | `Decoder.LSTM` | $(B, 1, D_{\text{trg}}), (h, c)$ | $(B, 1, |V_{\text{bn}}|), (h', c')$ | Step-by-step target token generation |

---

## ⚡ Lab 05: Transformer Architecture (Encoder-Only Classifier)

Lab 05 implements an **Encoder-Only Transformer** (BERT-style architecture) from scratch using PyTorch primitives, evaluating it on sentiment classification.

### 5.1 Encoder-Only Transformer Architecture Overview

```
                          Input Token Indices (B, T)
                                      │
                                      ▼
                           Token Embeddings (B, T, D)
                                      │
                                      ▼  Scaled by sqrt(d_model)
                         + Positional Encoding (1, T, D)
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │   TransformerEncoderLayer 1   │ ◄── Padding Mask: (B, T)
                      │  • Multi-Head Self-Attention  │
                      │  • Add & LayerNorm            │
                      │  • Feed-Forward (d_ff=256)    │
                      │  • Add & LayerNorm            │
                      └───────────────────────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │   TransformerEncoderLayer 2   │
                      └───────────────────────────────┘
                                      │
                                      ▼
                          Contextual States (B, T, D)
                                      │
                                      ▼
                           Mean Pooling: mean(dim=1)
                                      │
                                      ▼
                         Sentence Representation (B, D)
                                      │
                                      ▼
                         Linear Classifier Head (B, 1)
```

---

### 5.2 Sinusoidal Positional Encoding & Buffer Registration
Because self-attention operations are permutation-invariant, positional signatures are injected via sinusoidal waves of varying frequencies:

$$\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)$$

$$\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)$$

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        
        # register_buffer persists pe with model state without computing gradients
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Broadcast addition across batch dimension
        return x + self.pe[:, :x.size(1), :]
```

---

### 5.3 Multi-Head Attention, Feed-Forward Sublayers & Padding Masking

#### Scaled Dot-Product Attention Formula
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M_{\text{padding}}\right) V$$

```python
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=2, num_classes=1, max_len=8):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x, padding_mask=None):
        # 1. Scaled embedding + positional encoding injection
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # 2. Multi-layer Transformer Encoder with key padding mask
        encoded = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # 3. Sequence Mean Pooling over active tokens
        pooled = encoded.mean(dim=1) # (B, D)
        
        # 4. Classification Logit
        logits = self.classifier(pooled).squeeze(-1) # (B,)
        return logits
```

---

### 5.4 Sequence Mean Pooling & Classification Head
- **Padding Mask Construction:** `padding_mask = (inputs == 0)` creates a boolean mask where `True` indicates `<PAD>` tokens that must receive $-\infty$ attention logits.
- **Mean Pooling:** Compresses sequence dimension $T \to 1$ by computing $\frac{1}{T} \sum_{t=1}^T \mathbf{h}_t \in \mathbb{R}^{B \times D}$.
- **Loss Computation:** `nn.BCEWithLogitsLoss()` directly processes unnormalized logits for numerical stability.

---

### 5.5 Transformer Tensor Shape Lifecycle & Data Flow Matrix

| Stage | Operation / Call | Input Tensor Shape | Output Tensor Shape | Purpose & Mathematical Context |
| :--- | :--- | :--- | :--- | :--- |
| **1. Input Tokens** | `inputs` | `(32, 8)` | `(32, 8)` | Batch of 32 sequences padded to max length $T=8$. |
| **2. Padding Mask** | `(inputs == 0)` | `(32, 8)` | `(32, 8)` [Bool] | Identifies `<PAD>` positions for attention masking. |
| **3. Embedding** | `self.embedding(x)` | `(32, 8)` | `(32, 8, 64)` | Maps token IDs to $D=64$ dense continuous embeddings. |
| **4. Scale & Pos** | `x * sqrt(D) + PE` | `(32, 8, 64)` | `(32, 8, 64)` | Scales amplitudes and adds sinusoidal positional signals. |
| **5. Encoder** | `TransformerEncoder` | `(32, 8, 64)` | `(32, 8, 64)` | Computes Multi-Head Self-Attention ($h=2$) and Feed-Forward. |
| **6. Mean Pooling** | `encoded.mean(dim=1)` | `(32, 8, 64)` | `(32, 64)` | Collapses temporal dimension $T$ into sentence vector. |
| **7. Classifier** | `self.classifier(pooled)`| `(32, 64)` | `(32, 1)` | Computes raw unnormalized binary classification logit. |
| **8. Squeeze** | `logits.squeeze(-1)` | `(32, 1)` | `(32,)` | Aligns output shape with ground truth target shape `y`. |

---

## 🛠️ Environment Setup & Installation

### 1. Initialize Virtual Environment
```bash
# Navigate to workspace
cd "/home/shuvo/4-1/_NLP Lab"

# Create and activate virtual environment
python3 -m venv .nlp_venv
source .nlp_venv/bin/activate
```

### 2. Install Core Dependencies
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install gensim nltk scikit-learn numpy pandas python-docx jupyter
```

### 3. Download Required NLTK Corpora
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### 4. Launch Jupyter Notebook Server
```bash
jupyter notebook --port=8888 --no-browser
```
