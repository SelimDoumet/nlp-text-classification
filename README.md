# NLP Text Classification - 20 Newsgroups

A complete, from-scratch NLP pipeline for multi-class topic classification on ~3,750 posts from the 20 Newsgroups dataset, spanning four topics: hockey, baseball, medicine, and space.

**Best model: Logistic Regression on TF-IDF features - 92.8% accuracy, 92.7% macro F1** on held-out test data.

## Pipeline

1. **Preprocessing** — tokenization, lowercasing, stopword removal, and stemming with NLTK
2. **Feature engineering** — Bag-of-Words vs. TF-IDF vectorization (compared directly)
3. **Modeling** — Multinomial Naive Bayes and Logistic Regression, evaluated head-to-head
4. **Hyperparameter tuning** — `GridSearchCV` over regularization strength for Logistic Regression
5. **Interpretability** — feature-importance analysis on the trained classifier, plus manual error analysis on misclassified examples
6. **Extra analysis**:
   - Part-of-speech tagging with spaCy to compare grammatical patterns across topics (e.g. sports vs. science)
   - Unsupervised topic modeling with Latent Dirichlet Allocation (LDA), used to independently validate that the dataset's topical structure matches the four labeled categories
7. **Ethical considerations** — a short audit of the dataset's limitations (temporal/demographic specificity to 1990s English-speaking Usenet users, no demographic metadata for fairness analysis)

## Key findings

- TF-IDF outperforms raw Bag-of-Words by ~4 points in accuracy, confirming the value of inverse-document-frequency weighting for this task
- Logistic Regression outperforms Naive Bayes by ~5 points — consistent with discriminative models generally beating generative ones when training data is plentiful
- Tuned regularization (C = 5.0) adds a further 0.5 points over the default
- LDA topic modeling independently recovers the same four topics as the ground-truth labels, without ever seeing them — a useful sanity check that the classification task is well-posed

## Reproducing this

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Then open `nlp_sentiment_analysis.ipynb` and run all cells top to bottom. The dataset is fetched automatically via `sklearn.datasets.fetch_20newsgroups` — no manual download needed.

## Tech stack

Python, scikit-learn, NLTK, spaCy, gensim (LDA), pandas, matplotlib/seaborn, wordcloud

## Author

Selim Doumet BSc Mathematics, Data Science option, Université Saint-Joseph de Beyrouth
