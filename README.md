# qa-bot

## text_retrievers

This folder contains the Python classes for different types of text retrievers.
- text_retriever_tfidf.py: This class uses TF-IDF vectorization with cosine similarity to find most relevant documents for a query.
- text_retriever_sbert.py: This class uses SEBRT (Sentence BERT) with cosine similarity to find most relevant documents for a query.
- text_retriever_spacy.py: This class uses Spacy sentence similarity to find most relevant documents for a query.
Any of the above text retriever classes can be used for retrieving relevant documents. To evaluate a particular method, 