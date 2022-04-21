from sentence_transformers import SentenceTransformer
from sklearn import metrics
import re
import os


class TextRetriever:

    def __init__(self) -> None:
        pass

    def preprocess_text(self, text):
        # Eliminating URLs
        text = re.sub(r'http\S+', '', text)
        # Eliminating non-alphabetical characters
        non_alpha_chars = re.compile('[^A-Za-z]')
        processed_text = re.sub('  ', ' ', non_alpha_chars.sub(' ', text))
        # Removing any extra spaces and converting into lower case
        return re.sub('\s+',' ', processed_text).lower()

    def compute_similarity_score(self, text_1, text_2):
        return metrics.pairwise.cosine_similarity(text_1, text_2)
    
    def get_vector_representation(self, text):
        bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
        embedded_text = bert_model.encode(text)
        return embedded_text

"""
retriever_object = TextRetriever()
print(retriever_object.preprocess_text('masks are useful for preventing covid-19'))

rep1 = retriever_object.get_vector_representation('covid-19 mask mandate')
rep2 = retriever_object.get_vector_representation('masks are useful for preventing covid-19')
rep3 = retriever_object.get_vector_representation('the treasure is located in the depths of the ocean')
print(retriever_object.compute_similarity_score([rep1], [rep2, rep3]))

retriever_object('covid-19 mask mandate', 'masks are useful for preventing covid-19')
"""