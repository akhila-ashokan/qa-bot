from sentence_transformers import SentenceTransformer
from sklearn import metrics
import re
import os
import pandas as pd
import numpy as np

from paths import WEB_DATA_EMBEDDINGS_PATH, WEB_DATA_PATH


class TextRetriever:

    def __init__(self) -> None:
        """
        Loads pre-processed BERT embeddings for documents
        """
        self.doc_embeddings_directory = WEB_DATA_EMBEDDINGS_PATH
        self.doc_directory = WEB_DATA_PATH
        self.document_embeddings = {}
        for subdirectory in os.listdir(self.doc_embeddings_directory):
            if os.path.isfile(self.doc_embeddings_directory + subdirectory):
                continue
            for file in os.listdir(self.doc_embeddings_directory + subdirectory):
                key = self.doc_directory + subdirectory + '/' + file
                self.document_embeddings[key] = np.load(self.doc_embeddings_directory + subdirectory + '/' + file)

        self.documents_df = pd.DataFrame(self.document_embeddings.items(), columns=['Path', 'Embedding'])

    def preprocess_text(self, text):
        """
        Preprocesses text including removal of URLs, non-alphabetical characters and extra spaces.

        Keyword arguments:
        text -- the text to be preprocessed
        """
        # Eliminating URLs
        text = re.sub(r'http\S+', '', text)
        # Eliminating non-alphabetical characters
        non_alpha_chars = re.compile('[^A-Za-z]')
        processed_text = re.sub('  ', ' ', non_alpha_chars.sub(' ', text))
        # Removing any extra spaces and converting into lower case
        return re.sub('\s+',' ', processed_text).lower()

    def compute_similarity_score(self, text_1, text_2):
        """
        Computes pairwise similarity score between two arrays of embeddings

        Keyword arguments:
        text_1 -- First array of word embeddings
        text_2 -- Second array of word embeddings
        """
        return metrics.pairwise.cosine_similarity(text_1, text_2)
    
    def get_vector_representation(self, text):
        """
        Gets the BERT vector representation (embedding) for text.

        Keyword arguments:
        text -- the text to be embedded
        """
        bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
        embedded_text = bert_model.encode(text)
        return embedded_text

    def get_highest_matching_docs(self, query, num_docs):
        """
        Gets the highest matching documents for a query

        Keyword arguments:
        query -- query to be processed
        num_docs -- number of matching documents to be returned
        """
        query_vector = self.get_vector_representation(query)
        similarity_scores = self.compute_similarity_score([query_vector], self.documents_df['Embedding'].tolist())
        new_document_df = self.documents_df.copy()
        new_document_df['Similarity Scores'] = similarity_scores[0]
        new_document_df = new_document_df.sort_values(by='Similarity Scores', ascending=False)
        retrieved_docs = new_document_df[:num_docs]['Path'].values

        retrieved_docs_content = []

        for document in retrieved_docs:
            read_file = document[:-3] + 'txt'
            file_reader = open(read_file, 'r')
            web_content = file_reader.read()
            retrieved_docs_content.append(web_content)

        final_docs = pd.DataFrame({'Text': retrieved_docs_content, 'Similarity Scores': new_document_df[:num_docs]['Similarity Scores'].values})
        return final_docs

"""
#Example Use:
retriever_object = TextRetriever()
print(retriever_object.get_highest_matching_docs('masks are useful for preventing covid-19', 5))
"""