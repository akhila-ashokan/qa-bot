from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from sklearn import metrics
import re
import os
import pandas as pd
import numpy as np

from paths.paths_sbert_reranking import WEB_DATA_EMBEDDINGS_PATH, WEB_DATA_PATH, PREPROCESSED_DATA_PATH


class TextRetrieverSBERTReranking:
    """
    Implements retreival of documents based on similarity of SBERT embeddings of query and the documents followed by re-ranking using cross-encoders.
    Uses Python's Sentence-Transformers framework to generate the embeddings using a pre-trained model. 
    Cross-encoder is also based on a pre-trained model from the Sentence-Transformers framework. 
    """
    def __init__(self) -> None:
        """
        Loads pre-processed SBERT embeddings for documents
        """
        self.doc_embeddings_directory = WEB_DATA_EMBEDDINGS_PATH
        self.doc_directory = WEB_DATA_PATH
        self.preprocessed_data_directory = PREPROCESSED_DATA_PATH
        self.document_embeddings = []
        self.document_content = []
        self.document_paths = []

        for subdirectory in os.listdir(self.doc_embeddings_directory):
            if os.path.isfile(self.doc_embeddings_directory + subdirectory):
                continue
            for file in os.listdir(self.doc_embeddings_directory + subdirectory):
                key = self.doc_directory + subdirectory + '/' + file
                self.document_paths.append(key)
                self.document_embeddings.append(np.load(self.doc_embeddings_directory + subdirectory + '/' + file))
                with open(self.preprocessed_data_directory + subdirectory + '/' + file[:-3] + 'txt') as f:
                    self.document_content.append(f.read())

        self.documents_df = pd.DataFrame({'Path': self.document_paths, 'Embedding': self.document_embeddings, 'Content': self.document_content})
        # Loading Cross Encoder for Re-ranking
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-4-v2', max_length=510, num_labels=1)

    def preprocess_text(self, text):
        """
        Preprocesses text including removal of URLs, non-alphabetical characters, extra spaces, and stopwords.

        Keyword arguments:
        text -- the text to be preprocessed
        """
        # Eliminating URLs
        text = re.sub(r'http\S+', '', text)
        # Eliminating non-alphabetical characters
        non_alpha_chars = re.compile('[^A-Za-z]')
        processed_text = re.sub('  ', ' ', non_alpha_chars.sub(' ', text))
        # Removing any extra spaces and converting into lower case
        processed_text = re.sub('\s+',' ', processed_text).lower()
        processed_text = processed_text.split()
        processed_text = [self.lemmatizer.lemmatize(word) for word in processed_text if not word in set(stopwords.words())]
        processed_text = ' '.join(processed_text)
        return processed_text

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
        Gets the SBERT vector representation (embedding) for text.

        Keyword arguments:
        text -- the text to be embedded
        """
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        sbert_model.max_seq_length = 512
        embedded_text = sbert_model.encode(text)
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
        new_document_df = new_document_df[:30]
        
        # Performing re-ranking here with cross-encoder
        list_of_pairs = []
        for index, row in new_document_df.iterrows():
            list_of_pairs.append((query, row['Content']))

        reranker_scores = self.reranker.predict(list_of_pairs)
        
        new_document_df['Reranked Score'] = reranker_scores
        new_document_df = new_document_df.sort_values(by='Reranked Score', ascending=False)

        retrieved_docs = new_document_df[:num_docs]['Path'].values

        retrieved_docs_content = []
        urls = []

        for document in retrieved_docs:
            read_file = document[:-3] + 'txt'
            file_reader = open(read_file, 'r', encoding='utf-8')
            file_lines = file_reader.readlines()
            url = file_lines[0]
            retrieved_docs_content.append(file_lines)
            urls.append(url)
            file_reader.close()

        final_docs = pd.DataFrame({'Text': retrieved_docs_content, 'URL': urls, 'Similarity Scores': new_document_df[:num_docs]['Reranked Score'].values})

        return final_docs
