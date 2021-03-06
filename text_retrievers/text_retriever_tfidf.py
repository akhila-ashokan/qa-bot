from sklearn import metrics
import re
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from os.path import exists
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from paths.paths_tfidf import WEB_DATA_PATH, PREPROCESSED_DATA_PATH

class TextRetrieverTFIDF:
    """
    Implements retreival of documents based on similarity of TF-IDF vectors for query and the documents.
    Uses Python's TF-IDF Vectorizer to generate the TF-IDF representations.
    """
    def __init__(self) -> None:
        # Loading path for web data documents and the pre-processed version of the documents
        self.doc_directory = WEB_DATA_PATH
        self.preprocessed_data_directory = PREPROCESSED_DATA_PATH
        
        # Loading lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Training the TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(max_features=100000,ngram_range=(1, 3))
        document_list = []
        paths = []
        self.document_embeddings = {}
        
        for subdirectory in os.listdir(self.doc_directory):
            if os.path.isfile(self.doc_directory + subdirectory):
                continue
            for file in os.listdir(self.doc_directory + subdirectory):
                # Loading the content of the document from pre-processed data directory instead of the original web data directory
                with open(self.preprocessed_data_directory + subdirectory + '/' + file) as f:
                    content = f.read()
                    document_list.append(content)
                    paths.append(self.doc_directory + subdirectory + '/' + file)
        
        self.vectorizer.fit(document_list)
        transformed_vecs = self.vectorizer.transform(document_list)
        self.documents_df = pd.DataFrame({'Path': paths,
        'Embedding': transformed_vecs
        })

    def preprocess_text(self, text):
        """
        Preprocesses text including removal of URLs, non-alphabetical characters, extra spaces and stopwords.
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
        Gets the vector representation (embedding) for text.
        Keyword arguments:
        text -- the text to be embedded
        """
        embedded_text = self.vectorizer.transform([text])
        return embedded_text

    def get_highest_matching_docs(self, query, num_docs):
        """
        Gets the highest matching documents for a query
        Keyword arguments:
        query -- query to be processed
        num_docs -- number of matching documents to be returned
        """
        query_vector = self.get_vector_representation(query)
        scores = []
        for index, row in self.documents_df.iterrows():
            scores.append(self.compute_similarity_score(query_vector, row['Embedding'])[0])
        new_document_df = self.documents_df.copy()
        new_document_df['Score'] = scores
        new_document_df = new_document_df.sort_values(by='Score', ascending=False)
        retrieved_docs = new_document_df[:num_docs]['Path'].values
        retrieved_scores = new_document_df[:num_docs]['Score'].values

        retrieved_docs_content = []
        urls = []

        for document in retrieved_docs:
            read_file = document[:-3] + 'txt'
            file_reader = open(read_file, 'r', encoding='utf-8')
            content = file_reader.readlines()
            url = content[0]
            content = ' '.join(content)
            retrieved_docs_content.append(content.strip())
            urls.append(url)
            file_reader.close()

        docs_subset = pd.DataFrame({'Path': retrieved_docs, 'Content': retrieved_docs_content, 'URL': urls, 'Score': retrieved_scores})
        
        final_docs = pd.DataFrame({'Text': docs_subset['Content'].values, 'URL': docs_subset['URL'].values, 'Similarity Scores': docs_subset['Score'].values})
        print("Similarity: " + str(docs_subset[:num_docs]['Score'].values[0]))
        return final_docs
