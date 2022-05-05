from sklearn import metrics
import re
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import pickle
from os.path import exists
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer


from paths_w2v import WEB_DATA_EMBEDDINGS_PATH, WEB_DATA_PATH


class TextRetrieverW2V:

    def __init__(self) -> None:
        self.doc_embeddings_directory = WEB_DATA_EMBEDDINGS_PATH
        self.doc_directory = WEB_DATA_PATH
        self.tokenizer = RegexpTokenizer(' ', gaps=True)
        
        # Training the W2V Model
        document_list = []
        paths = []
        self.document_embeddings = {}
        for subdirectory in os.listdir(self.doc_directory):
            if os.path.isfile(self.doc_directory + subdirectory):
                continue
            for file in os.listdir(self.doc_directory + subdirectory):
                with open(self.doc_directory + subdirectory + '/' + file) as f:
                    content = f.read()
                    temp_data = []
                    for token in self.tokenizer.tokenize(content):
                        temp_data.append(token)
                    document_list.append(temp_data)
                    paths.append(self.doc_directory + subdirectory + '/' + file)
        
        print("Completed Preprocessing")
        self.model = Word2Vec(document_list, size=10, window=5, min_count=1, workers=5, sg=1)
        print("Initialised model")
        transformed_vecs = self.fit_data_to_model(document_list)
        print("Transformed Vectors")
        self.documents_df = pd.DataFrame({'Path': paths,
        'Embedding': transformed_vecs
        })
        print("Initialised Dataframe")
        self.documents_df.to_pickle(WEB_DATA_EMBEDDINGS_PATH + 'w2v_embeddings.pkl')
        with open(WEB_DATA_EMBEDDINGS_PATH + 'w2v_model.pkl', 'wb') as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Wrote to pickle files")

    def fit_data_to_model(self, document_list):
        fitted_documents = []
        for document in document_list:
            temp_data = []
            for word in document:
                if word in self.model.wv:
                    temp_data.append(np.mean(self.model.wv[word]))		
            fitted_documents.append(temp_data)
        return fitted_documents

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
        embedded_text = self.fit_data_to_model([text])
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
            mod_row_embedding = row['Embedding']
            mod_query_embedding = query_vector[0]
            if len(query_vector[0]) > len(row['Embedding']):
                mod_row_embedding += [0] * (len(query_vector[0])  - len(row['Embedding']))
            elif len(query_vector[0]) < len(row['Embedding']):
                mod_query_embedding += [0] * (len(row['Embedding'])  - len(query_vector[0]))
            scores.append(self.compute_similarity_score([mod_query_embedding], [mod_row_embedding])[0])
        """
        print(self.documents_df['Embedding'].tolist())
        scores = self.compute_similarity_score(query_vector, [self.documents_df['Embedding'].tolist()[0]])[0]
        """
        """
        for index, row in self.documents_df.iterrows():
            scores.append(self.compute_similarity_score([query_vector], row['Embedding'])[0])
        """
        new_document_df = self.documents_df.copy()
        new_document_df['Score'] = scores
        new_document_df = new_document_df.sort_values(by='Score', ascending=False)
        retrieved_docs = new_document_df[:30]['Path'].values
        retrieved_scores = new_document_df[:30]['Score'].values

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
        
        final_docs = pd.DataFrame({'Text': docs_subset[:num_docs]['Content'].values, 'URL': docs_subset[:num_docs]['URL'].values, 'Similarity Scores': docs_subset[:num_docs]['Score'].values})
        print("Similarity: " + str(docs_subset[:num_docs]['Score'].values[0]))
        return final_docs

"""
retriever_object = TextRetrieverW2V()
"""