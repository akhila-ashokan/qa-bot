from sentence_transformers import SentenceTransformer, util
from sklearn import metrics
import re
import os
import pandas as pd
import numpy as np
import torch
import spacy
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import pickle
from os.path import exists



from paths_sbert_long import WEB_DATA_EMBEDDINGS_PATH, WEB_DATA_PATH


def dialogue_act_features(post):
    """
    This function is directly taken from https://www.nltk.org/book/ch06.html.
    """
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features

class TextRetrieverSBERTLong:

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

        # This section is to train a classifier that can identify posts that are questions
        # The code is take from https://www.nltk.org/book/ch06.html
        #nltk.download('nps_chat')
        """
        if exists('question_model.sav'):
            print("Model already exists!")
            self.question_model = pickle.load(open('question_model.sav', 'rb'))
            vectorizer = TfidfVectorizer(analyzer='word')
            example = vectorizer.fit_transform(['how do i enroll for classes if i did not register on time. it was a mistake from my side'])
            pred = self.question_model.predict(example)
            print(pred)
        else:
            nps_chat_posts = nltk.corpus.nps_chat.xml_posts()[:10000]
            features = [post.text for post in nps_chat_posts]
            size = int(len(features) * 0.1)
            train_set, test_set = features[size:], features[:size]
            vectorizer = TfidfVectorizer(analyzer='word')
            x_train = vectorizer.fit_transform(train_set)
            x_test = vectorizer.transform(test_set)

            y = [post.get('class') for post in nps_chat_posts]
            y_train, y_test = y[size:], y[:size]
            self.question_model = GradientBoostingClassifier(n_estimators = 400, random_state=0)
            self.question_model.fit(x_train, y_train)
            predictions = self.question_model.predict(x_test)
            print(classification_report(y_test, predictions))
            pickle.dump(self.question_model, open('question_model.sav', 'wb'))
        """

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
        bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Increasing maximum sequence length
        bert_model.max_seq_length = 512
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
        new_document_df['Score'] = similarity_scores[0]
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
            retrieved_docs_content.append(self.preprocess_text(content.strip()))
            urls.append(url)
            file_reader.close()

        docs_subset = pd.DataFrame({'Path': retrieved_docs, 'Content': retrieved_docs_content, 'URL': urls, 'Score': retrieved_scores})
        
        final_docs = pd.DataFrame({'Text': docs_subset[:num_docs]['Content'].values, 'URL': docs_subset[:num_docs]['URL'].values, 'Similarity Scores': docs_subset[:num_docs]['Score'].values})
        print("Similarity: " + str(docs_subset[:num_docs]['Score'].values[0]))
        return final_docs

"""
#Example Use:
retriever_object = TextRetrieverSBERTLong()
print(retriever_object.get_highest_matching_docs('masks are useful for preventing covid-19', 5))


retriever_object = TextRetrieverSBERTLong()
#highest_matching_docs = retriever_object.get_highest_matching_docs('Quick question about community college transfer class. Incoming ECE freshman here. So I\'m taking a replacement for physics 211 at a local community college during the fall. If I took physics 211 at UIUC, there would be a prerequisite/concurrent requirement for calc 2, but my community college only requires calc 1 (which I have). I\'m kind of doubting a 4 on my BC exam so I was planning on taking calc 2 during the spring. Would my cc physics class transfer (even though I don\'t have the calc 2 requirement)?', 30)
highest_matching_docs = retriever_object.get_highest_matching_docs('Would my cc physics class transfer (even though I don\'t have the calc 2 requirement)?', 30)
print(highest_matching_docs)
"""