from sentence_transformers import SentenceTransformer, util
from sklearn import metrics
import re
import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from gensim.parsing.preprocessing import remove_stopwords


from paths_sbert_long_preprocessed import WEB_DATA_EMBEDDINGS_PATH, WEB_DATA_PATH


class TextRetrieverSBERTLongPreprocessed:

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

        # Load cross-encoder for re-ranking
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-4-v2', max_length=510, num_labels=1)
        #self.reranker = spacy.load('en_core_web_md')

    def preprocess_text(self, text):
        """
        Preprocesses text including removal of URLs, non-alphabetical characters and extra spaces.

        Keyword arguments:
        text -- the text to be preprocessed
        """
        text_lines = text.split('\n')
        text_lines = list(filter(lambda line: len(line.strip()) >= 100, text_lines))
        text = ' '.join(text_lines)
        # Eliminating URLs
        text = re.sub(r'http\S+', '', text)
        # Eliminating non-alphabetical characters
        non_alpha_chars = re.compile('[^A-Za-z]')
        processed_text = re.sub('  ', ' ', non_alpha_chars.sub(' ', text))
        # Removing any extra spaces and converting into lower case
        processed_text = re.sub('\s+',' ', processed_text).lower()
        # Removing stop words
        processed_text = remove_stopwords(processed_text)
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
        Gets the BERT vector representation (embedding) for text.

        Keyword arguments:
        text -- the text to be embedded
        """
        bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Increasing maximum sequence length
        bert_model.max_seq_length = 510
        embedded_text = bert_model.encode(text)
        return embedded_text

    def get_highest_matching_docs(self, query, num_docs):
        """
        Gets the highest matching documents for a query

        Keyword arguments:
        query -- query to be processed
        num_docs -- number of matching documents to be returned
        """
        query = self.preprocess_text(query)
        query_vector = self.get_vector_representation(query)
        similarity_scores = self.compute_similarity_score([query_vector], self.documents_df['Embedding'].tolist())
        new_document_df = self.documents_df.copy()
        new_document_df['Similarity Scores'] = similarity_scores[0]
        new_document_df = new_document_df.sort_values(by='Similarity Scores', ascending=False)
        #retrieved_docs = new_document_df[:num_docs]['Path'].values
        retrieved_docs = new_document_df[:30]['Path'].values

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

        docs_subset = pd.DataFrame({'Path': retrieved_docs, 'Content': retrieved_docs_content, 'URL': urls})
        """
        embedded_query = self.reranker(query)
        
        reranker_scores = []
        for index, row in docs_subset.iterrows():
            embedded_doc = self.reranker(row['Content'])
            reranker_scores.append(embedded_query.similarity(embedded_doc))
        """
        list_of_pairs = []
        for index, row in docs_subset.iterrows():
            list_of_pairs.append((query, row['Content']))

        reranker_scores = self.reranker.predict(list_of_pairs)
        docs_subset['Score'] = reranker_scores
        docs_subset = docs_subset.sort_values(by='Score', ascending=False)

        #print(docs_subset)

        #final_docs = pd.DataFrame({'Text': retrieved_docs_content, 'URL': urls, 'Similarity Scores': new_document_df[:num_docs]['Similarity Scores'].values})
        final_docs = pd.DataFrame({'Text': docs_subset[:num_docs]['Content'].values, 'URL': docs_subset[:num_docs]['URL'].values, 'Similarity Scores': docs_subset[:num_docs]['Score'].values})
        return final_docs

"""
#Example Use:
retriever_object = TextRetrieverSBERTLong()
print(retriever_object.get_highest_matching_docs('masks are useful for preventing covid-19', 5))


retriever_object = TextRetrieverSBERTLongPreprocessed()
highest_matching_docs = retriever_object.get_highest_matching_docs("Quick question about community college transfer class. Incoming ECE freshman here. So I’m taking a replacement for physics 211 at a local community college during the fall. If I took physics 211 at UIUC, there would be a prerequisite/concurrent requirement for calc 2, but my community college only requires calc 1 (which I have). I’m kind of doubting a 4 on my BC exam so I was planning on taking calc 2 during the spring. Would my cc physics class transfer (even though I don’t have the calc 2 requirement)?", 30)
print(highest_matching_docs)

"""

retriever_object = TextRetrieverSBERTLongPreprocessed()
highest_matching_docs = retriever_object.get_highest_matching_docs("Quick question about community college transfer class. Incoming ECE freshman here. So I’m taking a replacement for physics 211 at a local community college during the fall. If I took physics 211 at UIUC, there would be a prerequisite/concurrent requirement for calc 2, but my community college only requires calc 1 (which I have). I’m kind of doubting a 4 on my BC exam so I was planning on taking calc 2 during the spring. Would my cc physics class transfer (even though I don’t have the calc 2 requirement)?", 30)
print(highest_matching_docs)
