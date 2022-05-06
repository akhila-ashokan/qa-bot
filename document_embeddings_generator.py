from gzip import WRITE
import os
import numpy as np
import pickle

from text_retrievers.text_retriever_sbert import TextRetrieverSBERT

"""
This Python code is used for generating pre-processed embeddings for documents retrieved and stored in 
web_data folder. The word embeddings are written to the folder web_data_embeddings. This code only needs to be 
once to generate pre-generate the embeddings for all the documents.
"""

READ_PATH = 'preprocessed_docs/'
WRITE_PATH = 'web_data_embeddings/web_data_embeddings_sbert/'

text_retriever_util = TextRetrieverSBERT()

for subdirectory in os.listdir(READ_PATH):
    if os.path.isfile(READ_PATH + subdirectory):
        continue
    for file in os.listdir(READ_PATH+ subdirectory):
        file_reader = open(READ_PATH + subdirectory + '/' + file, 'r')
        web_content = file_reader.read()
        #preprocessed_content = text_retriever_util.preprocess_text(web_content)
        word_embedding = text_retriever_util.get_vector_representation(web_content)
        if not os.path.exists(WRITE_PATH + subdirectory):
            os.mkdir(WRITE_PATH + subdirectory)
        with open(WRITE_PATH + subdirectory + '/' + file[:-4] + '.npy', 'wb') as write_file:
            np.save(write_file, word_embedding)