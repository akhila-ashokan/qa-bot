from gzip import WRITE
import os
import numpy as np

from text_retriever import TextRetriever

"""
This Python code is used for generating BERT embeddings for web dodcuments retrived and stored in 
web_data folder. The word embeddings are written to the folder web_data_embeddings. This code only needs to be 
once to generate pre-generate the embeddings for all the documents.
"""

WRITE_PATH = 'web_data_embeddings/'

text_retriever_util = TextRetriever()

for subdirectory in os.listdir('web_data/'):
    if os.path.isfile('web_data/' + subdirectory):
        continue
    for file in os.listdir('web_data/' + subdirectory):
        file_reader = open('web_data/' + subdirectory + '/' + file, 'r')
        web_content = file_reader.read()
        preprocessed_content = text_retriever_util.preprocess_text(web_content)
        word_embedding = text_retriever_util.get_vector_representation(preprocessed_content)
        print(type(word_embedding))
        if not os.path.exists(WRITE_PATH + subdirectory):
            os.mkdir(WRITE_PATH + subdirectory)
        with open(WRITE_PATH + subdirectory + '/' + file[:-4] + '.npy', 'wb') as write_file:
            np.save(write_file, word_embedding)