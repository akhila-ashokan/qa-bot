"""
This script calls the text retriever on the entire test set and return the evaluation metrics
"""
import os
from text_retriever import TextRetriever
if __name__ == "__main__":
    # create text retriever object 
    retriever_object = TextRetriever()

    # retrieve the entire test set and call the retriever on each question
    with open('testing_data/test_set.txt') as f:
        for line in f:
            nextline = next(f)
        
            # get the highest matching docs 
            top_docs = retriever_object.get_highest_matching_docs(line, 5)['URL'].tolist()

            # check if correct url is in top 5 list 
            if nextline in top_docs:
                print("Q: " +  line)
                print("A: " + nextline)

            