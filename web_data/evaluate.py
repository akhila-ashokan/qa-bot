"""
This script calls the text retriever on the entire test set and return the evaluation metrics
"""
import os
# from text_retriever import TextRetriever
from text_retriever2 import *
from text_retriever2 import rank_docs
if __name__ == "__main__":
    # create text retriever object 
    ctr = 0
    # retrieve the entire test set and call the retriever on each question
    with open('../testing_data/test_set.txt') as f:
        for line in f:
            nextline = next(f)
            nextline = nextline.strip()
            if nextline[-1] == '/':
                nextline = nextline[0:len(nextline)-1]
            nextline = nextline.lower()
        
            # get the highest matching docs 
            # top_docs = retriever_object.get_highest_matching_docs(line, 5)['URL'].tolist()
            initialize()
            top_docs = rank_docs(line)[0:5]
            # print(top_docs)
            # check if correct url is in top 5 list 
            
            if nextline in top_docs:
                print("Q: " +  line)
                print("A: " + nextline)
                ctr+=1
            #print(nextline)
            #print(top_docs[2])
    print(str(ctr) + " correct answers")

            