"""
This script calls the text retriever on the entire test set and return the evaluation metrics
"""
import os
from text_retriever import TextRetriever

if __name__ == "__main__":
    # create text retriever object
    retriever_object = TextRetriever()

    ctr = 0

    # retrieve the entire test set and call the retriever on each question
    with open('testing_data/test_set.txt') as f:
        for line in f:
            nextline = next(f)
            nextline = nextline.strip()
            if nextline[-1] == '/':
                nextline = nextline[0:len(nextline)-1]
            nextline = nextline.lower()
        
            # get the highest matching docs 
            top_urls = retriever_object.get_highest_matching_docs(line.strip(), 5)['URL'].tolist()
            top_urls_clean = []
            for i in range(len(top_urls)):
                url = top_urls[i].strip()
                if url[-1] == '/':
                    url = url[0:len(url)-1]
                url = url.lower()
                top_urls_clean.append(url)
                
            # check if correct url is in top 5 list 
            if nextline in top_urls_clean:
                print("Q: " +  line.strip())
                print("A: " + nextline)
                print("Top Answers:" + str(top_urls_clean))
                ctr+=1
    print(str(ctr) + " correct retrievals")