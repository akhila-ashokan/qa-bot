"""
This script calls the text retriever on the entire test set and return the evaluation metrics
"""
import os
from text_retriever import TextRetriever

if __name__ == "__main__":
    # create text retriever object
    retriever_object = TextRetriever()

    ctr = 0
    total_questions = 0 

    # retrieve the entire test set and call the retriever on each question
    with open('testing_data/test_set.txt') as f:
        for line in f:
            total_questions += 1
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
                print("Correct Response Found!")
                print("Q: " +  line.strip())
                print("A: " + nextline)
                ctr+=1
    print(str(ctr) + " correct retrievals")

    accuracy = ctr/total_questions
    print("Accuracy: " + str(accuracy))


    with open('testing_data/reddit_query_answers.txt', 'r') as f:
        for line in f:
            nextline = next(f)
            total_questions += 1

            # query preprocessing  
            line = open("queries/" + line.strip(), "r", encoding='utf-8')
            query = line.read()
            query = query.replace("\n", " ")
            query = query.replace("&amp;#x200B;", "")
            query = query.replace("&gt;", "")
            
            # correct url preprocessing
            nextline = nextline.strip()
            if nextline[-1] == '/':
                nextline = nextline[0:len(nextline)-1]
            nextline = nextline.lower()
              
            # get the highest matching docs
            top_docs = retriever_object.get_highest_matching_docs(query.strip(), 5)
            top_urls = top_docs['URL'].tolist()
            top_urls_clean = []
            for i in range(len(top_urls)):
                url = top_urls[i].strip()
                if url[-1] == '/':
                    url = url[0:len(url)-1]
                url = url.lower()
                top_urls_clean.append(url)

            # check if correct url is in top 5 list 
            if nextline in top_urls_clean:
                index = top_urls_clean.index(nextline)
                print("Correct Response Found!")
                print("Q: " +  query.strip())
                print("A: " + nextline)
                print("Similarity Score:" + str(top_docs['Similarity Scores'].tolist()[index])  + "\n")
                ctr+=1
            if nextline == "none":
                print("No Correct Response!")
                print("Q: " + query.strip())
                print("Similarity Score:" + str(top_docs['Similarity Scores'].tolist())  + "\n")
    print(str(ctr) + " correct retrievals")

    accuracy = ctr/total_questions
    print("Accuracy: " + str(accuracy))
            

