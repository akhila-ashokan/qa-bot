from cmath import log
from data_factory import *
import re
import math

processed_data = dict()

def process_query(query):
    query = re.sub(r'[^\w\s]', '', query)
    query = query.lower()
    query = query.split(" ")
    result = dict()
    stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}
    for word in query:
        if word not in stop_words:
            result[word] = result.get(word, 0) + 1

    # print(result)
    return result

def process_doc(doc):
    doc = re.sub(r'[^\w\s]', '', doc)
    doc = doc.replace("  ", " ")
    doc = doc.lower()
    doc = doc.split(" ")
    stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}
    result = dict()
    for word in doc:
        if word not in stop_words:
            result[word] = result.get(word, 0) + 1
    # print(result)
    return result

def score_doc(query, doc):
    score = 0
    doc_length = 0
    for word in doc:
        # print("hi: " + word)
        doc_length += doc[word]
    #print(doc_length)
    for word in query:
        if word not in doc:
            continue
        c_w_q = query[word]
        temp = math.log(1 + doc[word]/doc_length)
        # print(temp)
        score += (c_w_q*temp)
    # print(score)
    return score
        


def rank_docs(query, docs=processed_data):
    # Preprocess query
    query = process_query(query)
    # print(len(docs))
    scores = []
    for url,v in docs.items():
        scores.append((score_doc(query, v),url))
    scores.sort(reverse=True)
    rankings = [i2 for i1,i2 in scores]
        
    # print(rankings)
    return rankings

def initialize():
    a = get_web_data("admissions")
    b = get_web_data("covid19")
    c = get_web_data("registrar")
    web_data = {**a, **b, **c}
    for url,v in web_data.items():
        if url[-1] == "/":
            url = url[0:len(url)-1]
        url = url.lower()
        processed_data[url] = process_doc(v)

def main():
    query = "Hey guys, I'm currently debating whether to transfer into UIUC engineering from a smaller engineering school. What's the best way for me to figure out how my credits transfer over?"
    initialize()
    rank_docs(query, processed_data)


if __name__ == "__main__":
    main()