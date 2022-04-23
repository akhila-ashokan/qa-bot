from text_retriever import TextRetriever
import pickle
import pandas as pd

def get_best_posts(n):
    '''
    Parameters
    ----------
    n : int
        number of reddit posts to get.

    Returns
    -------
    n posts for which the model found documents with the highest scores.
    '''
    modele = TextRetriever()
    with open("reddit_posts/reddit_posts_2021.pickle", 'rb') as f:
        posts = pickle.load(f)
        print("There are", len(posts), "posts to score.")

    post_scores = []
    
    i = 1
    for title,flair,text in posts:
        res = modele.get_highest_matching_docs(title+". "+text, 3)
        post_scores.append((title, flair, text, res.iloc[0,1]))
        if i % 100 == 0:
            print("scoring post#", i)
        i += 1
    
    post_scores.sort(key=lambda t: -t[3])
    return post_scores[:n]

if __name__ == "__main__":
    post_scores = get_best_posts(60)
    i = 1
    for title, flair, text, score in post_scores:
        with open("queries/reddit_query_"+str(i)+".txt", "w", encoding="utf-8") as f:
            f.write(title + ". " + text)
        i += 1