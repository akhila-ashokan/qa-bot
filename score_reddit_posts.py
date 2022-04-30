from text_retriever_sbert_long import TextRetrieverSBERTLong
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
    modele = TextRetrieverSBERTLong()
    with open("reddit_posts/reddit_posts_2021v2.pickle", 'rb') as f:
        posts = pickle.load(f)
        print("There are", len(posts), "posts to score.")

    post_scores = []
    
    i = 1
    j = 1
    for title,flair,text in posts:
        if "?" in title + " " + text:
            res = modele.get_highest_matching_docs(title+" "+text, 1)
            post_scores.append((title, flair, text, res.iloc[0,2]))
            if i % 100 == 0:
                print("scoring post#", i)
                with open("queriesv2/checkpoint"+str(i)+".pickle", "wb") as f:
                    pickle.dump(post_scores, f)
            i += 1
        if j % 100 == 0:
            print("processed post#", j)
        j += 1
    
    print("computed scores for", len(post_scores), "posts")
    post_scores.sort(key=lambda t: -t[3])
    return post_scores[:n]

if __name__ == "__main__":
    post_scores = get_best_posts(80)
    i = 1
    for title, flair, text, score in post_scores:
        with open("queriesv2/reddit_query_"+str(i)+".txt", "w", encoding="utf-8") as f:
            f.write(title + " " + text)
        i += 1