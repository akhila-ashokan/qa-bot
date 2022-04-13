from psaw import PushshiftAPI
import pickle
import datetime
import time
from collections import defaultdict

api = PushshiftAPI()

def get_posts(subreddit, filename):
    '''
    A python list of posts is pickled.
    Each entry of the list consists of a tuple of three strings
        (title text, flair text, body text)
    '''
    start_epoch = int(datetime.datetime(2022, 2, 1).timestamp())
    end_epoch = int(datetime.datetime(2022, 3, 1).timestamp())
    gen = api.search_submissions(subreddit=subreddit,
                                 after=start_epoch,
                                 before=end_epoch,
                                 limit=None)

    posts = [] # (title, flair, text)
    flairs = ["COVID-19", "New Student Question", "Prospective Students"]

    try:
        i = 0
        for thread in gen:
            if not hasattr(thread, "selftext") \
               or not hasattr(thread, "link_flair_text") \
               or not hasattr(thread, "title") \
               or thread.selftext[:4] in ("[del", "[rem"):
                # skip this post
                continue
            
            if len(thread.selftext) + len(thread.title) > 100 \
               and thread.link_flair_text in flairs:
                # accept this post
                
                posts.append((thread.title, thread.link_flair_text, thread.selftext))
                i += 1
                if i % 100 == 0:
                    print(i, datetime.datetime.fromtimestamp(thread.created))
                    
    except AttributeError as e:
        print(e)
    finally:
        with open(filename, 'wb') as f:
            pickle.dump(posts, f)
        print("Collected {:d} posts into {:}".format(len(posts), filename))
        
def open_pickle(filename):
    '''
    sample function to read the list of posts from a pickle file
    '''
    with open(filename, 'rb') as f:
        posts = pickle.load(f)
    print(len(posts))
    for i in range(1):
        print(posts[i])

def pickle_to_text(filename):
    with open(filename, 'rb') as f:
        posts = pickle.load(f)
    with open(filename[:-7]+".txt", 'w', encoding="utf-8") as f:
        for p in posts:
            f.write(str(p)+'\n\n')

fname = "reddit_posts_feb2022.pickle"
get_posts("UIUC", fname)
open_pickle(fname)
# pickle_to_text(fname)
