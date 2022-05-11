import praw
import os
from dotenv import load_dotenv
from keep_alive import keep_alive
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from text_retrievers.text_retriever_sbert_reranking import TextRetrieverSBERTReranking


keep_alive()
load_dotenv()

# create text retriever object
retriever_object = TextRetrieverSBERTReranking()

# read Reddit credentials from .env file
r = praw.Reddit(
    client_id=os.getenv('client_id'),
    client_secret=os.getenv('client_secret'),
    password=os.getenv('password'),
    user_agent=os.getenv('user_agent'),
    username="Bubbly-Bath7845",
    redirect_uri='http://localhost:8080'
)

# read file with posts already replied to 
if not os.path.isfile("reddit_bot/posts_replied_to.txt"):
    posts_replied_to = []
else:
    with open("reddit_bot/posts_replied_to.txt", "r") as f:
       posts_replied_to = f.read()
       posts_replied_to = posts_replied_to.split("\n")
       posts_replied_to = list(filter(None, posts_replied_to))

# access subreddit
subreddit = r.subreddit("CS510")

# reply to the new posts 
for submission in subreddit.stream.submissions():
    if submission.id not in posts_replied_to:
        query = submission.title 
        query = query.replace("\n", " ")
        query = query.replace("&amp;#x200B;", "")
        query = query.replace("&gt;", "")
        print(query)

        top_docs = retriever_object.get_highest_matching_docs(query.strip(), 5)
        response = "Here are some helpful links:" 
        links = ""
        for val in top_docs['URL'].tolist():
            links = links + val + "\n"
        response = response + "\n" + links
        submission.reply(response)
        posts_replied_to.append(submission.id)

        # save list of posts already replied to
        with open("reddit_bot/posts_replied_to.txt", "w") as f:
            for post_id in posts_replied_to:
                f.write(post_id + "\n")