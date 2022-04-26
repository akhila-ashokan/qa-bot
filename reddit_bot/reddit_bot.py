import praw
import os
from dotenv import load_dotenv

load_dotenv()

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

# create posts with test dataset 

# reply to the new posts 
for submission in subreddit.new(limit=100):
    if submission.id not in posts_replied_to:
        submission.reply("This is a bot replying!")
        posts_replied_to.append(submission.id)

# save list of posts already replied to
with open("reddit_bot/posts_replied_to.txt", "w") as f:
    for post_id in posts_replied_to:
        f.write(post_id + "\n")