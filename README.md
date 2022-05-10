# QA-Bot for Reddit 

## About the Project 

## Project Structure 

The overall project structure is shown in the diagram, but a more detailed overview of each folder is given below 

    ├── paths                                      # files that define path variables 
    ├── preprocessed_docs                          # contains preprocessed web_data documents 
    ├── queries                                    # sixty Reddit posts we first used as an evaluation dataset
    ├── queriesv2                                  # eighty Reddit posts of our final evaluation dataset
    ├── reddit_bot                                 # scripts to run Reddit bot
    ├── reddit_posts                               # script to pull Reddit posts from Pushshift 
    ├── testing_data		                       # contains testing data files and answers 
    ├── text_retrievers		                       # Python classes for different types of text retrievers
    ├── web_data			                       # methods that can be used to scrape web data
    ├── web_data_embeddings                        #  
    ├── LICENSE                                    # MIT License File  
    ├── README.md                                  #  methods that can be used to scrape web data
    ├── collect_pushshift.py                       #  methods that can be used to scrape web data
    ├── document_embeddings_generator.py           #  methods that can be used to scrape web data
    ├── evaluate.py                                #  methods that can be used to scrape web data
    ├── qa_detection.py                            #  methods that can be used to scrape web data
    └──score_reddit_posts.py                       #  methods that can be used to scrape web data
    

#### 📁 paths/

There should be one Python file for each of the text retrievers in the folder. The current files are as follows:

* `paths_tfidf.py`
* `paths_sbert.py`
* `paths_sbert_reranking.py`

Each of these files contain the value of three variables WEB_DATA_EMBEDDINGS_PATH, WEB_DATA_PATH and PREPROCESSED_DATA_PATH. The definitions of the variables are as follows:
* WEB_DATA_EMBEDDINGS_PATH: This is the path of the folder where any pre-computed document vectors for the text retriever are stored. For example, for TextRetrieverSBERT, the value of this variable is  'web_data_embeddings/web_data_embeddings_sbert/'. This variable is not defined for TextRetrieverTFIDF because we are not pre-computing and storing the document vectors (before the runtime). 
* WEB_DATA_PATH: This is the path where the scraped web data is located and is currently set to 'web_data/' for all the retriever classes.
* PREPROCESSED_DATA_PATH: This is the path to the folder contain any preprocessed form of the original documents (in WEB_DATA_PATH) if required for the retriever. Currently, this variable is set to 'preprocessed_docs/' for all the retrievers.

#### 📁 preprocessed_docs/
This folder contains the same subfolders and file names as web_data/, but the content is preprocessed text instead of the original text. The text has been preprocessed by removing URLs, non-alphabetical characters, extra spaces and stopwords. Since such pre-propcessing takes a significant amount of time, we are storing the preprocessed documents in order to save the time during runtime. 

#### 📁 queries/
This folder contains the sixty Reddit posts we first used as an evaluation dataset. The Reddit posts are stored as plain text in sixty separate files.

#### 📁 queries_v2/
This folder contains the eighty Reddit posts of our final evaluation dataset. The Reddit posts are stored as plain text in eighty separate files. Compared to the sixty posts in the “queries/” folder, these posts span a greater set of topics and were preprocessed and filtered more strictly. 

#### 📁 reddit_bot/
This folder contains all the Python scripts required for running the Reddit bot. This code will work for both local and external hosting options. Simply run `reddit_bot.py` to run the bot.  

##### 📁 reddit_posts/
This folder contains one script, GetRedditPosts.py. The script accesses Pushshift’s database and builds a Python list of r/UIUC subreddit posts matching one of the four flairs we chose. This list is stored in a pickle file for later processing. 

#### 📁 testing_data/
In addition to the Reddit queries in queries/ and queries_v2/, we also created queries manually. These queries along with their answers are provided in this folder. The answer set to the queries given in queries/ and queries_v2/ is also located in this folder. 

#### 📁 text_retrievers/
This folder contains the Python classes for different types of text retrievers.

* `text_retriever_tfidf.py`: This class uses TF-IDF vectorization with cosine similarity to find most relevant documents for a query.
* `text_retriever_sbert.py`: This class uses SBERT (Sentence BERT) with cosine similarity to find most relevant documents for a query.
* `text_retriever_sbert_reranking.py`: This class uses SBERT (Sentence BERT) with cosine similarity followed by ranking with a cross-encoder to find most relevant documents for a query.

To create a new retriever class, you should add a new file under the folder text_retrievers/ and the newly created class should contain all the above functions in order to function correctly. 

#### 📁 web_data/
This folder contains the methods that can be used to scrape and retrieve the text data of any website, as well as all the scraped web documents in txt format.
* `web_scrape.py`: contains methods for scraping specific webpage as well as an entire subdomain.
* `data_factory.py`: contains method for retrieving a dictionary from url to text data of any specified subdomain.
