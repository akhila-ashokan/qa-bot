"""
This script callsthe text retriever on the entire test set and return the evaluation metrics(accuracy, recall, precision, F1).
"""

# retrieve the entire test set 

# create text retriever object 
retriever_object = TextRetriever()

# get the highest matching docs 
print(retriever_object.get_highest_matching_docs('masks are useful for preventing covid-19', 5))