import sentence_transformers
import os


class TextRetriever:

    def __init__(self) -> None:
        pass

    def compute_similarity_score(text_1, text_2):
        pass
    
    def get_vector_representation(self, text):
        bert_model = sentence_transfomers.SentenceTransformer('bert-base-nli-mean-tokens')
        embedded_text = bert_model.encode(text)
        return embedded_text


retriever_object = TextRetriever()
print(retriever_object.get_vector_representation('covid-19 mask mandate'))
#retriever_object('covid-19 mask mandate', 'masks are useful for preventing covid-19')
