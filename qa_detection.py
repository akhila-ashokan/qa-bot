import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

nltk.download('nps_chat')
posts = nltk.corpus.nps_chat.xml_posts()

train_text = [post.text for post in posts]

y = [post.get('class') for post in posts]

y_train = ["Q" if item == "whQuestion" or item == "ynQuestion" else "None" for item in y ]

y_test = []
test_text = []
with open('testing_data/reddit_query_answers_v2.txt', 'r') as f:
    for line in f:
        answer = next(f).strip()
        if answer != "None":
            answer = "Q"

        y_test.append(answer)

        query = open("queriesv2/" + line.strip(), "r", encoding='utf-8')
        query = query.read()
        query = query.replace("\n", " ")
        query = query.replace("&amp;#x200B;", "")
        query = query.replace("&gt;", "")
        test_text.append(query)
        
#Get TFIDF features
vectorizer = TfidfVectorizer(ngram_range=(1,3), 
                             min_df=0.001, 
                             max_df=0.7, 
                             analyzer='word')

X_train = vectorizer.fit_transform(train_text)
X_test = vectorizer.transform(test_text)

# Fitting Gradient Boosting classifier to the Training set
gb = GradientBoostingClassifier(n_estimators = 400, random_state=0)

gb.fit(X_train, y_train)

predictions_rf = gb.predict(X_test)

# Accuracy: 50%
print(accuracy_score(y_test, predictions_rf))

# Precision: 
print(precision_score(y_test, predictions_rf, pos_label = 'Q'))

# Recall: 
print(recall_score(y_test, predictions_rf, pos_label = 'Q'))

# F1 Score: 
print(f1_score(y_test, predictions_rf, pos_label = 'Q'))