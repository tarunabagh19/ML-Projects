import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Download required NLTK data
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews

# Prepare dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Convert to text
texts = [" ".join(words) for words, label in documents]
labels = [label for words, label in documents]

# Split data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Vectorisation
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train classifier
clf = MultinomialNB()
clf.fit(X_train_vect, y_train)

# Evaluate
y_pred = clf.predict(X_test_vect)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Function for predicting sentiment
def predict_sentiment(text):
    vect_text = vectorizer.transform([text])
    prediction = clf.predict(vect_text)[0]
    return prediction

# Test the model
while True:
    user_input = input("\nEnter a review (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    sentiment = predict_sentiment(user_input)
    print(f"Predicted Sentiment: {sentiment.capitalize()}")
