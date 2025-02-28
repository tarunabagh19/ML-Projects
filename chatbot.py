import random
import nltk
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# NLTK setup
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

# Training data
training_data = {
    "hi": "Hello! How can I help you?",
    "hello": "Hi there! What can I do for you?",
    "how are you": "I'm just a bot, but I'm functioning as expected!",
    "what is your name": "I'm a simple chatbot created by machine learning.",
    "bye": "Goodbye! Have a great day.",
    "help": "Sure, I'm here to help. Ask me anything!",
    "thanks": "You're welcome!"
}

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    return " ".join([lemmatizer.lemmatize(word) for word in tokens])

# Prepare corpus
corpus = [preprocess(q) for q in training_data.keys()]
responses = list(training_data.values())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

model = LogisticRegression()
model.fit(X, responses)

def is_low_confidence(input_text):
    # Check prediction confidence
    X_test = vectorizer.transform([preprocess(input_text)])
    probs = model.predict_proba(X_test)
    confidence = max(probs[0])
    return confidence < 0.5  # You can adjust the threshold

def search_wikipedia(query):
    try:
        summary = wikipedia.summary(query, sentences=2)
        return f"I looked it up: {summary}"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"That topic is too broad. Try being more specific: {e.options[:3]}"
    except wikipedia.exceptions.PageError:
        return "Sorry, I couldn't find anything on that topic."
    except Exception:
        return "Something went wrong while searching."

def chatbot_response(user_input):
    if is_low_confidence(user_input):
        return search_wikipedia(user_input)
    else:
        X_test = vectorizer.transform([preprocess(user_input)])
        return model.predict(X_test)[0]

# Chat loop
print("Chatbot: Hello! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Bye!")
        break
    response = chatbot_response(user_input)
    print("Chatbot:", response)
