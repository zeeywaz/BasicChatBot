# **Basic NLP Chatbot using NLTK and Scikit-learn**
This is a basic yet functional chatbot built from scratch using core Natural Language Processing (NLP) techniques. It uses the NLTK library for text preprocessing and Scikit-learn's SGDClassifier for intent classification.

### 🧠 Features:
Reads from an intents.json file structured by intent, patterns, and responses.

Tokenizes and lemmatizes input using NLTK for better generalization.

Converts text into numerical features using CountVectorizer.

Trains a simple machine learning model (SGD Classifier) to predict the intent of the user's input.

Responds based on the predicted intent using a set of predefined responses.

Easy to update — just edit the intents.json file with new intents, patterns, and responses.

### 🛠️ Tech Stack:
Python

NLTK (punkt, wordnet)

Scikit-learn (SGDClassifier, CountVectorizer, LabelEncoder)

JSON for storing intents

### 💬 Example:
```
You: hello
Miku: Hi there! How can I help you today?
```
### 📦 Setup:
Make sure you have Python installed.

Install required libraries:
```
pip install nltk scikit-learn
```
Run the chatbot:
```
python chatbot.py
```
