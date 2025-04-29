import json
import random
import nltk
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')  

class ChatbotAssistant:
    def __init__(self, intents_file):
        self.intents_file = intents_file
        self.intents = []
        self.intents_responses = {}
        self.documents = []
        self.vocabulary = []
        self.model = None
        self.vectorizer = CountVectorizer(stop_words='english')
        self.encoder = LabelEncoder()
        self.lemmatizer = WordNetLemmatizer()
    
    def tokenize_and_lemmatize(self, text):
        """
        Tokenizes and lemmatizes input text. Returns a list of lemmatized words.
        """
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(word.lower()) for word in tokens]
        print(f"Tokens: {lemmatized_tokens}")  # Debugging line
        return lemmatized_tokens

    def parse_intents(self):
        """
        Parses intents from the intents file and prepares the documents and vocabulary.
        """
        print("Loading intents...")
        try:
            with open(self.intents_file, 'r') as f:
                data = json.load(f)
            self.intents = data["intents"]
            print(f"Loaded {len(self.intents)} intents.")

            for intent in self.intents:
                if "patterns" not in intent or "tag" not in intent:
                    print(f"Warning: Missing 'patterns' or 'tag' in intent: {intent}")
                    continue

                self.intents_responses[intent["tag"]] = intent["responses"]
                for pattern in intent["patterns"]:
                    if not pattern.strip(): 
                        continue  # Skip empty patterns

                    # Tokenize and lemmatize each pattern
                    tokens = self.tokenize_and_lemmatize(pattern)
                    self.documents.append((tokens, intent["tag"]))
                    self.vocabulary.extend(tokens)

        except Exception as e:
            print(f"Error parsing intents: {e}")

    def prepare_data(self):
        """
        Prepares the feature matrix (X) and label vector (y) for training.
        """
        print("Preparing data...")
        texts = [' '.join(doc) for doc, _ in self.documents]
        labels = [label for _, label in self.documents]

        # Create feature vectors and encode labels
        X = self.vectorizer.fit_transform(texts)
        y = self.encoder.fit_transform(labels)
        return X, y

    def train_model(self, batch_size=8, lr=0.01, epochs=200):
        """
        Trains the chatbot model using SGDClassifier.
        """
        print("Training model...")
        try:
            X, y = self.prepare_data()
            self.model = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=lr, max_iter=epochs)
            self.model.fit(X, y)
            print("Model training complete.")
        except Exception as e:
            print(f"Error during model training: {e}")

    def get_response(self, user_input):
        """
        Gets the chatbot's response to a user's input.
        """
        tokens = self.tokenize_and_lemmatize(user_input)
        X = self.vectorizer.transform([' '.join(tokens)])

        # Make prediction
        try:
            prediction = self.model.predict(X)[0]
            tag = self.encoder.inverse_transform([prediction])[0]
            responses = self.intents_responses.get(tag, ["I'm not sure I understand."])
            return random.choice(responses)
        except Exception as e:
            print(f"Error during response generation: {e}")
            return "Sorry, I couldn't process that."

# Example usage
if __name__ == '__main__':
    bot = ChatbotAssistant("intents.json")
    bot.parse_intents()
    bot.train_model()

    while True:
        inp = input("You: ")
        if inp.lower() in ["exit", "quit"]:
            break
        print("Miku:", bot.get_response(inp))
