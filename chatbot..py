import numpy as np
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Load model and data
model = load_model('chatbot_model.model')
lemmatizer = WordNetLemmatizer()

# Load intents and classes
intents = json.loads(open('intense.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def preprocess_input(text):
    """
    Tokenize, lemmatize, and convert text to bag-of-words representation.
    """
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha()]

    bag = [0] * len(words)
    for token in tokens:
        if token in words:
            bag[words.index(token)] = 1

    return np.array(bag)

def predict_intent(text):
    """
    Predict the intent of the given text using the trained model.
    """
    bag = preprocess_input(text)
    prediction = model.predict(np.array([bag]))[0]
    max_index = np.argmax(prediction)
    intent = classes[max_index]

    return intent, prediction[max_index]

def get_response(intent):
    """
    Get a response based on the predicted intent.
    """
    responses = {
        'greeting': [
            "Hello! How can I assist you today?",
            "Hi there! How can I help you?"
        ],
        'where_to_donate': [
            "You can donate to our organization by visiting our website and clicking on the 'Donate' button.",
            "To make a donation, please go to our website and follow the instructions on the 'Support Us' page."
        ],
        'y4d_foundation': [
            "Y4D Foundation focuses on youth development through education and training.",
            "They offer various programs aimed at empowering young individuals."
        ],
        'katalyst': [
            "Katalyst works on improving access to education and vocational training.",
            "Their initiatives include scholarships and skill development workshops."
        ],
        'seva_sahayog': [
            "Seva Sahayog is dedicated to community development and social welfare.",
            "They provide resources and support for underprivileged communities."
        ],
        'default': [
            "I'm sorry, I don't have information on that topic. Could you please rephrase?"
        ]
    }

    return responses.get(intent, responses['default'])

def main():
    print("Hello! I am your NGO chatbot. Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        intent, probability = predict_intent(user_input)
        response = get_response(intent)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
