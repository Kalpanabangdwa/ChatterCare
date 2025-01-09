from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import nltk
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import pickle
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# NLTK downloads
nltk.download('punkt')
nltk.download('wordnet')

import json

# Load intents.json (Make sure this is done before using it)
with open('intents.json') as file:
    intents_json = json.load(file)  # This should be a dictionary

# Preprocessing
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Tokenize and prepare data
for intent in intents_json['intents']:
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        documents.append((tokens, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatization
lemmatizer = nltk.WordNetLemmatizer()
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile and train
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save model
model.save('chatbot_model.keras')

# Load trained model
model = load_model('chatbot_model.keras')

# Helper functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [nltk.WordNetLemmatizer().lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    try:
        # Convert sentence to bag of words
        bow_vector = bow(sentence, words)
        res = model.predict(np.array([bow_vector]))[0]
        
        # Log the raw prediction results
        print(f"Raw prediction result: {res}")

        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        
        # Log the results that pass the threshold
        print(f"Filtered results (above threshold): {results}")
        
        results.sort(key=lambda x: x[1], reverse=True)

        # Return the sorted results as list of dictionaries
        return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    
    except Exception as e:
        print(f"Error in predict_class: {e}")
        return []


def get_response(tag, intents_json):
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I couldn't find an appropriate response."

# Flask app setup
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def chatbot_response():
    try:
        # Get the message from the user
        msg = request.form.get("msg", "")
        print(f"Received message: {msg}")

        if not msg:
            return jsonify({"response": "Please type a message."})

        # Call the prediction function
        intents = predict_class(msg)
        print(f"Predicted intents: {intents}")

        if not intents:
            return jsonify({"response": "Sorry, I couldn't understand that."})

        # Get the predicted intent tag
        predicted_tag = intents[0]['intent']
        print(f"Predicted intent tag: {predicted_tag}")

        # Get the response for the predicted intent
        response = get_response(predicted_tag, intents_json)  # Pass the intent tag and the full intents_json
        print(f"Bot response: {response}")

        if not response:
            return jsonify({"response": "Sorry, I couldn't find an appropriate response."})

        return jsonify({"response": response})

    except Exception as e:
        print(f"Error in chatbot_response: {e}")
        return jsonify({"response": "Sorry, there was an error processing your request."})

if __name__ == "__main__":
    app.run(debug=True)