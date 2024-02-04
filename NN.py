print("Initializing...")
import numpy as np
import nltk
import re
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Load the model at startup
choice = input("Train or load model?[T/L]: ").lower()
with open("xaa", encoding="UTF-8") as f:
    text = f.read()

textSizeLimit = int(input("Text limit(e.g 10000): "))

# Tokenize and lemmatize the documents
lemmatizer = WordNetLemmatizer()
tokenized_text = word_tokenize(text.lower()[:textSizeLimit])
lemmatized_text = [lemmatizer.lemmatize(word) for word in tokenized_text]

# Create n-grams from the lemmatized text
n = 3  # You can change this to the desired n-gram size
n_grams = list(ngrams(lemmatized_text, n))

# Convert n-grams back to sentences
corpus = []
for n_gram in n_grams:
    sentence = ' '.join(n_gram)  # Take all but the last word of the n-gram
    corpus.append(f"{sentence}")

# Create a vocabulary using Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# Create input sequences for training
input_sequences = []
for doc in corpus:
    sequence = tokenizer.texts_to_sequences([doc])[0]
    input_sequences.append(sequence)

# Pad sequences for input data
max_sequence_length = max([len(seq) for seq in input_sequences])
padded_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Split input sequences into predictors and labels
X, y = padded_sequences[:, :-1], padded_sequences[:, -1]
y = np.array(y)

# Convert labels to one-hot encoding
y = np.eye(total_words)[y]

# Build and train the RNN model
model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=150, input_length=max_sequence_length - 1))
model.add(SimpleRNN(100, return_sequences=True))
model.add(SimpleRNN(100))
model.add(Dense(total_words, activation='softmax'))
def lr_schedule(epoch):
    return 0.001 * (0.1 ** (epoch // 10))
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
learning_rate_scheduler = LearningRateScheduler(lr_schedule)
if choice == "t":
    model.fit(X, y, epochs=int(input("Epochs count: ")), verbose=1)
    choiceB = input("Save model?[Y/N]: ").lower()
    if choiceB == "y":
        savefileName = input("Save model as: ")
        model.save(savefileName)
if choice == "l":
    loadfileName = input("Enter model to load: ")
    model = load_model(loadfileName)

# Generate a sentence
outputToken = int(input("Enter # of tokens to generate: "))
while True:
    seed_text = input("USER: ")
    for _ in range(outputToken):
        sequence = tokenizer.texts_to_sequences([seed_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_sequence_length - 1, padding='pre')
        predicted_word_index = np.argmax(model.predict(sequence), axis=-1)
        predicted_word = tokenizer.index_word[predicted_word_index[0]]
        seed_text += " " + predicted_word
    print(seed_text)
