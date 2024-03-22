import numpy as np

mem = 25
output_length = 25
fileName = "xab"
n = 2  # Change n to the desired size of n-grams

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def softmax(self, x):
        exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def forward(self, X):
        # Forward pass
        self.hidden_output = self.sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        self.predicted_output = self.softmax(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output)
        return self.predicted_output

    def backward(self, X, y, learning_rate):
        # Backpropagation
        output_error = self.predicted_output - y

        # Output layer gradients
        dW_output = np.dot(self.hidden_output.T, output_error)
        db_output = np.sum(output_error, axis=0, keepdims=True)

        # Hidden layer gradients
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * self.sigmoid_derivative(self.hidden_output)
        dW_hidden = np.dot(X.T, hidden_error)
        db_hidden = np.sum(hidden_error, axis=0, keepdims=True)

        # Update weights and biases
        self.weights_hidden_output -= learning_rate * dW_output
        self.bias_output -= learning_rate * db_output
        self.weights_input_hidden -= learning_rate * dW_hidden
        self.bias_hidden -= learning_rate * db_hidden

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward and backward pass for each training example
            self.forward(X)
            self.backward(X, y, learning_rate)
            if epoch % 1 == 0: # Update rate for console
                loss = np.mean(-np.sum(y * np.log(self.predicted_output + 1e-8), axis=1))
                accuracy = self.calculate_accuracy(X, y)
                print(f'Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy}')

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def calculate_accuracy(self, X, y):
        correct_predictions = 0
        total_predictions = X.shape[0]
        for i in range(total_predictions):
            predicted_word_index = np.argmax(self.forward(X[i:i+1]))
            if y[i, predicted_word_index] == 1:
                correct_predictions += 1
        return correct_predictions / total_predictions

def build_vocabulary(ngrams):
    vocabulary = {}
    index = 0
    for ngram in ngrams:
        for word in ngram:
            if word not in vocabulary:
                vocabulary[word] = index
                index += 1
    return vocabulary

def generate_training_data(ngrams, vocabulary):
    X_train = []
    y_train = []
    for ngram in ngrams:
        input_ngram = ngram[:-1]
        output_word = ngram[-1]
        input_vector = np.zeros(len(vocabulary))
        output_vector = np.zeros(len(vocabulary))
        for word in input_ngram:
            input_vector[vocabulary[word]] = 1
        output_vector[vocabulary[output_word]] = 1
        X_train.append(input_vector)
        y_train.append(output_vector)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train

# Sample text
with open(fileName, 'r') as file:
    text = file.read()
# Split text into words
words = text.split()
# Generate n-grams
ngrams = [words[i:i+n] for i in range(len(words) - n + 1)]
# Build vocabulary from n-grams
vocabulary = build_vocabulary(ngrams)

# Generate training data from n-grams
X_train, y_train = generate_training_data(ngrams[:mem], vocabulary)

# Initialize and train the neural network
nn = NeuralNetwork(input_size=len(vocabulary), hidden_size=300, output_size=len(vocabulary))
nn.train(X_train, y_train, epochs=15, learning_rate=0.1)

# Inference loop
while True:
    input_ngram = input("Enter n-gram separated by space:").split()
    if len(input_ngram) == n - 1 and all(word in vocabulary for word in input_ngram):
        input_vector = np.zeros(len(vocabulary))
        for word in input_ngram:
            input_vector[vocabulary[word]] = 1
        prediction = nn.forward(input_vector.reshape(1, -1))
        predicted_word_index = np.argmax(prediction)
        predicted_word = list(vocabulary.keys())[predicted_word_index]
        output = predicted_word + " "
        for _ in range(output_length - 1):
            input_ngram = input_ngram[1:] + [predicted_word]
            input_vector = np.zeros(len(vocabulary))
            for word in input_ngram:
                input_vector[vocabulary[word]] = 1
            prediction = nn.forward(input_vector.reshape(1, -1))
            predicted_word_index = np.argmax(prediction)
            predicted_word = list(vocabulary.keys())[predicted_word_index]
            output += predicted_word + " "
        print(output)
    else:
        print("Invalid input. Please enter a valid n-gram.")

