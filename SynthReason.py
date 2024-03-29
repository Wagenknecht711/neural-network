import numpy as np
import random
import math
resource_limit = 50
output_length = 250
fileName = "xaa"
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

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
    #eval based on accuracy perhaps?
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
   
    def softmax(self, x):
        exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def calculate_accuracy(self, X, y):
        correct_predictions = 0
        total_predictions = X.shape[0]
        for i in range(total_predictions):
            predicted_word_index = np.argmax(self.forward(X[i:i+1]))
            if y[i, predicted_word_index] == 1:
                correct_predictions += 1
        return correct_predictions / total_predictions

def build_vocabulary(text_file):
    with open(text_file, 'r') as file:
        words = file.read().lower().split()
    vocabulary = {}
    index = 0
    for word in words:
        if word not in vocabulary:
            vocabulary[word] = index
            index += 1
    return vocabulary

def generate_training_data(text_file, vocabulary):
    with open(text_file, 'r') as file:
        words = list(set(file.read().lower().split()))[:resource_limit]
        
    X_train = []
    y_train = []
    for i in range(len(words) - 1):
        input_word = words[i]
        output_word = words[i+1]
        input_vector = np.zeros(len(vocabulary))
        output_vector = np.zeros(len(vocabulary))
        input_vector[vocabulary[input_word]] = 1
        output_vector[vocabulary[output_word]] = 1
        X_train.append(input_vector)
        y_train.append(output_vector)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train

# Example usage
if __name__ == "__main__":
    # Sample vocabulary
    vocabulary = build_vocabulary(fileName)

    X_train, y_train = generate_training_data(fileName, vocabulary)
    # Generate training data from text file

    # Initialize and train the neural network
    nn = NeuralNetwork(input_size=len(vocabulary), hidden_size=300, output_size=len(vocabulary))
    nn.train(X_train, y_train, epochs=15, learning_rate=0.1)
    
    while(True):
        # Test prediction
        input_word = input("Enter word:")
        if input_word in vocabulary:
            input_vector = np.zeros(len(vocabulary))
            input_vector[vocabulary[input_word]] = 1
            prediction = nn.forward(input_vector.reshape(1, -1))
            predicted_word_index = np.argmax(prediction)
            predicted_word = list(vocabulary.keys())[list(vocabulary.values()).index(predicted_word_index)]
            output = predicted_word + " "
            for i in range(output_length):
                input_word = predicted_word
                input_vector = np.zeros(len(vocabulary))
                input_vector[vocabulary[input_word]] = 1
                prediction = nn.forward(input_vector.reshape(1, -1))
                predicted_word_index = np.argmax(prediction)
                predicted_word = list(vocabulary.keys())[list(vocabulary.values()).index(predicted_word_index)]
                output += predicted_word + " "
            print(output)
