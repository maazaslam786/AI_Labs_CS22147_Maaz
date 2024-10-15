import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.bias_output = np.random.rand(output_size)
        self.losses = []

    def forward(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_layer_input)
        return self.output
    
    def backward(self, X, y):
        output_error = y - self.output
        self.losses.append(np.mean(np.abs(output_error)))
        d_output = output_error * sigmoid_derivative(self.output)
        hidden_layer_error = d_output.dot(self.weights_hidden_output.T)
        d_hidden_layer = hidden_layer_error * sigmoid_derivative(self.hidden_layer_output)
        self.weights_hidden_output += self.hidden_layer_output.T.dot(d_output) * self.learning_rate
        self.bias_output += np.sum(d_output, axis=0) * self.learning_rate
        self.weights_input_hidden += X.T.dot(d_hidden_layer) * self.learning_rate
        self.bias_hidden += np.sum(d_hidden_layer, axis=0) * self.learning_rate

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)

    def plot_loss(self):
        plt.plot(self.losses)
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

mlp = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)
mlp.train(X, y, epochs=1000)
mlp.plot_loss()

print("Testing MLP on training inputs:")
for i in range(len(X)):
    output = mlp.forward(np.array([X[i]]))
    print(f"Input: {X[i]} => Predicted Output: {(output)}")

print("\nFeedforward Testing with custom inputs...")

while True:
    user_input = input("Enter two binary values separated by comma (or type 'exit' to quit): ").strip()
    if user_input.lower() == 'exit':
        print("Exiting...")
        break
    test_values = np.array([list(map(int, user_input.split(',')))])
    output = mlp.forward(test_values)
    print(f"Input: {test_values[0]} => Predicted Output: {(output)}")
