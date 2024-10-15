import numpy as np
import matplotlib.pyplot as plt

def activation(x):
    return 1 if x >= 0 else 0

def perceptron_train(X, y, weights, learning_rate, bias, max_epochs):
    epochs = 0
    losses = []

    for epoch in range(max_epochs):
        global_error = 0
        for inputs, target in zip(X, y):
            weighted_sum = np.dot(inputs, weights) + bias
            output = activation(weighted_sum)
            error = target - output
            weights += learning_rate * error * inputs
            bias += learning_rate * error
            global_error += abs(error)

        loss = global_error / len(y)
        losses.append(loss)
        epochs += 1
        if global_error == 0:
            break
    
    return epochs, weights, bias, losses

def perceptron_test(X, weights, bias):
    for inputs in X:
        weighted_sum = np.dot(inputs, weights) + bias
        output = activation(weighted_sum)
        print(f"Input: {inputs} => Predicted Output: {output}")

def feedforward_test(weights, bias):
    while True:
        user_input = input("Enter new inputs (comma-separated) or 'exit' to quit: ")
        if user_input.lower() == 'exit':
            break
        inputs = np.array(list(map(float, user_input.split(","))))
        weighted_sum = np.dot(inputs, weights) + bias
        output = activation(weighted_sum)
        print(f"Input: {inputs} => Predicted Output: {output}")

def plot_loss(losses):
    plt.plot(losses)
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()

weights = np.array(list(map(float, input("Enter initial weights (comma-separated): ").split(","))))
learning_rate = float(input("Enter learning rate: "))
bias = float(input("Enter initial bias: "))
max_epochs = int(input("Enter max epochs: "))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

epochs, final_weights, final_bias, losses = perceptron_train(X, y, weights, learning_rate, bias, max_epochs)

print(f"Training completed in {epochs} epochs.")
print(f"Final weights: {final_weights}, Final bias: {final_bias}")

print("Testing the trained perceptron with training inputs...")
perceptron_test(X, final_weights, final_bias)

print("Performing feedforward testing with user input...")
feedforward_test(final_weights, final_bias)

plot_loss(losses)
