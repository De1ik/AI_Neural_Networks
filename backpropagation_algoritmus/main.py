import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.units import momentum


# Define the activation functions
class Sigmoid:
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, d_out):
        return d_out * self.output * (1 - self.output)


class Tanh:
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, d_out):
        return d_out * (1 - self.output ** 2)


class ReLU:
    def forward(self, x):
        self.output = np.maximum(0, x)
        return self.output

    def backward(self, d_out):
        return d_out * (self.output > 0)


# Define the loss function
class MSELoss:
    def forward(self, y_pred, y_true):
        self.error = y_pred - y_true
        return np.mean(self.error ** 2)

    def backward(self):
        return 2 * self.error / self.error.size


# Define the linear layer
class Linear:
    def __init__(self, input_dim, output_dim, momentum):
        self.weights = np.random.randn(input_dim, output_dim) * 0.1
        self.bias = np.zeros((1, output_dim))

        self.momentum = momentum
        self.prev_weight_d = np.zeros_like(self.weights)
        self.prev_bias_d = np.zeros_like(self.bias)

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, d_out, learning_rate):
        d_weights = np.dot(self.input.T, d_out)
        d_bias = np.sum(d_out, axis=0, keepdims=True)
        d_input = np.dot(d_out, self.weights.T)

        # Update weights and biases
        if self.momentum > 0:
            self.prev_weight_d = self.momentum * self.prev_weight_d + d_weights
            self.prev_bias_d = self.momentum * self.prev_bias_d + d_bias
            self.weights -= learning_rate * self.prev_weight_d
            self.bias -= learning_rate * self.prev_bias_d
        else:
            self.weights -= learning_rate * d_weights
            self.bias -= learning_rate * d_bias

        return d_input


# Define the model
class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad, learning_rate):
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                loss_grad = layer.backward(loss_grad, learning_rate)
            else:
                loss_grad = layer.backward(loss_grad)



def train(model, tr_x, tr_y, epochs, learning_rate, text="Training Loss Over Epochs"):
    loss_func = MSELoss()
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, tr_x.shape[0], 2):
            x = tr_x[i:i+1]
            y = tr_y[i:i+1]

            output = model.forward(x)
            loss = loss_func.forward(output, y)

            total_loss += loss
            grad_output = loss_func.backward()

            model.backward(grad_output, learning_rate)

        losses.append(total_loss / tr_x.shape[0])
        avg_loss = total_loss / tr_x.shape[0]

        if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")




    # Plot training loss
    plt.figure(figsize=(6, 4))
    plt.plot(range(epochs), losses, label="Training Loss")
    plt.title(text)
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid()
    plt.show()



layers_2 = [
    Linear(2, 4, momentum=0), ReLU(),
    Linear(4, 1, momentum=0), Sigmoid()
]

layers_2_with_momentum = [
    Linear(2, 4, momentum=0.8), ReLU(),
    Linear(4, 1, momentum=0.8), Sigmoid()
]

layers_3 = [
    Linear(2, 4, momentum=0), ReLU(),
    Linear(4, 4, momentum=0), ReLU(),
    Linear(4, 1, momentum=0), Sigmoid()
]

layers_3_with_momentum = [
    Linear(2, 4, momentum=0.8), ReLU(),
    Linear(4, 4, momentum=0.8), ReLU(),
    Linear(4, 1, momentum=0.8), Sigmoid()
]

# AND
x_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([[0], [0], [0], [1]])

# OR
x_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([[0], [1], [1], [1]])

# XOR
x_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])


# # ------------------------- AND / Layer 1 / Momentum / No Momentum / Ls 0.1 / Ls 0.01 -----------------------------

print("Training AND / Layer 1 / Without momentum / learning rate = 0.1")
model = Model(layers_2)
train(model, x_and, y_and, epochs=500, learning_rate=0.1, text="Training AND / Layer 1 / Without momentum / learning rate = 0.1")


print("Training AND / Layer 1 / With momentum = 0.8 / learning rate = 0.1")
model = Model(layers_2_with_momentum)
train(model, x_and, y_and, epochs=500, learning_rate=0.1, text="Training AND / Layer 1 / With momentum = 0.8 / learning rate = 0.1")


print("Training AND / Layer 1 / Without momentum / learning rate = 0.1")
model = Model(layers_2)
train(model, x_and, y_and, epochs=500, learning_rate=0.01, text="Training AND / Layer 1 / Without momentum / learning rate = 0.01")


print("Training AND / Layer 1 / With momentum = 0.8 / learning rate = 0.1")
model = Model(layers_2_with_momentum)
train(model, x_and, y_and, epochs=500, learning_rate=0.01, text="Training AND / Layer 1 / With momentum = 0.8 / learning rate = 0.01")

# ------------------------- AND / Layer 2 / Momentum / No Momentum / Ls 0.1 / Ls 0.01 -----------------------------

print("Training AND / Layer 2 / Without momentum / learning rate = 0.1")
model = Model(layers_3)
train(model, x_and, y_and, epochs=500, learning_rate=0.1, text="Training AND / Layer 2 / Without momentum / learning rate = 0.1")


print("Training AND / Layer 2 / With momentum = 0.8 / learning rate = 0.1")
model = Model(layers_3_with_momentum)
train(model, x_and, y_and, epochs=500, learning_rate=0.1, text="Training AND / Layer 2 / With momentum = 0.8 / learning rate = 0.1")


print("Training AND / Layer 2 / Without momentum / learning rate = 0.01")
model = Model(layers_3)
train(model, x_and, y_and, epochs=500, learning_rate=0.01, text="Training AND / Layer 2 / Without momentum / learning rate = 0.01")


print("Training AND / Layer 2 / With momentum = 0.8 / learning rate = 0.01")
model = Model(layers_3_with_momentum)
train(model, x_and, y_and, epochs=500, learning_rate=0.01, text="Training AND / Layer 1 / With momentum = 0.8 / learning rate = 0.01")



# ------------------------- OR / Layer 1 / Momentum / No Momentum / Ls 0.1 / Ls 0.01 -----------------------------

print("Training OR / Layer 1 / Without momentum / learning rate = 0.1")
model = Model(layers_2)
train(model, x_or, y_or, epochs=500, learning_rate=0.1, text="Training OR / Layer 1 / Without momentum / learning rate = 0.1")


print("Training OR / Layer 1 / With momentum = 0.8 / learning rate = 0.1")
model = Model(layers_2_with_momentum)
train(model, x_or, y_or, epochs=500, learning_rate=0.1, text="Training OR / Layer 1 / With momentum = 0.8 / learning rate = 0.1")


print("Training OR / Layer 1 / Without momentum / learning rate = 0.01")
model = Model(layers_2)
train(model, x_or, y_or, epochs=500, learning_rate=0.01, text="Training OR / Layer 1 / Without momentum / learning rate = 0.01")


print("Training OR / Layer 1 / With momentum = 0.8 / learning rate = 0.01")
model = Model(layers_2_with_momentum)
train(model, x_or, y_or, epochs=500, learning_rate=0.01, text="Training OR / Layer 1 / With momentum = 0.8 / learning rate = 0.01")
#
# # ------------------------- OR / Layer 2 / Momentum / No Momentum / Ls 0.1 / Ls 0.01 -----------------------------

print("Training OR / Layer 2 / Without momentum / learning rate = 0.1")
model = Model(layers_3)
train(model, x_or, y_or, epochs=500, learning_rate=0.1, text="Training OR / Layer 2 / Without momentum / learning rate = 0.1")



print("Training OR / Layer 2 / With momentum = 0.8 / learning rate = 0.1")
model = Model(layers_3_with_momentum)
train(model, x_or, y_or, epochs=500, learning_rate=0.1, text="Training OR / Layer 2 / With momentum = 0.8 / learning rate = 0.1")


print("Training OR / Layer 2 / Without momentum / learning rate = 0.01")
model = Model(layers_3)
train(model, x_or, y_or, epochs=500, learning_rate=0.01, text="Training OR / Layer 2 / Without momentum / learning rate = 0.01")



print("Training OR / Layer 2 / With momentum = 0.8 / learning rate = 0.01")
model = Model(layers_3_with_momentum)
train(model, x_or, y_or, epochs=500, learning_rate=0.01, text="Training OR / Layer 2 / With momentum = 0.8 / learning rate = 0.01")

# ------------------------- XOR / Layer 1 / Momentum / No Momentum / Ls 0.1 / Ls 0.01 -----------------------------

print("Training XOR / Layer 1 / Without momentum / learning rate = 0.1")
model = Model(layers_2)
train(model, x_xor, y_xor, epochs=500, learning_rate=0.1, text="Training XOR / Layer 1 / Without momentum / learning rate = 0.1")


print("Training XOR / Layer 1 / With momentum = 0.8 / learning rate = 0.1")
model = Model(layers_2_with_momentum)
train(model, x_xor, y_xor, epochs=500, learning_rate=0.1, text="Training XOR / Layer 1 / With momentum = 0.8 / learning rate = 0.1")


print("Training XOR / Layer 1 / Without momentum / learning rate = 0.01")
model = Model(layers_2)
train(model, x_xor, y_xor, epochs=500, learning_rate=0.01, text="Training XOR / Layer 1 / Without momentum / learning rate = 0.01")


print("Training XOR / Layer 1 / With momentum = 0.8 / learning rate = 0.01")
model = Model(layers_2_with_momentum)
train(model, x_xor, y_xor, epochs=500, learning_rate=0.01, text="Training XOR / Layer 1 / With momentum = 0.8 / learning rate = 0.01")

# ------------------------- XOR / Layer 2 / Momentum / No Momentum / Ls 0.1 / Ls 0.01 -----------------------------

print("Training XOR / Layer 2 / Without momentum / learning rate = 0.1")
model = Model(layers_3)
train(model, x_xor, y_xor, epochs=500, learning_rate=0.1, text="Training XOR / Layer 2 / Without momentum / learning rate = 0.1")


print("Training XOR / Layer 2 / With momentum = 0.8 / learning rate = 0.1")
model = Model(layers_3_with_momentum)
train(model, x_xor, y_xor, epochs=500, learning_rate=0.1, text="Training XOR / Layer 2 / With momentum = 0.8 / learning rate = 0.1")


print("Training XOR / Layer 2 / Without momentum / learning rate = 0.01")
model = Model(layers_3)
train(model, x_xor, y_xor, epochs=500, learning_rate=0.01, text="Training XOR / Layer 2 / Without momentum / learning rate = 0.01")


print("Training XOR / Layer 2 / With momentum = 0.8 / learning rate = 0.01")
model = Model(layers_3_with_momentum)
train(model, x_xor, y_xor, epochs=500, learning_rate=0.01, text="Training XOR / Layer 2 / With momentum = 0.8 / learning rate = 0.01")


