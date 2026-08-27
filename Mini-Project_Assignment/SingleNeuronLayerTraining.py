import numpy as np

class SimpleDenseLayer:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases randomly
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.bias = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        # CACHE: Save inputs for the backward pass
        self.inputs = inputs
        # Linear transformation
        return np.dot(inputs, self.weights) + self.bias

    def backward(self, dvalues):
        # Compute gradients with respect to weights, bias, and inputs
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbias = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class ReLUActivation:
    def forward(self, inputs):
        # CACHE: Save inputs for thresholding gradients
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, dvalues):
        # Gradient is pass-through where input > 0, else 0
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

# --- Execution of One Training Step ---
# 1. Setup data and layers
X = np.array([[1.0, -2.0], [3.0, 1.0]])  # Batch of 2 samples, 2 features
y_true = np.array([[1.0], [0.0]])        # Target values
layer = SimpleDenseLayer(n_inputs=2, n_neurons=1)
relu = ReLUActivation()

# 2. FORWARD PASS (with caching)
z = layer.forward(X)
predictions = relu.forward(z)

# Calculate Loss (MSE) and initial loss gradient (dLoss/dPred)
loss = np.mean((predictions - y_true) ** 2)
dloss_dpred = 2 * (predictions - y_true) / X.shape[0]

# 3. BACKWARD PASS (utilizing cache)
relu.backward(dloss_dpred)
layer.backward(relu.dinputs)

# 4. SGD UPDATE
learning_rate = 0.01
layer.weights -= learning_rate * layer.dweights
layer.bias -= learning_rate * layer.dbias

print(f"Computed Loss: {loss:.4f}")
