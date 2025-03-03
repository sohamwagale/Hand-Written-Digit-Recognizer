import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def softmax(z):
    z = z - np.max(z)  # Prevent overflow
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    """Derivative of the ReLU function."""
    return (z > 0).astype(float)

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)  # Xavier initialization
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.dropout_rate = 0.8  # Keep 80% of neurons

    def feedforward(self, a, is_training=False):
        # Store activations and z values for backprop if training
        activations = [a]
        zs = []
        dropout_masks = []
        
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, a) + b
            zs.append(z)
            
            if i < self.num_layers - 2:  # Hidden layers: ReLU + Dropout
                a = relu(z)  # ReLU
                
                # Apply dropout only during training
                if is_training:
                    dropout_mask = np.random.binomial(1, self.dropout_rate, size=a.shape) / self.dropout_rate
                    a *= dropout_mask
                    dropout_masks.append(dropout_mask)
                else:
                    dropout_masks.append(None)
            else:  # Output layer: Softmax
                a = softmax(z)
            
            activations.append(a)
        
        if is_training:
            return a, activations, zs, dropout_masks
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        best_accuracy = 0
        best_weights = self.weights
        best_biases = self.biases
        
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:
                accuracy = self.evaluate(test_data)
                print(f"Epoch {j}: {accuracy} / {n_test}")
                
                # Save best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
            else:
                print(f"Epoch {j} complete")
        
        # Restore best model
        if test_data:
            self.weights = best_weights
            self.biases = best_biases
            print(f"Best accuracy: {best_accuracy} / {n_test}")

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # Weight decay (L2 regularization)
        lambda_val = 0.0001
        self.weights = [(1 - eta * lambda_val / len(mini_batch)) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                      for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Forward pass with dropout
        _, activations, zs, dropout_masks = self.feedforward(x, is_training=True)
        
        # Output layer error (cross-entropy with softmax)
        delta = activations[-1] - y
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        # Hidden layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            
            # Use ReLU derivative for hidden layers
            activation_prime = relu_prime(z)
            
            # Backpropagate error
            delta = np.dot(self.weights[-l+1].transpose(), delta) * activation_prime
            
            # Apply dropout mask from forward pass
            if dropout_masks[-l+1] is not None:
                delta *= dropout_masks[-l+1]
            
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives ∂C_x/∂a for the output activations."""
        return (output_activations - y)