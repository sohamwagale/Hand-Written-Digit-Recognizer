import network
import mnist_loader
import pickle
import numpy as np
import time

def train_network():
    print("Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    
    # Use a smaller network for faster training
    print("Initializing neural network...")
    net = network.Network([784, 256,256,10])
    
    # Initialize with improved weights
    for w in net.weights:
        w *= 0.1  # Scale down initial weights to prevent exploding gradients
    
    print(f"Training on {len(training_data)} samples...")
    start_time = time.time()
    
    # Train with a smaller number of epochs for testing
    net.SGD(
        training_data, 
        epochs=50,             # Reduced epochs for testing
        mini_batch_size=64,    # Mini-batch size
        eta=0.1,             # Learning rate
        test_data=test_data    # Validation during training
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f} seconds")
    
    # Test the network
    correct = net.evaluate(test_data)
    total = len(test_data)
    print(f"Final accuracy: {correct}/{total} ({100.0*correct/total:.2f}%)")
    
    # Save the trained network
    print("Saving trained network...")
    with open('trained_network.pkl', 'wb') as f:
        pickle.dump((net.weights, net.biases), f)
    print("Network saved to 'trained_network.pkl'")

if __name__ == "__main__":
    train_network()