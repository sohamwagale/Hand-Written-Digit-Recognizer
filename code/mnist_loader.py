import pickle
import gzip
import numpy as np
from scipy.ndimage import rotate, shift
import os

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data."""
    # Check if the data file exists
    if not os.path.exists('../data/mnist.pkl.gz'):
        print("Error: MNIST data file not found. Please download it from:")
        print("http://deeplearning.net/data/mnist/mnist.pkl.gz")
        print("and place it in the ../data/ directory.")
        raise FileNotFoundError("MNIST data file not found")
        
    # Load the data
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere. Used to convert a digit (0...9)
    into a corresponding desired output from the neural network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def augment_image(image):
    """Create an augmented version of the input image with
    random rotation and shift."""
    # Reshape to 2D for image operations
    img_2d = image.reshape(28, 28)
    
    # Random rotation (-15 to +15 degrees)
    angle = np.random.uniform(-15, 15)
    rotated = rotate(img_2d, angle, reshape=False, mode='nearest')
    
    # Random shift (±2 pixels)
    dx, dy = np.random.randint(-2, 3, size=2)
    shifted = shift(rotated, (dy, dx), mode='constant', cval=0.0)
    
    # Ensure values are in [0, 1] range
    shifted = np.clip(shifted, 0.0, 1.0)
    
    # Reshape back to column vector
    return shifted.reshape(784, 1)

def load_data_wrapper():
    """Return a tuple containing (training_data, validation_data, test_data).
    
    The training_data is returned as a list containing (x, y) tuples,
    where x is a 784-dimensional numpy.ndarray containing the input image,
    and y is a 10-dimensional numpy.ndarray containing the unit vector
    representation of the digit (0...9).
    
    validation_data and test_data are lists containing (x, y) tuples
    where x is a 784-dimensional numpy.ndarray and y is the corresponding
    classification, i.e., the digit values (integers) corresponding to x.
    """
    tr_d, va_d, te_d = load_data()
    
    # Format training data
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    # Use a subset of training data for testing
    training_data = list(zip(training_inputs, training_results))[:10000]  # Use only 10,000 examples    
    # Format validation data
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    
    # Format test data
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    
    """
    # Augment training data
    print("Augmenting training data...")
    training_data_augmented = []
    for x, y in training_data:
        training_data_augmented.append((x, y))  # Original sample
        training_data_augmented.append((augment_image(x), y))  # Augmented sample
        
    """
    
    print("Using original training data without augmentation...")
    training_data_augmented = training_data 
    
    print(f"Original training samples: {len(training_data)}")
    print(f"Augmented training samples: {len(training_data_augmented)}")
    
    return (training_data_augmented, validation_data, test_data)