"""
MNIST Convolutional Neural Network (CNN) Script
Built using the Keras 3 Sequential API.
"""

import os
# Idiomatic Keras 3 multi-backend setup (Defaulting to JAX or TensorFlow)
os.environ["KERAS_BACKEND"] = "jax"  

import keras
from keras import layers
import numpy as np

def main():
    # 1. DATA PREPARATION (Spatial Tensors)
    print("--- Loading and Preprocessing Data ---")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # CNNs need 3D tensors: (height, width, channels). MNIST is grayscale, so channels = 1.
    # Images are expanded to (28, 28, 1) and normalized to a [0, 1] range.
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0

    # Categorical encoding (One-Hot Vectors)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)


    # 2. MODEL DESIGN (The Sequential API for linear stacks of layers)
    print("\n--- Designing the Convolutional Sequential Architecture ---")
    
    # We pass a list of layers directly to the Sequential constructor
    model = keras.Sequential([
        # Define the structural entry shape explicitly
        keras.Input(shape=(28, 28, 1), name="spatial_input"),
        
        # First Convolutional block: learns 32 distinct 3x3 local feature maps
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", name="conv_1"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool_1"),
        
        # Second Convolutional block: deepens the representation space to 64 maps
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", name="conv_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool_2"),
        
        # Flattening transforms 3D spatial feature maps into a 1D feature vector
        layers.Flatten(name="flatten_layer"),
        
        # Regularization to combat overfitting before classification
        layers.Dropout(0.5, name="dropout_layer"),
        
        # Dense classification layer yielding 10 class probability distributions
        layers.Dense(10, activation="softmax", name="predictions")
    ], name="chollet_style_sequential_cnn")
    
    # Output the architectural blueprint
    model.summary()


    # 3. MODEL COMPILATION
    print("\n--- Compiling the Model ---")
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )


    # 4. TRAINING
    print("\n--- Training the CNN via .fit() ---")
    history = model.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=5,
        validation_split=0.1666
    )


    # 5. EVALUATION
    print("\n--- Evaluating Generalization Performance on Test Data ---")
    test_scores = model.evaluate(x_test, y_test, verbose=False)
    print(f"Test Loss: {test_scores[0]:.4f}")
    print(f"Test Accuracy: {test_scores[1]:.4f}")

if __name__ == "__main__":
    main()
