"""
Low-Level Custom Training Loop using Keras 3 Tensors.
This mirrors the pure-tensor workflow Chollet details for research-level control.
"""

import os
os.environ["KERAS_BACKEND"] = "jax"  # Works flawlessly with jax, tensorflow, or torch

import keras
from keras import layers
import numpy as np

def main():
    # 1. DATA MANIPULATION
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)

    # 2. ARCHITECTURE & OPTIMIZATION TENSORS
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(10, activation="softmax")
    ])

    optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
    loss_fn = keras.losses.CategoricalCrossentropy()
    
    # Keras stateless metric trackers
    train_acc_metric = keras.metrics.CategoricalAccuracy()

    # Create a raw python iterable batch dataset
    batch_size = 128
    
    # 3. THE CUSTOM STEP & LOOP ENGINE
    epochs = 3
    print("--- Beginning Low-Level Custom Training Loop ---")
    
    for epoch in range(epochs):
        print(f"\nStart of epoch {epoch + 1}")
        
        # Reset the accuracy tracker tensor at the beginning of each epoch
        train_acc_metric.reset_state()
        
        # Shuffle indices manually for pure-tensor iterations
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        
        # Batch Loop
        for step in range(len(x_train) // batch_size):
            batch_indices = indices[step * batch_size : (step + 1) * batch_size]
            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Open a GradientTape to record operations for automatic differentiation
            with keras.GradientTape() as tape:
                # Forward pass tracking predictions
                logits = model(x_batch, training=True)
                # Compute the loss tensor
                loss_value = loss_fn(y_batch, logits)

            # Retrieve the gradients of the trainable weights with respect to the loss
            grads = tape.gradient(loss_value, model.trainable_weights)
            
            # Update the model's weights using the optimizer
            optimizer.apply(grads, model.trainable_weights)

            # Update the live metrics state
            train_acc_metric.update_state(y_batch, logits)

            # Log progress every 100 batches
            if step % 100 == 0:
                current_acc = train_acc_metric.result().numpy()
                print(f"Batch {step}: Loss = {float(loss_value):.4f}, Accuracy = {float(current_acc):.4f}")

        # Final Epoch Summary
        epoch_acc = train_acc_metric.result().numpy()
        print(f"--- Epoch {epoch + 1} Final Tracking Accuracy: {float(epoch_acc):.4f} ---")

if __name__ == "__main__":
    main()
