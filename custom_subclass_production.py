"""
Keras 3 Subclassed Layer and Custom Train Loop Model.
Compiled automatically via JAX/XLA graph optimizations during `.fit()`.
"""

import os
# Force Keras to utilize the JAX backend for XLA graph tracing compilation
os.environ["KERAS_BACKEND"] = "jax"  

import keras
from keras import layers
from keras import ops  # Keras backend-agnostic tensor operations framework
import numpy as np

# =====================================================================
# 1. THE CUSTOM LAYER LAYOUT (Subclassing)
# =====================================================================
@keras.utils.register_keras_serializable(package="CustomElements", name="LinearResNetBlock")
class LinearResNetBlock(layers.Layer):
    """
    A custom layer architecture implementing a dense skip-connection block.
    Inherits state tracking and weight instantiation handling natively.
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        # Layers are recursively composable; we define child components here
        self.dense_1 = layers.Dense(units, activation="relu")
        self.dense_2 = layers.Dense(units)

    def build(self, input_shape):
        """Defines weights conditionally when input tensor shape is known."""
        # Check if a residual projection layer is needed to match vector shapes
        if input_shape[-1] != self.units:
            self.residual_proj = layers.Dense(self.units)
        else:
            self.residual_proj = None

    def call(self, inputs):
        """The computational graph logic applied to incoming tensors."""
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        
        # Calculate skip connection matching shapes
        residual = inputs if self.residual_proj is None else self.residual_proj(inputs)
        
        # Use Keras 'ops' to perform backend-agnostic mathematical operations
        return ops.add(x, residual)

    def get_config(self):
        """Allows Keras saving serialization engine to store configurations."""
        config = super().get_config()
        config.update({"units": self.units})
        return config


# =====================================================================
# 2. THE CUSTOM TRAINING LOOP MODEL (Overriding train_step)
# =====================================================================
@keras.utils.register_keras_serializable(package="CustomElements", name="CustomLoopModel")
class CustomLoopModel(keras.Model):
    """
    A Model that replaces the default step logic of `.fit()` with a custom loop.
    Enables low-level tensor manipulation while retaining native Callbacks and JIT.
    """
    def __init__(self, backbone, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        # Setup tracking state variables manually for compilation metrics
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.CategoricalAccuracy(name="accuracy")

    def call(self, inputs, training=False):
        return self.backbone(inputs, training=training)

    @property
    def metrics(self):
        """Informs Keras to automatically clear/reset tracker states every epoch."""
        return [self.loss_tracker, self.acc_tracker]

    def train_step(self, data):
        """
        The core training step. When `.fit()` executes this under JAX,
        it undergoes XLA graph-compilation for maximum speed.
        """
        # Unpack the inputs and targets passed from the dataset generator
        x, y = data

        # Use Keras GradientTape to monitor tensor operations
        with keras.GradientTape() as tape:
            # Forward Pass execution
            y_pred = self(x, training=True)
            # Compute loss via the metric compiled in .compile()
            loss = self.compute_loss(x=x, y=y, y_pred=y_pred)

        # Compute gradients relative to the trainable parameters
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Modify optimization step using the compiled optimizer instance
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update internal metric states
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(y, y_pred)

        # Return status dictionary for logs and callbacks tracking
        return {m.name: m.result() for m in self.metrics}


# =====================================================================
# 3. RUNNING THE COMPILED EXECUTION ENGINE
# =====================================================================
def main():
    print("--- 1. Organizing Preprocessed Data ---")
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype("float32") / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)

    print("\n--- 2. Stitching Custom Layers & Model Architecture ---")
    # Build core network utilizing our custom LinearResNetBlock
    backbone_network = keras.Sequential([
        layers.Input(shape=(784,)),
        LinearResNetBlock(units=256),
        LinearResNetBlock(units=128),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax")
    ])

    # Instantiating the custom training loop model container
    model = CustomLoopModel(backbone=backbone_network)

    print("\n--- 3. Compiling Model (Triggers XLA optimizations) ---")
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        loss="categorical_crossentropy"
    )

    print("\n--- 4. Executing Fast Graph-Compiled Training Loop via .fit() ---")
    # The first epoch handles structural graph tracing; successive epochs run at hardware limits
    model.fit(x_train, y_train, batch_size=256, epochs=3)

    print("\n--- 5. Verifying Serialization System Integration ---")
    model.save("custom_compiled_resnet.keras")
    print("Model serialized safely with Custom Elements catalogued successfully!")

if __name__ == "__main__":
    main()
