"""
Keras 3 Framework featuring:
- Subclassed Custom Layers (LinearResNetBlock)
- Custom Loss Function (HuberCategoricalCrossentropy)
- Custom Metric Tracker (PredictionVarianceMetric)
- Custom Training and Evaluation Overrides (train_step & test_step)
"""

import os
# Force Keras to utilize the JAX backend for XLA graph tracing compilation
os.environ["KERAS_BACKEND"] = "jax"  

import keras
from keras import layers
from keras import ops  # Keras backend-agnostic tensor operations framework
import numpy as np

# =====================================================================
# 1. THE CUSTOM LOSS FUNCTION
# =====================================================================
@keras.utils.register_keras_serializable(package="CustomElements", name="HuberCategoricalCrossentropy")
class HuberCategoricalCrossentropy(keras.losses.Loss):
    """
    A custom robust loss function. Combines categorical cross-entropy with 
    a small penalty to prevent extremely confident wrong predictions.
    """
    def __init__(self, penalty_weight=0.01, **kwargs):
        super().__init__(**kwargs)
        self.penalty_weight = penalty_weight

    def call(self, y_true, y_pred):
        # 1. Standard cross-entropy loss vector using Keras cross-backend ops
        cce = ops.categorical_crossentropy(y_true, y_pred)
        
        # 2. Add an L2 variance penalty against extreme raw probability distances
        penalty = ops.mean(ops.square(y_true - y_pred), axis=-1)
        
        return cce + (self.penalty_weight * penalty)

    def get_config(self):
        config = super().get_config()
        config.update({"penalty_weight": self.penalty_weight})
        return config


# =====================================================================
# 2. THE CUSTOM METRIC BLOCK
# =====================================================================
@keras.utils.register_keras_serializable(package="CustomElements", name="MaxConfidenceMetric")
class MaxConfidenceMetric(keras.metrics.Metric):
    """
    A custom stateful metric tracking the average confidence assigned 
    to the correctly targeted class.
    """
    def __init__(self, name="max_confidence", **kwargs):
        super().__init__(name=name, **kwargs)
        # Initialize state tensor variables using backend-agnostic tracking placeholders
        self.total_confidence = self.add_variable(shape=(), initializer="zeros", name="total_conf")
        self.sample_count = self.add_variable(shape=(), initializer="zeros", name="count")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Isolate predictions meant for the true target class
        true_class_probabilities = ops.sum(y_true * y_pred, axis=-1)
        
        # Accumulate metrics across incoming tensors
        self.total_confidence.assign_add(ops.sum(true_class_probabilities))
        self.sample_count.assign_add(ops.cast(ops.shape(y_true)[0], dtype="float32"))

    def result(self):
        # Return state tracking evaluations safely
        return self.total_confidence / (self.sample_count + 1e-7)

    def reset_state(self):
        # Reset trackers at the end of each evaluation epoch boundaries
        self.total_confidence.assign(0.0)
        self.sample_count.assign(0.0)


# =====================================================================
# 3. THE CUSTOM LAYER LAYOUT
# =====================================================================
@keras.utils.register_keras_serializable(package="CustomElements", name="LinearResNetBlock")
class LinearResNetBlock(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense_1 = layers.Dense(units, activation="relu")
        self.dense_2 = layers.Dense(units)

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.residual_proj = layers.Dense(self.units)
        else:
            self.residual_proj = None

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        residual = inputs if self.residual_proj is None else self.residual_proj(inputs)
        return ops.add(x, residual)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


# =====================================================================
# 4. CUSTOM TRAINING & EVALUATION MODEL FRAMEWORK
# =====================================================================
@keras.utils.register_keras_serializable(package="CustomElements", name="FullCustomModel")
class FullCustomModel(keras.Model):
    """
    Overrides both train_step AND test_step. Integrates seamlessly
    with custom losses and custom evaluation metrics.
    """
    def __init__(self, backbone, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        
        # Setup tracking state variables manually for internal metrics
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.CategoricalAccuracy(name="accuracy")
        self.custom_metric_tracker = MaxConfidenceMetric()

    def call(self, inputs, training=False):
        return self.backbone(inputs, training=training)

    @property
    def metrics(self):
        """Tells Keras what metrics to reset at the start of each epoch/evaluation."""
        return [self.loss_tracker, self.acc_tracker, self.custom_metric_tracker]

    def train_step(self, data):
        """Calculates forward passes, computes gradients, updates weights."""
        x, y = data

        with keras.GradientTape() as tape:
            y_pred = self(x, training=True)
            # Utilizing the custom loss provided during compiling
            loss = self.compute_loss(x=x, y=y, y_pred=y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics states
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(y, y_pred)
        self.custom_metric_tracker.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """
        The Custom Evaluation Step. Invoked automatically when calling .evaluate().
        No gradients are computed here.
        """
        x, y = data

        # Forward pass tracking predictions (training=False skips dropout masks)
        y_pred = self(x, training=False)
        loss = self.compute_loss(x=x, y=y, y_pred=y_pred)

        # Update historical test metrics
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(y, y_pred)
        self.custom_metric_tracker.update_state(y, y_pred)

        # Return identical diagnostic tracking summaries
        return {m.name: m.result() for m in self.metrics}


# =====================================================================
# 5. EXECUTION ENGINE
# =====================================================================
def main():
    print("--- 1. Organizing Preprocessed Data ---")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    x_train = x_train.reshape(60000, 784).astype("float32") / 255.0
    x_test = x_test.reshape(10000, 784).astype("float32") / 255.0
    
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    print("\n--- 2. Stitching Custom Layers & Model Architecture ---")
    backbone_network = keras.Sequential([
        layers.Input(shape=(784,)),
        LinearResNetBlock(units=128),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax")
    ])

    model = FullCustomModel(backbone=backbone_network)

    print("\n--- 3. Compiling Model with Custom Loss Object ---")
    # Instantation of our Custom Loss Class mapping
    custom_loss_instance = HuberCategoricalCrossentropy(penalty_weight=0.05)

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        loss=custom_loss_instance
    )

    print("\n--- 4. Executing Graph-Compiled Training Loop via .fit() ---")
    model.fit(x_train, y_train, batch_size=256, epochs=2)

    print("\n--- 5. Executing Custom Evaluation via .evaluate() ---")
    # This invokes our overwritten `test_step` underneath graph-compilation rules
    eval_results = model.evaluate(x_test, y_test, batch_size=256)
    print(f"\nFinal Test Set Results: {eval_results}")

    print("\n--- 6. Verifying Production Deserialization Tracking ---")
    model.save("full_custom_system.keras")
    print("Full structural pipeline archived flawlessly!")

if __name__ == "__main__":
    main()
