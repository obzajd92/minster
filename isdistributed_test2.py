"""
Advanced Validation Pipeline evaluating Mixed-Precision (mixed_float16) Math
and Keras Profiler Execution Analytics under Distributed Layout Constraints.
"""

import os
# Force Keras to utilize the JAX backend (highly optimal for mixed-precision XLA tracing)
os.environ["KERAS_BACKEND"] = "jax"  

import keras
from keras import layers
from keras import ops  
import numpy as np

# =====================================================================
# 1. MIXED-PRECISION POLICY CONFIGURATION
# =====================================================================
# Configures 16-bit math for compute execution and 32-bit for master storage variables
keras.mixed_precision.set_dtype_policy("mixed_float16")
print(f"--- Global DType Policy Initialized: {keras.mixed_precision.dtype_policy()} ---")


# =====================================================================
# 2. DISTRIBUTED MODEL DEFINITIONS (Condition-Aware Architecture)
# =====================================================================
@keras.utils.register_keras_serializable(package="TestFramework", name="TestCGAN")
class TestConditionalGAN(keras.Model):
    """
    Mixed-precision safe Conditional GAN structure.
    Explicitly uses 'self.compute_dtype' for dynamically allocated noise tensors.
    """
    def __init__(self, discriminator, generator, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        
        # State tracking metrics
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn, **kwargs):
        super().compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        """Mixed-precision execution block graph-compiled via JAX/XLA."""
        real_images, labels = data
        batch_size = ops.shape(real_images)
        one_hot_labels = ops.cast(labels, dtype=self.compute_dtype)

        # CRITICAL FOR MIXED PRECISION: Match the dynamic noise tensor to the compute dtype policy
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), 
            dtype=self.compute_dtype
        )

        # -----------------------------------------------------------------
        # STEP 1: DISCRIMINATOR LOOP
        # -----------------------------------------------------------------
        with keras.GradientTape() as tape:
            generated_images = self.generator([random_latent_vectors, one_hot_labels], training=True)
            combined_images = ops.concatenate([generated_images, real_images], axis=0)
            combined_labels = ops.concatenate([one_hot_labels, one_hot_labels], axis=0)
            
            validity_targets = ops.concatenate(
                [ops.zeros((batch_size, 1)), ops.ones((batch_size, 1))], axis=0
            )
            # Match target matrices to the precision policy
            validity_targets = ops.cast(validity_targets, dtype=self.compute_dtype)

            predictions = self.discriminator([combined_images, combined_labels], training=True)
            d_loss = self.loss_fn(validity_targets, predictions)

        # Backpropagation updates master variables via optimizer scaling
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # -----------------------------------------------------------------
        # STEP 2: GENERATOR LOOP
        # -----------------------------------------------------------------
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), 
            dtype=self.compute_dtype
        )
        misleading_targets = ops.cast(ops.ones((batch_size, 1)), dtype=self.compute_dtype)

        with keras.GradientTape() as tape:
            fake_images = self.generator([random_latent_vectors, one_hot_labels], training=True)
            predictions = self.discriminator([fake_images, one_hot_labels], training=True)
            g_loss = self.loss_fn(misleading_targets, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Record metrics
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}


# =====================================================================
# 3. VERIFICATION ENGINE WITH PROF_LOGGING SETUP
# =====================================================================
def main():
    LATENT_DIM = 32
    NUM_CLASSES = 10
    BATCH_SIZE = 64  # Tensor Core optimizations favor multiples of 8 or 64
    IMAGE_SHAPE = (28, 28, 1)

    print("\n--- 1. Fabricating Precision Matrix Batches ---")
    # Synthetic tensors mirroring image structures
    mock_x = np.random.uniform(0.0, 1.0, size=(256, *IMAGE_SHAPE)).astype("float32")
    mock_y = keras.utils.to_categorical(np.random.randint(0, NUM_CLASSES, size=(256,)), NUM_CLASSES)

    print("\n--- 2. Setting Up Profiler Log Directories ---")
    log_dir = "logs/profiler_run"
    if os.path.exists(log_dir):
        import shutil
        shutil.rmtree(log_dir)

    # 3. CONSTRUCT CORE NETWORKS
    # Generator Node
    noise_in = layers.Input(shape=(LATENT_DIM,), name="noise_input")
    lbl_in = layers.Input(shape=(NUM_CLASSES,), name="label_input")
    g_merge = layers.Concatenate()([noise_in, lbl_in])
    g_dense = layers.Dense(7 * 7 * 32, activation="relu")(g_merge)
    g_reshape = layers.Reshape((7, 7, 32))(g_dense)
    g_upsample = layers.Conv2DTranspose(1, kernel_size=4, strides=4, padding="same", activation="sigmoid")(g_reshape)
    generator = keras.Model(inputs=[noise_in, lbl_in], outputs=g_upsample, name="test_gen")

    # Discriminator Node
    img_in = layers.Input(shape=IMAGE_SHAPE, name="img_input")
    cond_in = layers.Input(shape=(NUM_CLASSES,), name="cond_input")
    c_map = layers.Dense(28 * 28 * 1)(cond_in)
    c_map_reshaped = layers.Reshape(IMAGE_SHAPE)(c_map)
    d_merge = layers.Concatenate()([img_in, c_map_reshaped])
    d_conv = layers.Conv2D(32, kernel_size=3, strides=2, padding="same", activation="relu")(d_merge)
    d_flatten = layers.Flatten()(d_conv)
    # CRITICAL FOR MIXED-PRECISION STABILITY: Force output classification node to float32 explicitly
    d_out = layers.Dense(1, activation="sigmoid", dtype="float32", name="output_probability")(d_flatten)
    discriminator = keras.Model(inputs=[img_in, cond_in], outputs=d_out, name="test_disc")

    # 4. INSTANTIATE AND COMPILE PIPELINE MODEL
    cgan = TestConditionalGAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM)
    cgan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
        loss_fn=keras.losses.BinaryCrossentropy()
    )

    # 5. CONFIGURE KERAS PROFILER VIA TENSORBOARD CALLBACK
    print("\n--- 3. Activating TensorBoard Profile Engine Triggers ---")
    # Setting profile_batch allows you to isolate a specific window of batches to capture trace timelines
    profiler_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        profile_batch="2, 4",  # Traces hardware execution boundaries from batch 2 through batch 4
        update_freq="batch"
    )

    print("\n--- 4. Executing Mixed-Precision Profile Training Run ---")
    # Fitting on synthetic data to verify compilation matrices execute without memory fault lines
    cgan.fit(
        x=[mock_x, mock_y],
        batch_size=BATCH_SIZE,
        epochs=1,
        callbacks=[profiler_callback]
    )

    # 6. VERIFY DTYPE CONSTRAINTS WERE MET
    print("\n--- 5. Checking Mixed-Precision Variable Layer Types ---")
    print(f"Generator calculation compute data type: {generator.layers[2].compute_dtype}")
    print(f"Discriminator calculation compute data type: {discriminator.layers[3].compute_dtype}")
    print(f"Discriminator explicit Output Prediction data type: {discriminator.get_layer('output_probability').dtype_policy.output_dtype}")

    print("\n==================================================")
    print("     ALL ADVANCED PROFILER TESTS COMPLETED       ")
    print("==================================================")

if __name__ == "__main__":
    main()
