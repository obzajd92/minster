"""
Keras 3 Framework featuring:
- Custom Thread-Safe Data Pipeline (PyDataset)
- Generative Adversarial Network (GAN) Architecture
- Concurrent Dual-Model Management & Multi-Optimizer Custom Loops
"""

import os
# Force Keras to utilize the JAX backend for XLA graph tracing compilation
os.environ["KERAS_BACKEND"] = "jax"  

import keras
from keras import layers
from keras import ops  # Keras backend-agnostic tensor operations framework
import numpy as np

# =====================================================================
# 1. CUSTOM DATA PIPELINE (Keras PyDataset)
# =====================================================================
class MNISTVectorDataset(keras.utils.PyDataset):
    """
    A custom, thread-safe, scalable data loader pipeline. 
    Loads and batches vectors without relying on third-party frameworks.
    """
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        (x_train, _), _ = keras.datasets.mnist.load_data()
        # Normalize and flatten real images to continuous vectors of shape (784,)
        self.x = x_train.reshape(-1, 784).astype("float32") / 255.0
        self.batch_size = batch_size
        self.indices = np.arange(len(self.x))

    def __len__(self):
        """Returns total number of batches per epoch."""
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        """Fetches and prepares a single target batch."""
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        # In an actual setup, you can load images from disk here
        batch_x = self.x[batch_indices]
        return batch_x

    def on_epoch_end(self):
        """Shuffles index references at every epoch transition point."""
        np.random.shuffle(self.indices)


# =====================================================================
# 2. CONCURRENT DUAL-MODEL GAN ENGINE
# =====================================================================
@keras.utils.register_keras_serializable(package="CustomGAN", name="GenerativeAdversarialNetwork")
class GAN(keras.Model):
    """
    Manages two structural networks concurrently through an overwritten 
    train_step, coordinating dual tracking metrics and independent optimizers.
    """
    def __init__(self, discriminator, generator, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        
        # Setup independent metric trackers for both networks
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn, **kwargs):
        """Overrides compile to map dual optimization protocols."""
        super().compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        """
        Executes simultaneous adversarial updates. 
        Graph-compiled via JAX/XLA for rapid matrix execution.
        """
        batch_size = ops.shape(real_images)[0]
        
        # Generate random vector samples in latent space
        random_latent_vectors = keras.random.normal(shape=(batch_size, self.latent_dim))

        # -----------------------------------------------------------------
        # STEP 1: TRAIN THE DISCRIMINATOR (Real vs Fake classification)
        # -----------------------------------------------------------------
        with keras.GradientTape() as tape:
            # Generate fake image vectors using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            
            # Combine real and generated vectors into a single tensor block
            combined_images = ops.concatenate([generated_images, real_images], axis=0)
            
            # Create classification target labels (Fake = 0, Real = 1)
            labels = ops.concatenate(
                [ops.zeros((batch_size, 1)), ops.ones((batch_size, 1))], axis=0
            )
            
            # Subtly smooth labels to help stabilize discriminator training
            labels += 0.05 * keras.random.uniform(shape=ops.shape(labels))

            # Forward pass through the discriminator
            predictions = self.discriminator(combined_images, training=True)
            d_loss = self.loss_fn(labels, predictions)

        # Apply gradients to the discriminator weights
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # -----------------------------------------------------------------
        # STEP 2: TRAIN THE GENERATOR (Trick the discriminator)
        # -----------------------------------------------------------------
        # Generate fresh latent noise vectors
        random_latent_vectors = keras.random.normal(shape=(batch_size, self.latent_dim))
        
        # We want the discriminator to classify fake images as real (label = 1)
        misleading_labels = ops.ones((batch_size, 1))

        with keras.GradientTape() as tape:
            # Forward pass: Latent Noise -> Generator -> Discriminator
            fake_images = self.generator(random_latent_vectors, training=True)
            predictions = self.discriminator(fake_images, training=True)
            g_loss = self.loss_fn(misleading_labels, predictions)

        # Apply gradients to the generator weights
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metric trackers
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}


# =====================================================================
# 3. ASSEMBLY AND RUNTIME PIPELINE
# =====================================================================
def main():
    LATENT_DIM = 64
    BATCH_SIZE = 128

    print("--- 1. Launching Custom PyDataset Data Pipeline ---")
    dataset_pipeline = MNISTVectorDataset(batch_size=BATCH_SIZE)

    print("\n--- 2. Stitching Discriminator and Generator Networks ---")
    # Discriminator: Takes image vector (784,) -> outputs probability (1,)
    discriminator = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(256, activation=layers.LeakyReLU(negative_slope=0.2)),
        layers.Dense(128, activation=layers.LeakyReLU(negative_slope=0.2)),
        layers.Dense(1, activation="sigmoid")
    ], name="discriminator_net")

    # Generator: Takes noise vector (LATENT_DIM,) -> maps to synthetic image space (784,)
    generator = keras.Sequential([
        layers.Input(shape=(LATENT_DIM,)),
        layers.Dense(256, activation=layers.LeakyReLU(negative_slope=0.2)),
        layers.Dense(512, activation=layers.LeakyReLU(negative_slope=0.2)),
        layers.Dense(784, activation="sigmoid")
    ], name="generator_net")

    print("\n--- 3. Instantiating the Adversarial Unified Model Framework ---")
    gan_model = GAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM)

    print("\n--- 4. Compiling the Concurrent Sub-Systems ---")
    gan_model.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss_fn=keras.losses.BinaryCrossentropy()
    )

    print("\n--- 5. Training the GAN System with Dynamic Data Pipelines ---")
    # Training via the custom data loader pipeline
    gan_model.fit(dataset_pipeline, epochs=3)

    print("\n--- 6. Verifying Production Architecture Exportation ---")
    generator.save("generative_backbone.keras")
    print("Generator architecture isolated and archived successfully!")

if __name__ == "__main__":
    main()
