"""
Distributed Conditional GAN (cGAN) Architecture using Keras 3.
Utilizes the Keras Distribution API for Multi-GPU Data Parallelism.
"""

import os
# Force Keras to utilize the JAX backend (fully optimized for distribution arrays)
os.environ["KERAS_BACKEND"] = "jax"  

import keras
from keras import layers
from keras import ops  
import numpy as np

# =====================================================================
# 1. SETUP COOPERATIVE DISTRIBUTION STRATEGY (Multi-GPU/TPU)
# =====================================================================
def get_distribution_strategy():
    """
    Initializes a Data Parallel distribution matrix profile.
    Automatically assigns computational tensors across available hardware accelerators.
    """
    # 1. Retrieve information about available backend compute devices
    devices = keras.distribution.list_devices()
    print(f"--- Detected Compute Accelerators: {devices} ---")
    
    if len(devices) > 1:
        # 2. Define a Device Mesh structure grouping your accelerators
        mesh = keras.distribution.DeviceMesh(
            shape=(len(devices),), 
            axis_names=["data"], 
            devices=devices
        )
        # 3. Choose a layout strategy (DataParallel replicates models and splits batches)
        strategy = keras.distribution.DataParallel(mesh=mesh, layout_map=None)
        print("Using DataParallel strategy across multiple accelerators.")
        return strategy
    else:
        print("Single accelerator detected. Running on a standard single-device layout.")
        return None


# =====================================================================
# 2. CONCURRENT CONDITIONAL GAN ENGINE
# =====================================================================
@keras.utils.register_keras_serializable(package="CustomGAN", name="ConditionalGAN")
class ConditionalGAN(keras.Model):
    """
    Conditional GAN managing multi-input layers. Expects real images 
    and categorical condition vectors mapped simultaneously.
    """
    def __init__(self, discriminator, generator, latent_dim, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # State metric trackers
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
        """
        Receives real data containing both spatial image vectors and labels.
        Executed in parallel across hardware devices according to the distribution mesh.
        """
        # Unpack composite tuple data structure passed from dataset pipelines
        real_images, labels = data
        batch_size = ops.shape(real_images)[0]
        
        # Unpack condition mappings into 1-hot layout constraints
        one_hot_labels = ops.cast(labels, dtype="float32")

        # Sample random noise vectors for our generator's latent input space
        random_latent_vectors = keras.random.normal(shape=(batch_size, self.latent_dim))

        # -----------------------------------------------------------------
        # STEP 1: DISCRIMINATOR LOOP (Condition-Aware Pattern Identification)
        # -----------------------------------------------------------------
        with keras.GradientTape() as tape:
            # Generate fake image vectors conditioned on the target labels
            generated_images = self.generator([random_latent_vectors, one_hot_labels], training=True)
            
            # Combine real and fake data streams into unified tensors
            combined_images = ops.concatenate([generated_images, real_images], axis=0)
            combined_labels = ops.concatenate([one_hot_labels, one_hot_labels], axis=0)
            
            # Create classification target vectors (Fake = 0, Real = 1)
            validity_targets = ops.concatenate(
                [ops.zeros((batch_size, 1)), ops.ones((batch_size, 1))], axis=0
            )
            # Apply label smoothing to stabilize discriminator steps
            validity_targets += 0.05 * keras.random.uniform(shape=ops.shape(validity_targets))

            # Forward pass: Pass images along with their labels into the discriminator
            predictions = self.discriminator([combined_images, combined_labels], training=True)
            d_loss = self.loss_fn(validity_targets, predictions)

        # Apply gradients to the discriminator weights
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # -----------------------------------------------------------------
        # STEP 2: GENERATOR LOOP (Synthesize patterns to trick labels)
        # -----------------------------------------------------------------
        random_latent_vectors = keras.random.normal(shape=(batch_size, self.latent_dim))
        misleading_targets = ops.ones((batch_size, 1))

        with keras.GradientTape() as tape:
            # Generate fresh synthetic images using the target labels
            fake_images = self.generator([random_latent_vectors, one_hot_labels], training=True)
            # Pass the generated images and their target labels to the discriminator
            predictions = self.discriminator([fake_images, one_hot_labels], training=True)
            g_loss = self.loss_fn(misleading_targets, predictions)

        # Apply gradients to the generator weights
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Log training progress metrics
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}


# =====================================================================
# 3. ASSEMBLY AND DISTRIBUTION RUNTIME
# =====================================================================
def main():
    LATENT_DIM = 64
    NUM_CLASSES = 10
    BATCH_SIZE = 128

    print("--- 1. Organizing Preprocessed Label-Aware Arrays ---")
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)

    # Initialize distribution environment mappings
    strategy = get_distribution_strategy()

    # Define model generation tasks inside the strategy scope to handle automatic sharding
    if strategy is not None:
        strategy.__enter__()

    print("\n--- 2. Stitching Multi-Input Conditional Graph Entities ---")
    
    # --- BUILD GENERATOR SUB-GRAPH ---
    image_noise_input = layers.Input(shape=(LATENT_DIM,), name="noise_in")
    class_label_input = layers.Input(shape=(NUM_CLASSES,), name="class_in")
    
    # Merge condition vectors directly into the noise input feature vector
    g_merged = layers.Concatenate()([image_noise_input, class_label_input])
    g_dense = layers.Dense(7 * 7 * 64)(g_merged)
    g_reshape = layers.Reshape((7, 7, 64))(g_dense)
    g_upsample = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")(g_reshape)
    g_act1 = layers.LeakyReLU(negative_slope=0.2)(g_upsample)
    g_out_layer = layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding="same", activation="sigmoid")(g_act1)
    
    generator = keras.Model(inputs=[image_noise_input, class_label_input], outputs=g_out_layer, name="c_gen")

    # --- BUILD DISCRIMINATOR SUB-GRAPH ---
    spatial_image_input = layers.Input(shape=(28, 28, 1), name="spatial_in")
    class_condition_input = layers.Input(shape=(NUM_CLASSES,), name="condition_in")
    
    # Convert the 1D condition labels into a 3D tensor map matching the image dimension boundaries
    c_map = layers.Dense(28 * 28 * 1)(class_condition_input)
    c_map_reshaped = layers.Reshape((28, 28, 1))(c_map)
    
    # Merge the visual features and conditional layout maps
    d_merged = layers.Concatenate()([spatial_image_input, c_map_reshaped])
    d_conv1 = layers.Conv2D(64, kernel_size=3, strides=2, padding="same")(d_merged)
    d_act1 = layers.LeakyReLU(negative_slope=0.2)(d_conv1)
    d_flatten = layers.Flatten()(d_act1)
    d_out_layer = layers.Dense(1, activation="sigmoid")(d_flatten)
    
    discriminator = keras.Model(inputs=[spatial_image_input, class_condition_input], outputs=d_out_layer, name="c_disc")

    # --- COMPILE DISTRIBUTED STRUCTURAL WORKHORSE CONTAINER ---
    cgan = ConditionalGAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM)
    cgan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss_fn=keras.losses.BinaryCrossentropy()
    )

    if strategy is not None:
        strategy.__exit__(None, None, None)

    print("\n--- 3. Running Distributed Training Sequence via .fit() ---")
    # Tensors are seamlessly distributed across GPUs/TPUs under JAX/XLA rule configurations
    cgan.fit(x=[x_train, y_train],synthetic_image = generator.predict([random_noise, control_condition], verbose=0)print(f"Generated a conditional image array for digit '{target_digit}' with shape: {synthetic_image.shape}")
 batch_size=BATCH_SIZE, epochs=2)

    print("\n--- 4. Synthesizing Conditional Visual Targets ---")
    # Demonstrate target control by instructing the generator to output only the digit '7'
    target_digit = 7
    control_condition = np.zeros((1, NUM_CLASSES))
    control_condition[0, target_digit] = 1.0
    random_noise = np.random.normal(size=(1, LATENT_DIM))
if name == "main":main()

