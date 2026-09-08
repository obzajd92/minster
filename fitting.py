from keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. Instantiate the monitoring Callbacks
callbacks_list = [
    # Halts training when validation loss stops improving for 2 consecutive epochs
    EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True  # Automatically rolls back weights to the absolute best epoch
    ),
    # Automatically serializes and saves only the best performing model iteration
    ModelCheckpoint(
        filepath="best_mnist_cnn.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

# 2. Pass the list directly into the training engine
print("\n--- Training with Dynamic Callbacks ---")
history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=10, # Can set higher epochs safely; EarlyStopping will intervene
    validation_split=0.1666,
    callbacks=callbacks_list
)
