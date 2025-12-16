from common_training import run_training_loop

# --- Main script execution after defining functions ---
if __name__ == '__main__':
    # Define hyperparameters
    input_shape = [1] # Original_x has 1 dimension, lift_dim will be 1000
    output_dim = 1
    hidden_dims = [100, 10]
    epochs = 10000
    learning_rate = 1e-3
    log_every_n_epochs = 1000
    random_seed = 42
    min_val_loss = 0.5

    # Run the training loop
    final_model, final_train_loss_history, final_val_loss_history = run_training_loop(
        input_shape=input_shape,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        epochs=epochs,
        learning_rate=learning_rate,
        log_every_n_epochs=log_every_n_epochs,
        random_seed=random_seed,
        min_val_loss=min_val_loss,
        training_type='fomaml' # Changed to 'fomaml' to test the MAML loop
    )

    print("Training finished.")
    print(f"Final Training Loss: {final_train_loss_history[-1]:.4f}")
    if final_val_loss_history: # Check if validation loss history is not empty
        print(f"Final Validation Loss: {final_val_loss_history[-1]:.4f}")
    else:
        print("No validation loss recorded (possibly due to early stopping on first epoch).")