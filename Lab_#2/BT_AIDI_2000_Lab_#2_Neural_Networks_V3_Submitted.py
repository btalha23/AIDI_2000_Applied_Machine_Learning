import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import mlflow
import mlflow.keras
import pandas as pd

# Generate dummy data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR problem

# Define hyperparameters to try
neurons_list = [2, 4, 10, 25]
activation_list = ['relu', 'tanh', 'sigmoid']
epochs_list = [1000, 2500, 5000]
optimizers_list = ['adam', 'sgd', 'rmsprop']

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Start an MLflow experiment
mlflow.set_experiment("AIDI2000_Lab #2 - Model Tracking")

for neurons in neurons_list:
    for activation in activation_list:
        for epochs in epochs_list:
            for optimizer in optimizers_list:
                tags = {
                    "num_neurons": neurons,
                    "activation_func": activation,
                    "num_epochs": epochs,
                    "optimizer_type": optimizer
                }
                with mlflow.start_run():
                    mlflow.set_tags(tags=tags)
                    # Log hyperparameters
                    # mlflow.log_param("neurons", neurons)
                    # mlflow.log_param("activation", activation)
                    # mlflow.log_param("epochs", epochs)
                    # mlflow.log_param("optimizer", optimizer)

                    mlflow.log_params(
                        {
                            "neurons": neurons,
                            "activation": activation,
                            "epochs": epochs,
                            "optimizer": optimizer,
                        }
                    )
                    
                    # Build the model
                    model = Sequential([
                        Dense(neurons, input_dim=2, activation=activation),  # First hidden layer
                        Dense(1, activation='sigmoid')  # Output layer
                    ])

                    # Compile the model
                    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

                    # Train the model
                    history = model.fit(X, y, epochs=epochs, verbose=1)
                    
                    # Evaluate the model
                    loss, accuracy = model.evaluate(X, y, verbose=1)
                    # mlflow.log_metric("loss", loss)
                    # mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metrics(
                        {
                            "accuracy": accuracy,
                            "loss": loss,
                        }
                    )

                    # Log the model
                    mlflow.keras.log_model(model, "model")

                    # Make predictions and log them
                    predictions = model.predict(X)
                    print(f'Neurons: {neurons}, Activation: {activation}, Epochs: {epochs}, Optimizer: {optimizer}')
                    print(f'Loss: {loss}, Accuracy: {accuracy}')
                    print('Predictions:')
                    print(predictions)

                    # convert array into dataframe 
                    pred_df = pd.DataFrame(predictions) 
                    pred_df.to_csv("predictions.csv")
                    mlflow.log_artifact('predictions.csv')

