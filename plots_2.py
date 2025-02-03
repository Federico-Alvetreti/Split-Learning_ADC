import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Define the path where the results are stored
results_path = "results"  # Modify this to the correct path if necessary
plots_path = "plots/last_epoch"  # Folder where the plots will be saved

# Create the 'plots' folder if it doesn't exist
if not os.path.exists(plots_path):
    os.makedirs(plots_path)

# Prepare the lists to store the SNR and corresponding metrics
snr_values = []
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Loop through each folder (which corresponds to an SNR value)
for snr in range(-10, 11):  # From -10 to 10
    snr_folder = os.path.join(results_path, str(snr))
    
    if os.path.exists(snr_folder):
        # Load the JSON data for each folder
        json_file = os.path.join(snr_folder, "training_results.json")  # Assuming the JSON file is named "result.json"
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Append the data to the corresponding lists
        snr_values.append(snr)
        train_losses.append(data["Train losses"][-1])  # Mean of all training losses
        train_accuracies.append(data["Train accuracies"][-1])  # Mean of all training accuracies
        val_losses.append(data["Val losses"][-1])  # Mean of all validation losses
        val_accuracies.append(data["Val accuracies"][-1])  # Mean of all validation accuracies

# Plot Training Accuracy vs SNR
plt.figure(figsize=(10, 6))
plt.plot(snr_values, train_accuracies, marker='o', label='Training Accuracy', color='b')
plt.xlabel("SNR Value")
plt.ylabel("Training Accuracy")
plt.title("Training Accuracy vs SNR")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_path, "train_accuracy_vs_snr.png"))  # Save the plot as a PNG file
plt.close()

# Plot Training Loss vs SNR
plt.figure(figsize=(10, 6))
plt.plot(snr_values, train_losses, marker='o', label='Training Loss', color='r')
plt.xlabel("SNR Value")
plt.ylabel("Training Loss")
plt.title("Training Loss vs SNR")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_path, "train_loss_vs_snr.png"))  # Save the plot as a PNG file
plt.close()

# Plot Validation Accuracy vs SNR
plt.figure(figsize=(10, 6))
plt.plot(snr_values, val_accuracies, marker='o', label='Validation Accuracy', color='g')
plt.xlabel("SNR Value")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy vs SNR")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_path, "val_accuracy_vs_snr.png"))  # Save the plot as a PNG file
plt.close()

# Plot Validation Loss vs SNR
plt.figure(figsize=(10, 6))
plt.plot(snr_values, val_losses, marker='o', label='Validation Loss', color='m')
plt.xlabel("SNR Value")
plt.ylabel("Validation Loss")
plt.title("Validation Loss vs SNR")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_path, "val_loss_vs_snr.png"))  # Save the plot as a PNG file
plt.close()

print("Plots saved in the 'plots' folder.")
