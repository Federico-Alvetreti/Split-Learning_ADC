import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Define the path where the results are stored
results_path = "results"  # Replace with your actual path if different
plots_path = "plots"
os.makedirs(plots_path, exist_ok=True)

# List all the SNR folders
snr_folders = [str(i) for i in range(-10, 11)]  # SNR values from -10 to 10

# Initialize lists to hold the training results
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Loop through each folder (SNR value) and load the corresponding training results
for snr in snr_folders:
    folder_path = os.path.join(results_path, snr)
    result_file = os.path.join(folder_path, "training_results.json")
    
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            data = json.load(f)
        
        # Extract the relevant data
        train_losses.append(data["Train losses"])
        train_accuracies.append(data["Train accuracies"])
        val_losses.append(data["Val losses"])
        val_accuracies.append(data["Val accuracies"])

# Convert SNR values to float for plotting and color mapping
snr_values = [float(snr) for snr in snr_folders]

# Set up a color map for SNR values
import matplotlib.colors as mcolors
norm = mcolors.Normalize(vmin=min(snr_values), vmax=max(snr_values))
cmap = plt.cm.viridis  # You can choose a different colormap if you like

# Plot Training Accuracy vs Epochs for each SNR
plt.figure(figsize=(10, 6))
for i, accuracy in enumerate(train_accuracies):
    plt.plot(range(1, 21), accuracy, marker='o', color=cmap(norm(snr_values[i])))

# Add a colorbar for the SNR values (using current axes)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Empty array to make the colorbar work
cbar = plt.colorbar(sm, ax=plt.gca(), label="SNR Value")  # Explicitly add colorbar to the current axes

plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.title("Training Accuracy vs Epochs for Different SNRs")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_path, "train_accuracy_vs_epochs.png"))  # Save the plot as a PNG file
plt.close()

# Plot Training Loss vs Epochs for each SNR
plt.figure(figsize=(10, 6))
for i, loss in enumerate(train_losses):
    plt.plot(range(1, 21), loss, marker='o', color=cmap(norm(snr_values[i])))

# Add a colorbar for the SNR values (using current axes)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Empty array to make the colorbar work
cbar = plt.colorbar(sm, ax=plt.gca(), label="SNR Value")  # Explicitly add colorbar to the current axes

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epochs for Different SNRs")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_path, "train_loss_vs_epochs.png"))  # Save the plot as a PNG file
plt.close()

# Plot Validation Accuracy vs Epochs for each SNR
plt.figure(figsize=(10, 6))
for i, accuracy in enumerate(val_accuracies):
    plt.plot(range(1, 21), accuracy, marker='o', color=cmap(norm(snr_values[i])))

# Add a colorbar for the SNR values (using current axes)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Empty array to make the colorbar work
cbar = plt.colorbar(sm, ax=plt.gca(), label="SNR Value")  # Explicitly add colorbar to the current axes

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy vs Epochs for Different SNRs")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_path, "val_accuracy_vs_epochs.png"))  # Save the plot as a PNG file
plt.close()

# Plot Validation Loss vs Epochs for each SNR
plt.figure(figsize=(10, 6))
for i, loss in enumerate(val_losses):
    plt.plot(range(1, 21), loss, marker='o', color=cmap(norm(snr_values[i])))

# Add a colorbar for the SNR values (using current axes)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Empty array to make the colorbar work
cbar = plt.colorbar(sm, ax=plt.gca(), label="SNR Value")  # Explicitly add colorbar to the current axes

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss vs Epochs for Different SNRs")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_path, "val_loss_vs_epochs.png"))  # Save the plot as a PNG file
plt.close()

print("Plots saved in the 'plots' folder.")
