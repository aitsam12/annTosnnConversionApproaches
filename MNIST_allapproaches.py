import torch
import torch.nn as nn
import time
import pandas as pd
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import snntorch as snn
from snntorch import surrogate
from snntorch.functional.acc import accuracy_rate

# Load Pretrained ANN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Model and Dataset Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Ensure normalization
])

test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define Conversion Functions with Weight Transfer
def convert_rate_based(ann_model):
    print("Converting ANN to SNN using Rate-Based approach...")
    snn_model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        snn.Leaky(beta=0.95),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        snn.Leaky(beta=0.95),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        snn.Leaky(beta=0.95),
        nn.Linear(128, 10),
        snn.Leaky(beta=0.95)
    ).to(device)

    # Transfer weights from ANN to SNN
    snn_model[0].weight.data = ann_model.conv1.weight.data.clone()
    snn_model[0].bias.data = ann_model.conv1.bias.data.clone()
    snn_model[3].weight.data = ann_model.conv2.weight.data.clone()
    snn_model[3].bias.data = ann_model.conv2.bias.data.clone()
    snn_model[7].weight.data = ann_model.fc1.weight.data.clone()
    snn_model[7].bias.data = ann_model.fc1.bias.data.clone()
    snn_model[9].weight.data = ann_model.fc2.weight.data.clone()
    snn_model[9].bias.data = ann_model.fc2.bias.data.clone()

    return snn_model

def convert_surrogate_gradient(ann_model):
    print("Converting ANN to SNN using Surrogate Gradient approach...")
    snn_model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        snn.Leaky(beta=0.95, spike_grad=surrogate.sigmoid()),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        snn.Leaky(beta=0.95, spike_grad=surrogate.sigmoid()),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        snn.Leaky(beta=0.95, spike_grad=surrogate.sigmoid()),
        nn.Linear(128, 10),
        snn.Leaky(beta=0.95, spike_grad=surrogate.sigmoid())
    ).to(device)

    # Transfer weights from ANN to SNN
    snn_model[0].weight.data = ann_model.conv1.weight.data.clone()
    snn_model[0].bias.data = ann_model.conv1.bias.data.clone()
    snn_model[3].weight.data = ann_model.conv2.weight.data.clone()
    snn_model[3].bias.data = ann_model.conv2.bias.data.clone()
    snn_model[7].weight.data = ann_model.fc1.weight.data.clone()
    snn_model[7].bias.data = ann_model.fc1.bias.data.clone()
    snn_model[9].weight.data = ann_model.fc2.weight.data.clone()
    snn_model[9].bias.data = ann_model.fc2.bias.data.clone()

    return snn_model


def convert_threshold_adjustment(ann_model):
    """
    Convert ANN to SNN using Threshold Adjustment.
    Uses hardcoded thresholds for the Leaky layers.
    """
    print("Converting ANN to SNN using Threshold Adjustment approach...")

    # Create SNN model
    snn_model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        snn.Leaky(beta=0.95),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        snn.Leaky(beta=0.95),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        snn.Leaky(beta=0.95),
        nn.Linear(128, 10),
        snn.Leaky(beta=0.95)
    ).to(device)

    # Transfer weights from ANN to SNN
    snn_model[0].weight.data = ann_model.conv1.weight.data.clone()
    snn_model[0].bias.data = ann_model.conv1.bias.data.clone()
    snn_model[3].weight.data = ann_model.conv2.weight.data.clone()
    snn_model[3].bias.data = ann_model.conv2.bias.data.clone()
    snn_model[7].weight.data = ann_model.fc1.weight.data.clone()
    snn_model[7].bias.data = ann_model.fc1.bias.data.clone()
    snn_model[9].weight.data = ann_model.fc2.weight.data.clone()
    snn_model[9].bias.data = ann_model.fc2.bias.data.clone()

    # Hardcode thresholds for Leaky layers
    hardcoded_thresholds = [0.5938, 1.2348, 0.9824, 0.8312]

    # Assign thresholds to Leaky layers
    leaky_layer_indices = [1, 4, 8, 10]  # Indices of Leaky layers in the model
    for idx, threshold in zip(leaky_layer_indices, hardcoded_thresholds):
        snn_model[idx].threshold = torch.tensor(threshold, device=device)
        print(f"Adjusted threshold for layer {idx}: {threshold}")

    return snn_model




def convert_hybrid_ann_snn(ann_model):
    """
    Convert ANN to SNN using Hybrid ANN-SNN approach.
    Early layers function as ANN layers, and later layers function as SNN layers.
    """
    print("Converting ANN to SNN using Hybrid ANN-SNN approach...")

    # Define Hybrid ANN-SNN Model
    snn_model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),  # ANN Layer
        nn.ReLU(),  # ANN Layer activation
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),  # ANN Layer
        snn.Leaky(beta=0.95),  # Transition to SNN Layer
        nn.MaxPool2d(2, 2),
        nn.Flatten(),  # Transition to fully connected layers
        nn.Linear(64 * 7 * 7, 128),  # SNN Layer
        snn.Leaky(beta=0.95),  # SNN Layer activation
        nn.Linear(128, 10),  # SNN Layer
        snn.Leaky(beta=0.95)  # SNN Layer activation
    ).to(device)

    # Transfer Weights from ANN to Hybrid ANN-SNN
    with torch.no_grad():
        # Map ANN layers to SNN layers that have weights and biases
        ann_layers = [ann_model.conv1, ann_model.conv2, ann_model.fc1, ann_model.fc2]
        snn_layers = [
            layer for layer in snn_model
            if isinstance(layer, (nn.Conv2d, nn.Linear))  # Exclude non-trainable layers
        ]

        for ann_layer, snn_layer in zip(ann_layers, snn_layers):
            snn_layer.weight.data = ann_layer.weight.data.clone()
            snn_layer.bias.data = ann_layer.bias.data.clone()

    # Manually adjust thresholds for spiking layers
    thresholds = [0.6, 1.2, 0.9, 0.8]  # Example thresholds for tuning
    threshold_idx = 0
    for layer in snn_model:
        if isinstance(layer, snn.Leaky):
            layer.threshold = torch.tensor(thresholds[threshold_idx], device=device)
            print(f"Set threshold for Leaky layer {threshold_idx}: {thresholds[threshold_idx]:.4f}")
            threshold_idx += 1

    return snn_model






def convert_temporal_coding(ann_model):
    """
    Convert ANN to SNN using Temporal Coding.
    The spiking neurons fire at specific time steps based on input magnitude.
    """
    print("Converting ANN to SNN using Temporal Coding approach...")

    # Create SNN model
    snn_model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid()),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid()),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid()),
        nn.Linear(128, 10),
        snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
    ).to(device)

    # Transfer weights from ANN to SNN
    snn_model[0].weight.data = ann_model.conv1.weight.data.clone()
    snn_model[0].bias.data = ann_model.conv1.bias.data.clone()
    snn_model[3].weight.data = ann_model.conv2.weight.data.clone()
    snn_model[3].bias.data = ann_model.conv2.bias.data.clone()
    snn_model[7].weight.data = ann_model.fc1.weight.data.clone()
    snn_model[7].bias.data = ann_model.fc1.bias.data.clone()
    snn_model[9].weight.data = ann_model.fc2.weight.data.clone()
    snn_model[9].bias.data = ann_model.fc2.bias.data.clone()

    return snn_model




# Updated Evaluation Function with Adjusted Parameters
def evaluate_snn(snn_model, num_steps=200):  # Increase time steps for accuracy
    snn_model.eval()
    total_accuracy = 0
    total_spikes = 0
    total_samples = 0
    start_time = time.time()

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Processing Batches"):
            images, labels = images.to(device), labels.to(device)
            
            # Initialize membrane potentials
            mem_rec = []
            for layer in snn_model:
                if isinstance(layer, snn.Leaky):
                    mem_rec.append(layer.init_leaky())

            spk_rec = []
            for step in tqdm(range(num_steps), desc="Simulating Time Steps", leave=False):
                spk_out = images
                mem_idx = 0

                for layer in snn_model:
                    if isinstance(layer, snn.Leaky):
                        spk_out, mem_rec[mem_idx] = layer(spk_out, mem_rec[mem_idx])
                        mem_idx += 1
                    else:
                        spk_out = layer(spk_out)

                spk_rec.append(spk_out)

            spk_rec = torch.stack(spk_rec, dim=0)

            # Calculate accuracy and spike count
            total_accuracy += accuracy_rate(spk_rec, labels) * images.size(0)
            total_spikes += spk_rec.sum().item()
            total_samples += images.size(0)

    latency = time.time() - start_time
    accuracy = total_accuracy / total_samples
    avg_spikes_per_neuron = total_spikes / (total_samples * 10 * num_steps)  # 10 output neurons

    return accuracy, avg_spikes_per_neuron, latency

# Process Conversion Methods
results = []
conversion_methods = {
    #"Rate-Based": convert_rate_based,
    #"Surrogate Gradient Descent": convert_surrogate_gradient
    #"Temporal Coding": convert_temporal_coding
    #"Threshold Adjustment": convert_threshold_adjustment
    "Hybrid Approach": convert_hybrid_ann_snn
}

print("\nEvaluation Results:")
print(f"{'Method':<25} {'Accuracy (%)':<15} {'Avg Spikes/Neuron':<20} {'Latency (s)':<10}")

for method_name, conversion_function in tqdm(conversion_methods.items(), desc="Evaluating Methods"):
    print(f"\nEvaluating {method_name} approach...")
    try:
        snn_model = conversion_function(model)
        accuracy, avg_spikes_per_neuron, latency = evaluate_snn(snn_model)
        results.append({
            "Method": method_name,
            "Accuracy (%)": accuracy * 100,
            "Avg Spikes/Neuron": avg_spikes_per_neuron,
            "Latency (s)": latency
        })
        print(f"{method_name:<25} {accuracy * 100:<15.2f} {avg_spikes_per_neuron:<20.4f} {latency:<10.4f}")
    except Exception as e:
        print(f"Error evaluating {method_name}: {e}")
        results.append({
            "Method": method_name,
            "Accuracy (%)": None,
            "Avg Spikes/Neuron": None,
            "Latency (s)": None
        })

# Save Results
results_df = pd.DataFrame(results)
results_df.to_csv("hyb_snn_evaluation_results.csv", index=False)
print("\nResults saved to: snn_evaluation_results.csv")
print(results_df)
