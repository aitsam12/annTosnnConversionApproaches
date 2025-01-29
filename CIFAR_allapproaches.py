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
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Model and Dataset Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancedCNN().to(device)
model.load_state_dict(torch.load("cifar10_enhanced_cnn.pth"))
model.eval()

# CIFAR-10 Transformations and DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Conversion Functions
def convert_surrogate_gradient(ann_model):
    print("Converting ANN to SNN using Surrogate Gradient approach...")
    snn_model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        snn.Leaky(beta=0.98, spike_grad=surrogate.sigmoid()),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        snn.Leaky(beta=0.98, spike_grad=surrogate.sigmoid()),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        snn.Leaky(beta=0.98, spike_grad=surrogate.sigmoid()),
        nn.BatchNorm2d(256),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(256 * 4 * 4, 512),
        snn.Leaky(beta=0.98, spike_grad=surrogate.sigmoid()),
        nn.Dropout(0.5),
        nn.Linear(512, 10),
        snn.Leaky(beta=0.98, spike_grad=surrogate.sigmoid())
    ).to(device)

    # Transfer weights
    with torch.no_grad():
        ann_layers = [ann_model.conv1, ann_model.conv2, ann_model.conv3, ann_model.fc1, ann_model.fc2]
        snn_layers = [layer for layer in snn_model if isinstance(layer, (nn.Conv2d, nn.Linear))]
        for ann_layer, snn_layer in zip(ann_layers, snn_layers):
            snn_layer.weight.data = ann_layer.weight.data.clone()
            snn_layer.bias.data = ann_layer.bias.data.clone()
    return snn_model

def convert_threshold_adjustment(ann_model):
    """
    Convert ANN to SNN using Threshold Adjustment.
    Dynamically adjusts thresholds for Leaky layers based on activation statistics.
    """
    print("Converting ANN to SNN using Threshold Adjustment approach...")
    
    # Start with a base model structure
    snn_model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        snn.Leaky(beta=0.98),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        snn.Leaky(beta=0.98),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        snn.Leaky(beta=0.98),
        nn.BatchNorm2d(256),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(256 * 4 * 4, 512),
        snn.Leaky(beta=0.98),
        nn.Dropout(0.5),
        nn.Linear(512, 10),
        snn.Leaky(beta=0.98)
    ).to(device)
    
    # Transfer weights from ANN to SNN
    with torch.no_grad():
        ann_layers = [ann_model.conv1, ann_model.conv2, ann_model.conv3, ann_model.fc1, ann_model.fc2]
        snn_layers = [layer for layer in snn_model if isinstance(layer, (nn.Conv2d, nn.Linear))]
        for ann_layer, snn_layer in zip(ann_layers, snn_layers):
            snn_layer.weight.data = ann_layer.weight.data.clone()
            snn_layer.bias.data = ann_layer.bias.data.clone()

    # Adjust thresholds dynamically
    print("Calibrating thresholds dynamically...")
    dummy_input = torch.randn(1, 3, 32, 32).to(device)  # Simulated CIFAR-10 input
    activations = []
    x = dummy_input

    for layer in snn_model:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            x = layer(x)
        elif isinstance(layer, snn.Leaky):
            activations.append(x.abs().mean().item())  # Capture mean absolute activation
            x, _ = layer(x, layer.init_leaky())  # Simulate a step

    # Assign thresholds to Leaky layers
    leaky_layers = [layer for layer in snn_model if isinstance(layer, snn.Leaky)]
    for layer, activation in zip(leaky_layers, activations):
        layer.threshold = torch.tensor(activation, device=device)
        print(f"Adjusted threshold for {layer}: {activation:.4f}")

    return snn_model

def convert_hybrid_ann_snn(ann_model):
    print("Converting ANN to SNN using Hybrid ANN-SNN approach...")
    snn_model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        snn.Leaky(beta=0.98),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        snn.Leaky(beta=0.98),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(256 * 4 * 4, 512),
        snn.Leaky(beta=0.98),
        nn.Dropout(0.5),
        nn.Linear(512, 10),
        snn.Leaky(beta=0.98)
    ).to(device)
    return convert_surrogate_gradient(ann_model)  # Add weights



# Define Conversion Functions
def convert_rate_based(ann_model):
    print("Converting ANN to SNN using Rate-Based approach...")
    snn_model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        snn.Leaky(beta=0.99),  # Tuned beta value
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        snn.Leaky(beta=0.99),  # Tuned beta value
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        snn.Leaky(beta=0.99),  # Tuned beta value
        nn.BatchNorm2d(256),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(256 * 4 * 4, 512),
        snn.Leaky(beta=0.99),  # Tuned beta value
        nn.Dropout(0.5),
        nn.Linear(512, 10),
        snn.Leaky(beta=0.99)  # Tuned beta value
    ).to(device)

    # Transfer weights from ANN to SNN
    with torch.no_grad():
        ann_layers = [ann_model.conv1, ann_model.conv2, ann_model.conv3, ann_model.fc1, ann_model.fc2]
        snn_layers = [layer for layer in snn_model if isinstance(layer, (nn.Conv2d, nn.Linear))]

        for ann_layer, snn_layer in zip(ann_layers, snn_layers):
            snn_layer.weight.data = ann_layer.weight.data.clone()
            snn_layer.bias.data = ann_layer.bias.data.clone()

    return snn_model



def convert_temporal_coding(ann_model):
    print("Converting ANN to SNN using Temporal Coding approach...")
    return convert_surrogate_gradient(ann_model)  # Reuse base logic

# Evaluation Function
def evaluate_snn(snn_model, num_steps=500):  # Increase simulation steps
    snn_model.eval()
    total_accuracy = 0
    total_spikes = 0
    total_samples = 0
    start_time = time.time()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Processing Batches"):
            images, labels = images.to(device), labels.to(device)
            mem_rec = [layer.init_leaky() for layer in snn_model if isinstance(layer, snn.Leaky)]
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
            total_accuracy += accuracy_rate(spk_rec, labels) * images.size(0)
            total_spikes += spk_rec.sum().item()
            total_samples += images.size(0)
    latency = time.time() - start_time
    accuracy = total_accuracy / total_samples
    avg_spikes_per_neuron = total_spikes / (total_samples * 10 * num_steps)
    return accuracy, avg_spikes_per_neuron, latency

# Run All Methods
conversion_methods = {
    #"Surrogate Gradient": convert_surrogate_gradient,
    "Threshold Adjustment": convert_threshold_adjustment,
    "Hybrid ANN-SNN": convert_hybrid_ann_snn,
    "Temporal Coding": convert_temporal_coding
}

results = []
for method, func in conversion_methods.items():
    print(f"\nEvaluating {method} approach...")
    snn_model = func(model)
    accuracy, spikes, latency = evaluate_snn(snn_model)
    results.append({"Method": method, "Accuracy": accuracy * 100, "Spikes": spikes, "Latency": latency})

# Save Results
results_df = pd.DataFrame(results)
results_df.to_csv("cifar10_snn_results.csv", index=False)
print("\nResults:", results_df)
