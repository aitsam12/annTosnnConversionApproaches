# ANN-to-SNN Conversion and Evaluation

This repository contains the implementation, evaluation, and results of different Artificial Neural Network (ANN) to Spiking Neural Network (SNN) conversion approaches. The methods were tested on three benchmark datasets: MNIST, Fashion-MNIST, and CIFAR-10. The repository is organized to provide a clear understanding of the training, conversion, and evaluation processes for ANN-to-SNN models.

## Repository Structure

### Files
- **`MNIST_training.py`**  
  Script to train the baseline ANN for the MNIST dataset.
  
- **`MNIST_allapproaches.py`**  
  Implements the five ANN-to-SNN conversion approaches for the MNIST dataset.
  
- **`MNIST_evaluation_results.csv`**  
  Evaluation results for the MNIST dataset, including accuracy, energy efficiency (spike counts), and latency.

- **`FMNIST_evaluation_results.csv`**  
  Evaluation results for the Fashion-MNIST dataset.

- **`CIFAR_training.py`**  
  Script to train the baseline ANN for the CIFAR-10 dataset.

- **`CIFAR_allapproaches.py`**  
  Implements the five ANN-to-SNN conversion approaches for the CIFAR-10 dataset.

- **`CIFAR_evaluation_results.csv`**  
  Evaluation results for the CIFAR-10 dataset.


## Datasets

- **MNIST**: Handwritten digit recognition dataset.
- **Fashion-MNIST**: Image dataset of clothing categories.
- **CIFAR-10**: Dataset of 10 classes of color images with diverse object categories.

---

## Conversion Approaches

The following ANN-to-SNN conversion approaches are implemented:
1. **Rate Coding**
2. **Temporal Coding**
3. **Threshold Adjustment**
4. **Surrogate Gradient Descent**
5. **Hybrid ANN-SNN Training**

---

## How to Use

### Clone the repository:
```bash
git clone https://github.com/aitsam12/ANN-to-SNN.git
cd ANN-to-SNN
