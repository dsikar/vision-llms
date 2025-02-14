#!/usr/bin/env python3
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix
import os

# Dataset specific configurations
DATASET_CONFIGS = {
    'mnist': {
        'num_classes': 10,
        'class_names': [str(i) for i in range(10)],
        'name': 'MNIST',
        'unknown_class': 10
    },
    'cifar10': {
        'num_classes': 10,
        'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck'],
        'name': 'CIFAR-10',
        'unknown_class': 10
    }
}

def infer_dataset(filepath):
    """Infer dataset type from filepath."""
    filepath = filepath.lower()
    if 'mnist' in filepath:
        return 'mnist'
    elif 'cifar10' in filepath or 'cifar-10' in filepath:
        return 'cifar10'
    else:
        raise ValueError("Could not infer dataset type from filepath. Please ensure 'mnist' or 'cifar10' is in the path.")

def load_results(filepath):
    """Load and validate results from numpy file."""
    try:
        results = np.load(filepath, allow_pickle=True).item()
        true_labels = results['true_labels']
        predicted_labels = results['predicted_labels']
        return true_labels, predicted_labels
    except Exception as e:
        raise ValueError(f"Error loading results from {filepath}: {str(e)}")

def print_confusion_matrix(conf_matrix, class_names):
    """Print confusion matrix in a readable format."""
    # Print header
    header = "True\\Pred |"
    for name in class_names:
        header += f" {name:>5} |"
    print(header)
    print("-" * len(header))
    
    # Print rows
    for i, row in enumerate(conf_matrix):
        line = f"{class_names[i]:>9} |"
        for val in row:
            line += f" {val:>5} |"
        print(line)

def analyze_results(true_labels, predicted_labels, dataset_config):
    """Analyze classification results and print statistics."""
    num_classes = dataset_config['num_classes']
    class_names = dataset_config['class_names']
    dataset_name = dataset_config['name']
    unknown_class = dataset_config['unknown_class']
    
    # Basic statistics
    total_samples = len(true_labels)
    valid_predictions = predicted_labels != unknown_class
    unknown_predictions = predicted_labels == unknown_class
    accuracy = np.mean(np.logical_and(true_labels == predicted_labels, valid_predictions))
    
    # Count unknown predictions
    total_unknown = np.sum(unknown_predictions)
    unknown_percentage = (total_unknown / total_samples) * 100
    
    # Per-class accuracy and unknown predictions
    class_accuracies = []
    unknown_per_class = []
    for i in range(num_classes):
        mask = true_labels == i
        if np.sum(mask) > 0:
            class_acc = np.mean(predicted_labels[mask][predicted_labels[mask] != unknown_class] == i)
            unknown_count = np.sum(predicted_labels[mask] == unknown_class)
            class_accuracies.append(class_acc)
            unknown_per_class.append(unknown_count)
        else:
            class_accuracies.append(0.0)
            unknown_per_class.append(0)
    
    # Compute confusion matrix for valid predictions
    valid_true = true_labels[valid_predictions]
    valid_pred = predicted_labels[valid_predictions]
    conf_mat = confusion_matrix(valid_true, valid_pred, 
                              labels=range(num_classes))
    
    # Print results
    print(f"\n{dataset_name} Classification Analysis")
    print("=" * 40)
    print(f"Total examples: {total_samples}")
    print(f"Unknown/Error predictions: {total_unknown} ({unknown_percentage:.2f}%)")
    print(f"Overall accuracy (excluding unknown): {accuracy:.4f}")
    
    print("\nPer-class Statistics:")
    print("-" * 40)
    for i, (name, acc, unk) in enumerate(zip(class_names, class_accuracies, unknown_per_class)):
        total_class = np.sum(true_labels == i)
        unk_percent = (unk / total_class * 100) if total_class > 0 else 0
        print(f"{name:>12}: Acc = {acc:.4f}, Unknown = {unk} ({unk_percent:.2f}%)")
    
    print("\nConfusion Matrix (excluding unknown predictions):")
    print("-" * 40)
    print_confusion_matrix(conf_mat, class_names)

def main():
    parser = argparse.ArgumentParser(description='Analyze classification results.')
    parser.add_argument('--filepath', required=True, 
                      help='Path to the .npy results file')
    args = parser.parse_args()
    
    # Expand user path if necessary
    filepath = os.path.expanduser(args.filepath)
    
    # Infer dataset type and get configuration
    dataset_type = infer_dataset(filepath)
    dataset_config = DATASET_CONFIGS[dataset_type]
    
    # Load and analyze results
    true_labels, predicted_labels = load_results(filepath)
    analyze_results(true_labels, predicted_labels, dataset_config)

if __name__ == "__main__":
    main()