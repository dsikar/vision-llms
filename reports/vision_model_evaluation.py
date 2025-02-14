#!/usr/bin/env python3
import numpy as np
import argparse
import os

# Dataset specific configurations
DATASET_CONFIGS = {
    'mnist': {
        'num_classes': 10,
        'class_names': [str(i) for i in range(10)],
        'name': 'MNIST'
    },
    'cifar10': {
        'num_classes': 10,
        'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck'],
        'name': 'CIFAR-10'
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

def analyze_results(true_labels, predicted_labels, dataset_config):
    """Simple analysis of classification results."""
    
    # Convert to numpy arrays of integers to handle type issues
    true_labels = np.array(true_labels, dtype=int)
    predicted_labels = np.array(predicted_labels, dtype=int)
    
    # Get dataset specific info
    class_names = dataset_config['class_names']
    dataset_name = dataset_config['name']
    
    # Sanity checks
    true_distinct = set(true_labels)
    predicted_distinct = set(predicted_labels)
    print(f"Distinct values in true_labels: {true_distinct}")
    print(f"Distinct values in predicted_labels: {predicted_distinct}")
    
    total_samples = len(true_labels)
    print(f"total_samples: {total_samples}")
    
    unknown_count = np.sum(predicted_labels == 10)
    unknown_percentage = (unknown_count / total_samples) * 100
    print(f"unknown_count: {unknown_count}, unknown_percentage: {unknown_percentage}")
    
    valid_mask = predicted_labels != 10
    print(f"np.sum(valid_mask): {np.sum(valid_mask)}")
    
    if np.any(valid_mask):
        overall_accuracy = np.mean(predicted_labels[valid_mask] == true_labels[valid_mask])
    else:
        overall_accuracy = 0.0

    print(f"\n{dataset_name} Classification Analysis")
    print("=" * 40)
    print(f"Total examples: {total_samples}")
    print(f"Unknown predictions (class 10): {unknown_count} ({unknown_percentage:.2f}%)")
    print(f"Overall accuracy (excluding unknown): {overall_accuracy:.4f}")
    
    print("\nPer-class Accuracy:")
    print("-" * 60)
    for i in range(10):
        class_mask = true_labels == i
        class_total = np.sum(class_mask)
        
        if class_total > 0:
            class_valid = np.logical_and(class_mask, valid_mask)
            class_valid_count = np.sum(class_valid)
            class_unknown = np.sum(np.logical_and(class_mask, predicted_labels == 10))
            
            if class_valid_count > 0:
                class_acc = np.mean(predicted_labels[class_valid] == i)
            else:
                class_acc = 0.0
                
            print(f"{class_names[i]:>12}: {class_acc:.4f} (Unknown: {class_unknown}/{class_total})")

def main():
    parser = argparse.ArgumentParser(description='Analyze classification results.')
    parser.add_argument('--filepath', required=True, help='Path to the .npy results file')
    args = parser.parse_args()
    
    filepath = os.path.expanduser(args.filepath)
    dataset_type = infer_dataset(filepath)
    dataset_config = DATASET_CONFIGS[dataset_type]
    
    results = np.load(filepath, allow_pickle=True).item()
    analyze_results(results['true_labels'], results['predicted_labels'], dataset_config)

if __name__ == "__main__":
    main()