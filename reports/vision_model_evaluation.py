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

def print_confusion_matrix(true_labels, pred_labels, class_names):
    """Print confusion matrix for valid predictions."""
    # Create matrix
    n_classes = len(class_names)
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    # Fill matrix
    for t, p in zip(true_labels, pred_labels):
        conf_matrix[t][p] += 1
    
    # Print matrix
    print("\nConfusion Matrix (excluding unknown predictions):")
    print("-" * 60)
    
    # Header
    print("True\\Pred |", end="")
    for name in class_names:
        print(f" {name:>4} |", end="")
    print("\n" + "-" * (9 + 7*len(class_names)))
    
    # Rows
    for i in range(n_classes):
        print(f"{class_names[i]:>8} |", end="")
        for j in range(n_classes):
            print(f" {conf_matrix[i][j]:>4} |", end="")
        print()

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
    unknown_count = np.sum(predicted_labels == 10)
    unknown_percentage = (unknown_count / total_samples) * 100
    valid_mask = predicted_labels != 10
    
    # Overall statistics
    valid_predictions = np.sum(valid_mask)
    if valid_predictions > 0:
        correct_predictions = np.sum(np.logical_and(predicted_labels == true_labels, valid_mask))
        overall_accuracy = correct_predictions / total_samples
        accuracy_when_predicted = correct_predictions / valid_predictions
    else:
        overall_accuracy = 0.0
        accuracy_when_predicted = 0.0

    print(f"\n{dataset_name} Classification Analysis")
    print("=" * 60)
    print(f"Total examples: {total_samples}")
    print(f"Overall success rate: {overall_accuracy:.2%} ({correct_predictions}/{total_samples})")
    print(f"Prediction rate: {valid_predictions/total_samples:.2%} ({valid_predictions}/{total_samples})")
    print(f"Accuracy when predicted: {accuracy_when_predicted:.2%} ({correct_predictions}/{valid_predictions})")
    print(f"Unknown predictions: {unknown_percentage:.1f}% ({unknown_count}/{total_samples})")
    
    print("\nPer-class Statistics:")
    print("-" * 60)
    for i in range(10):
        class_mask = true_labels == i
        class_total = np.sum(class_mask)
        
        if class_total > 0:
            class_valid = np.logical_and(class_mask, valid_mask)
            valid_count = np.sum(class_valid)
            unknown_count = class_total - valid_count
            correct_count = np.sum(np.logical_and(class_valid, predicted_labels == i))
            
            overall_success_rate = correct_count / class_total
            prediction_rate = valid_count / class_total
            accuracy_when_predicted = correct_count / valid_count if valid_count > 0 else 0.0
            unknown_rate = unknown_count / class_total
            
            print(f"{class_names[i]:>12}: {overall_success_rate:.1%} overall ({correct_count}/{class_total}), "
                  f"{accuracy_when_predicted:.1%} when predicted ({correct_count}/{valid_count}), "
                  f"{unknown_rate:.1%} unknown ({unknown_count}/{class_total})")
    
    # Print confusion matrix for valid predictions only
    if np.any(valid_mask):
        true_valid = true_labels[valid_mask]
        pred_valid = predicted_labels[valid_mask]
        print_confusion_matrix(true_valid, pred_valid, class_names)

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