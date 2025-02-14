#!/usr/bin/env python3
import numpy as np
import argparse
import os

def analyze_results(true_labels, predicted_labels):
    """Simple analysis of classification results."""
    total_samples = len(true_labels)
    
    # Count unknown predictions (class 10)
    unknown_count = np.sum(predicted_labels == 10)
    unknown_percentage = (unknown_count / total_samples) * 100
    
    # Calculate accuracy excluding unknown predictions
    valid_mask = predicted_labels != 10
    valid_predictions = predicted_labels[valid_mask]
    valid_true = true_labels[valid_mask]
    
    if len(valid_predictions) > 0:
        overall_accuracy = np.mean(valid_predictions == valid_true)
    else:
        overall_accuracy = 0.0

    print("\nClassification Analysis")
    print("=" * 40)
    print(f"Total examples: {total_samples}")
    print(f"Unknown predictions (class 10): {unknown_count} ({unknown_percentage:.2f}%)")
    print(f"Overall accuracy (excluding unknown): {overall_accuracy:.4f}")
    
    print("\nPer-class Accuracy:")
    print("-" * 40)
    for i in range(10):  # Assuming 10 classes
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
                
            print(f"Class {i:>2}: {class_acc:.4f} (Unknown: {class_unknown}/{class_total})")

def main():
    parser = argparse.ArgumentParser(description='Analyze classification results.')
    parser.add_argument('--filepath', required=True, help='Path to the .npy results file')
    args = parser.parse_args()
    
    results = np.load(os.path.expanduser(args.filepath), allow_pickle=True).item()
    analyze_results(results['true_labels'], results['predicted_labels'])

if __name__ == "__main__":
    main()