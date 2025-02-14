import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import os
import glob

# CIFAR10 class mapping
CIFAR10_CLASSES = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

# Create output directory if it doesn't exist
output_dir = 'cifar10_analysis_plots'
os.makedirs(output_dir, exist_ok=True)

# Function to process a single results file
def analyze_results(file_path):
    print(f"\nAnalyzing: {os.path.basename(file_path)}")
    
    # Load and prepare data
    results = np.load(file_path, allow_pickle=True).item()
    true_labels = np.array(results['true_labels'])
    predicted_labels = np.array(results['predicted_labels'])
    timestamp = results.get('timestamp', 'unknown')
    
    # Basic Statistics
    total_samples = len(true_labels)
    accuracy = np.mean(true_labels == predicted_labels)
    unparseable = np.sum(predicted_labels == 10)
    print(f"Timestamp: {timestamp}")
    print(f"Total samples: {total_samples}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"Unparseable predictions: {unparseable} ({unparseable/total_samples*100:.2f}%)")

    # Per-class statistics
    print("\nPer-class Statistics:")
    for class_idx, class_name in CIFAR10_CLASSES.items():
        mask = true_labels == class_idx
        total = np.sum(mask)
        if total > 0:
            class_acc = np.mean(predicted_labels[mask] == class_idx)
            class_unparseable = np.sum(predicted_labels[mask] == 10)
            print(f"\n{class_name.capitalize()} (class {class_idx}):")
            print(f"  Samples: {total}")
            print(f"  Accuracy: {class_acc:.4f}")
            print(f"  Unparseable: {class_unparseable} ({class_unparseable/total*100:.2f}%)")

    # Create plots with timestamp in filenames
    timestamp_str = f"_{timestamp}" if timestamp != 'unknown' else ""

    # Confusion Matrix
    plt.figure(figsize=(12, 10))
    conf_mat = confusion_matrix(true_labels, predicted_labels)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Timestamp: {timestamp})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # Add class labels
    plt.xticks(np.arange(len(CIFAR10_CLASSES)) + 0.5, [CIFAR10_CLASSES[i] for i in range(10)], rotation=45)
    plt.yticks(np.arange(len(CIFAR10_CLASSES)) + 0.5, [CIFAR10_CLASSES[i] for i in range(10)], rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix{timestamp_str}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Error Rate by Class
    plt.figure(figsize=(12, 6))
    error_rates = [np.mean(predicted_labels[true_labels == i] != i) for i in range(10)]
    plt.bar(range(10), error_rates)
    plt.title(f'Error Rate by Class (Timestamp: {timestamp})')
    plt.xlabel('Class')
    plt.xticks(range(10), [CIFAR10_CLASSES[i] for i in range(10)], rotation=45)
    plt.ylabel('Error Rate')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'error_rates{timestamp_str}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Normalized Confusion Matrix
    plt.figure(figsize=(12, 10))
    row_sums = conf_mat.sum(axis=1)[:, np.newaxis]
    row_sums = np.where(row_sums == 0, 1, row_sums)  # Replace zeros with ones to avoid division by zero
    conf_mat_norm = conf_mat.astype('float') / row_sums
    sns.heatmap(conf_mat_norm, annot=True, fmt='.2f', cmap='Blues')
    plt.title(f'Normalized Confusion Matrix (Timestamp: {timestamp})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(np.arange(len(CIFAR10_CLASSES)) + 0.5, [CIFAR10_CLASSES[i] for i in range(10)], rotation=45)
    plt.yticks(np.arange(len(CIFAR10_CLASSES)) + 0.5, [CIFAR10_CLASSES[i] for i in range(10)], rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'normalized_confusion_matrix{timestamp_str}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Accuracy by Class
    plt.figure(figsize=(12, 6))
    class_accuracies = []
    for i in range(10):
        mask = true_labels == i
        if np.sum(mask) > 0:
            class_accuracies.append(np.mean(predicted_labels[mask] == i))
        else:
            class_accuracies.append(0)
    plt.bar(range(10), class_accuracies)
    plt.title(f'Accuracy by Class (Timestamp: {timestamp})')
    plt.xlabel('Class')
    plt.xticks(range(10), [CIFAR10_CLASSES[i] for i in range(10)], rotation=45)
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'accuracy_by_class{timestamp_str}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Unparseable distribution plot
    if unparseable > 0:
        plt.figure(figsize=(12, 6))
        unparseable_mask = predicted_labels == 10
        unparseable_true_dist = true_labels[unparseable_mask]
        unparseable_counts = [np.sum(unparseable_true_dist == i) for i in range(10)]
        plt.bar(range(10), unparseable_counts)
        plt.title(f'Distribution of True Labels for Unparseable Predictions (Timestamp: {timestamp})')
        plt.xlabel('Class')
        plt.xticks(range(10), [CIFAR10_CLASSES[i] for i in range(10)], rotation=45)
        plt.ylabel('Count of Unparseable Predictions')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'unparseable_distribution{timestamp_str}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Calculate precision, recall, and F1 score
    precision, recall, f1, support = precision_recall_fscore_support(true_labels, predicted_labels, 
                                                                   labels=range(10))

    print("\nDetailed Metrics per Class:")
    print("\nClass\t\tPrecision\tRecall\t\tF1 Score\tSupport")
    print("-" * 70)
    for i in range(10):
        print(f"{CIFAR10_CLASSES[i]:<12}\t{precision[i]:.4f}\t\t{recall[i]:.4f}\t\t{f1[i]:.4f}\t\t{support[i]}")

    return timestamp

def main():
    results_dir = os.path.expanduser('~/archive/cifar10/results')
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Find all result files
    result_files = glob.glob(os.path.join(results_dir, 'cifar10_*.npy'))
    
    if not result_files:
        print(f"No result files found in {results_dir}")
        return

    print(f"Found {len(result_files)} result files")
    
    # Process each file
    for file_path in sorted(result_files):
        try:
            timestamp = analyze_results(file_path)
            print(f"\nPlots have been saved to the '{output_dir}' directory with timestamp {timestamp}:")
            print("1. confusion_matrix.png")
            print("2. error_rates.png")
            print("3. normalized_confusion_matrix.png")
            print("4. accuracy_by_class.png")
            if np.sum(np.load(file_path, allow_pickle=True).item()['predicted_labels'] == 10) > 0:
                print("5. unparseable_distribution.png")
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {str(e)}")

if __name__ == "__main__":
    main()