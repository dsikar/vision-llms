import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import os

# Create output directory if it doesn't exist
output_dir = 'mnist_analysis_plots'
os.makedirs(output_dir, exist_ok=True)

# Load and prepare data
file_path = os.path.expanduser('~/archive/mnist/results/mnist_training_iteration_Llama-3.2-11B-Vision-Instruct.npy')
results = np.load(file_path, allow_pickle=True).item()
true_labels = results['true_labels']
predicted_labels = results['predicted_labels']

# Basic Statistics
total_samples = len(true_labels)
accuracy = np.mean(true_labels == predicted_labels)
unparseable = np.sum(predicted_labels == 10)
print(f"Total samples: {total_samples}")
print(f"Overall accuracy: {accuracy:.4f}")
print(f"Unparseable predictions: {unparseable} ({unparseable/total_samples*100:.2f}%)")

# Print per-class statistics
print("\nPer-class Statistics:")
for digit in range(10):
    mask = true_labels == digit
    total = np.sum(mask)
    if total > 0:
        class_acc = np.mean(predicted_labels[mask] == digit)
        class_unparseable = np.sum(predicted_labels[mask] == 10)
        print(f"\nDigit {digit}:")
        print(f"  Samples: {total}")
        print(f"  Accuracy: {class_acc:.4f}")
        print(f"  Unparseable: {class_unparseable} ({class_unparseable/total*100:.2f}%)")

# Confusion Matrix
plt.figure(figsize=(12, 10))
conf_mat = confusion_matrix(true_labels, predicted_labels)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# Error Rate by Digit
plt.figure(figsize=(10, 6))
error_rates = [np.mean(predicted_labels[true_labels == i] != i) for i in range(10)]
plt.bar(range(10), error_rates)
plt.title('Error Rate by Digit')
plt.xlabel('Digit')
plt.ylabel('Error Rate')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'error_rates.png'), dpi=300, bbox_inches='tight')
plt.close()

# Normalized Confusion Matrix
plt.figure(figsize=(12, 10))
row_sums = conf_mat.sum(axis=1)[:, np.newaxis]
row_sums = np.where(row_sums == 0, 1, row_sums)  # Replace zeros with ones to avoid division by zero
conf_mat_norm = conf_mat.astype('float') / row_sums
sns.heatmap(conf_mat_norm, annot=True, fmt='.2f', cmap='Blues')
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'normalized_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# Accuracy by Digit
plt.figure(figsize=(10, 6))
class_accuracies = []
for i in range(10):
    mask = true_labels == i
    if np.sum(mask) > 0:
        class_accuracies.append(np.mean(predicted_labels[mask] == i))
    else:
        class_accuracies.append(0)
plt.bar(range(10), class_accuracies)
plt.title('Accuracy by Digit')
plt.xlabel('Digit')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'accuracy_by_digit.png'), dpi=300, bbox_inches='tight')
plt.close()

# Only create unparseable distribution plot if there are unparseable predictions
if unparseable > 0:
    plt.figure(figsize=(10, 6))
    unparseable_mask = predicted_labels == 10
    unparseable_true_dist = true_labels[unparseable_mask]
    unparseable_counts = [np.sum(unparseable_true_dist == i) for i in range(10)]
    plt.bar(range(10), unparseable_counts)
    plt.title('Distribution of True Labels for Unparseable Predictions')
    plt.xlabel('Digit')
    plt.ylabel('Count of Unparseable Predictions')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'unparseable_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Calculate precision, recall, and F1 score
precision, recall, f1, support = precision_recall_fscore_support(true_labels, predicted_labels, 
                                                               labels=range(10))

print("\nDetailed Metrics per Class:")
print("\nClass\tPrecision\tRecall\t\tF1 Score\tSupport")
print("-" * 60)
for i in range(10):
    print(f"{i}\t{precision[i]:.4f}\t\t{recall[i]:.4f}\t\t{f1[i]:.4f}\t\t{support[i]}")

print(f"\nPlots have been saved to the '{output_dir}' directory:")
print("1. confusion_matrix.png")
print("2. error_rates.png")
print("3. normalized_confusion_matrix.png")
print("4. accuracy_by_digit.png")
if unparseable > 0:
    print("5. unparseable_distribution.png")
