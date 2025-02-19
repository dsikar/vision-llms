import os
import torch
import numpy as np
from PIL import Image
from torchvision import datasets
from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import login
import re
import argparse
from datetime import datetime

# Path configurations
BASE_PATH = "/users/aczd097"
ARCHIVE_PATH = os.path.join(BASE_PATH, "archive")
CIFAR10_PATH = os.path.join(ARCHIVE_PATH, "cifar10")
RESULTS_PATH = os.path.join(CIFAR10_PATH, "results")
RAW_PATH = os.path.join(CIFAR10_PATH, "raw")

# CIFAR10 class mapping
CIFAR10_CLASSES = {
    'airplane': 0,
    'automobile': 1,
    'bird': 2,
    'cat': 3,
    'deer': 4,
    'dog': 5,
    'frog': 6,
    'horse': 7,
    'ship': 8,
    'truck': 9
}

# Reverse mapping for debug output
CIFAR10_IDX_TO_CLASS = {v: k for k, v in CIFAR10_CLASSES.items()}

def setup_argparse():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description='CIFAR10 Classification with Llama Vision')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--resume-from', type=str, help='Path to the saved results file to resume from')
    return parser.parse_args()

def setup_model():
    """Initialize and return the model and processor."""
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    if not hf_token:
        raise ValueError("Please set the HUGGINGFACE_API_KEY environment variable")
    
    login(token=hf_token)
    model_id = "meta-llama/Llama-3.2-90B-Vision-Instruct"
    
    print("Loading model and processor...")
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        local_files_only=True
    )
    
    processor = AutoProcessor.from_pretrained(
        model_id,
        token=hf_token,
        local_files_only=True
    )
    
    return model, processor

def extract_class(response, debug=False):
    """Extract class from model response and map to CIFAR10 index."""
    try:
        # Split by assistant marker and double newline
        parts = response.split('assistant<|end_header_id|>\n\n')
        if len(parts) < 2:
            if debug:
                print("No assistant response found")
            return 10
            
        # Take the part after the split and clean it up
        class_name = parts[1].split('<|eot_id|>')[0].strip().rstrip('.').lower()
        
        if debug:
            print(f"Found class name: '{class_name}'")
            print(f"Class index: {CIFAR10_CLASSES.get(class_name, 10)}")
            
        return CIFAR10_CLASSES.get(class_name, 10)
        
    except Exception as e:
        if debug:
            print(f"Error: {str(e)}")
        return 10

def load_previous_results(filepath):
    """Load previously saved results and determine the next index to process."""
    try:
        data = np.load(filepath, allow_pickle=True).item()
        indices = data['indices']
        true_labels = data['true_labels']
        predicted_labels = data['predicted_labels']
        timestamp = data.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        # Reconstruct results dictionary
        results = {
            'true_labels': true_labels,
            'predicted_labels': predicted_labels
        }
        
        # Find the last processed index
        last_idx = max(indices) if indices else -1
        next_idx = last_idx + 1
        
        print(f"Loaded {len(indices)} processed images. Resuming from image {next_idx}")
        return results, next_idx, timestamp
    except Exception as e:
        print(f"Error loading previous results: {str(e)}")
        return {'true_labels': [], 'predicted_labels': []}, 0, datetime.now().strftime("%Y%m%d_%H%M%S")

def save_results(results_dict, save_path, set_type, timestamp):
    """Save results to file with consistent timestamp."""
    indices = sorted(list(range(len(results_dict['true_labels']))))
    
    data = {
        'indices': indices,
        'true_labels': results_dict['true_labels'],
        'predicted_labels': results_dict['predicted_labels'],
        'timestamp': timestamp
    }
    
    save_file = os.path.join(
        save_path,
        f'cifar10_{set_type}_Llama-3.2-90B-Vision-Instruct_{timestamp}.npy'
    )
    np.save(save_file, data)

def process_dataset(dataset, model, processor, set_type, timestamp, start_idx=0, results=None, debug=False):
    """Process either training or testing dataset with resume capability."""
    if results is None:
        results = {
            'true_labels': [],
            'predicted_labels': []
        }
    
    total_images = len(dataset)
    print(f"Processing {set_type} dataset ({total_images} images)...")
    print(f"Starting from index {start_idx} with {len(results['true_labels'])} already processed images")
    
    # Create prompt with class options
    class_options = ", ".join(CIFAR10_CLASSES.keys())
    prompt = f"What object is shown in this image? Choose one from these options: {class_options}. Respond with only the class name."
    
    for idx in range(start_idx, total_images):
        if idx % 100 == 0 and not debug:
            print(f"Processing image {idx}/{total_images} ({idx/total_images*100:.1f}%)")
            
        image, label = dataset[idx]
        # Convert PIL image to RGB
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        
        try:
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)
            
            output = model.generate(**inputs, max_new_tokens=30)
            response = processor.decode(output[0])
            
            predicted_class = extract_class(response, debug)
            
            if debug:
                print("\n" + "="*50)
                print(f"Image {idx}")
                print(f"True label: {label} ({CIFAR10_IDX_TO_CLASS[label]})")
                print(f"Full model response:\n{response}")
                print(f"Predicted class: {predicted_class} ({CIFAR10_IDX_TO_CLASS.get(predicted_class, 'unknown')})")
                print("="*50 + "\n")
                input("Press Enter to continue...")
            
            results['true_labels'].append(label)
            results['predicted_labels'].append(predicted_class)
            
        except Exception as e:
            print(f"Error processing image {idx}: {str(e)}")
            results['true_labels'].append(label)
            results['predicted_labels'].append(10)
            
        # Save results periodically
        if (idx - start_idx + 1) % 1000 == 0 or idx == total_images - 1:
            save_results(results, RESULTS_PATH, f"{set_type}_iteration", timestamp)
    
    # Save final results
    save_results(results, RESULTS_PATH, set_type, timestamp)
    
    return results['true_labels'], results['predicted_labels']

def main():
    args = setup_argparse()
    
    # Create necessary directories
    for path in [CIFAR10_PATH, RESULTS_PATH, RAW_PATH]:
        os.makedirs(path, exist_ok=True)
    
    # Setup model and processor
    model, processor = setup_model()
    
    # Load datasets
    print("Loading CIFAR10 datasets...")
    train_dataset = datasets.CIFAR10(RAW_PATH, train=True, download=True)
    test_dataset = datasets.CIFAR10(RAW_PATH, train=False, download=True)
    
    if args.debug:
        print("Running in debug mode - will show detailed output for each prediction")
    
    # Process each dataset
    for dataset, set_type in [(train_dataset, 'training'), (test_dataset, 'testing')]:
        # Initialize results and starting index
        results = None
        start_idx = 0
        
        # If resuming, load previous results
        if args.resume_from and set_type in args.resume_from:
            results, start_idx, timestamp = load_previous_results(args.resume_from)
            print(f"Resuming {set_type} dataset from index {start_idx} with timestamp {timestamp}")
        else:
            # Generate new timestamp for fresh start
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"Starting new processing with timestamp: {timestamp}")
        
        # Process dataset
        true_labels, pred_labels = process_dataset(
            dataset, model, processor, set_type, timestamp,
            start_idx=start_idx, results=results, debug=args.debug
        )
        
        # Print basic statistics
        accuracy = sum(np.array(true_labels) == np.array(pred_labels)) / len(true_labels)
        print(f"\n{set_type.capitalize()} dataset complete!")
        print(f"{set_type.capitalize()} accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
    """
    CIFAR10 Classification with resume functionality.
    Example usage:
    python describe_CIFAR10_90B.py --resume-from /users/aczd097/archive/cifar10/results/cifar10_training_Llama-3.2-90B-Vision-Instruct_20250215_120000.npy
    """