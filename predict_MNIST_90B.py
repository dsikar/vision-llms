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
MNIST_PATH = os.path.join(ARCHIVE_PATH, "mnist")
RESULTS_PATH = os.path.join(MNIST_PATH, "results")
RAW_PATH = os.path.join(MNIST_PATH, "raw")

def setup_argparse():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description='MNIST Classification with Llama Vision')
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

def extract_digit(response, debug=False):
    """Extract digit from model response."""
    try:
        # Primary pattern - what we're seeing in actual responses
        match = re.search(r'<(\d)>', response)
        if match:
            digit = int(match.group(1))
            if debug:
                print(f"Extracted digit: {digit}")
            return digit
            
        if debug:
            print("No digit found in expected format")
        return 10
    except Exception as e:
        if debug:
            print(f"Error extracting digit: {str(e)}")
        return 10

def load_previous_results(filepath):
    """Load previously saved results and determine the next index to process."""
    try:
        data = np.load(filepath, allow_pickle=True).item()
        true_labels = data['true_labels']
        predicted_labels = data['predicted_labels']
        timestamp = data.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        # Reconstruct results dictionary
        results = {
            'true_labels': list(true_labels),
            'predicted_labels': list(predicted_labels),
            'timestamp': timestamp
        }
        
        # Determine the next index to process (length of processed data)
        next_idx = len(true_labels)
        
        print(f"Loaded {next_idx} processed images. Resuming from image {next_idx}.")
        return results, next_idx
    except Exception as e:
        print(f"Error loading previous results: {str(e)}")
        return {'true_labels': [], 'predicted_labels': [], 
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")}, 0

def save_results(results_dict, save_path, set_type, timestamp):
    """Save results to file with consistent timestamp."""
    save_file = os.path.join(
        save_path,
        f'mnist_{set_type}_Llama-3.2-90B-Vision-Instruct_{timestamp}.npy'
    )
    np.save(save_file, results_dict)

def process_dataset(dataset, model, processor, set_type, timestamp, start_idx=0, results=None, debug=False):
    """Process either training or testing dataset with resume capability."""
    if results is None:
        results = {
            'true_labels': [],
            'predicted_labels': [],
            'timestamp': timestamp
        }
    
    total_images = len(dataset)
    print(f"Processing {set_type} dataset ({total_images} images)...")
    print(f"Starting from index {start_idx} with {len(results['true_labels'])} already processed images")
    
    for idx in range(start_idx, total_images):
        if idx % 100 == 0 and not debug:
            print(f"Processing image {idx}/{total_images} ({idx/total_images*100:.1f}%)")
            
        image, label = dataset[idx]
        # Convert to RGB as model expects color images
        image = image.convert('RGB')
        
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "What digit (0-9) is shown in this image? Provide your answer in <answer> tags."}
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
            
            predicted_digit = extract_digit(response, debug)
            
            if debug:
                print("\n" + "="*50)
                print(f"Image {idx}")
                print(f"True label: {label}")
                print(f"Full model response:\n{response}")
                print(f"Predicted digit: {predicted_digit}")
                print("="*50 + "\n")
                input("Press Enter to continue...")
            
            results['true_labels'].append(label)
            results['predicted_labels'].append(predicted_digit)
            
        except Exception as e:
            print(f"Error processing image {idx}: {str(e)}")
            results['true_labels'].append(label)
            results['predicted_labels'].append(10)
            
        # Save results periodically - count from start_idx for proper intervals
        if (idx - start_idx + 1) % 1000 == 0 or idx == total_images - 1:
            save_results(results, RESULTS_PATH, f"{set_type}_iteration", results['timestamp'])
    
    # Save final results
    save_results(results, RESULTS_PATH, set_type, results['timestamp'])
    
    return results['true_labels'], results['predicted_labels']

def main():
    args = setup_argparse()
    
    # Create necessary directories
    for path in [MNIST_PATH, RESULTS_PATH, RAW_PATH]:
        os.makedirs(path, exist_ok=True)
    
    # Setup model and processor
    model, processor = setup_model()
    
    # Load datasets
    print("Loading MNIST datasets...")
    train_dataset = datasets.MNIST(RAW_PATH, train=True, download=True)
    test_dataset = datasets.MNIST(RAW_PATH, train=False, download=True)
    
    if args.debug:
        print("Running in debug mode - will show detailed output for each prediction")
    
    # Process each dataset
    for dataset, set_type in [(train_dataset, 'training'), (test_dataset, 'testing')]:
        # Initialize results and starting index
        results = None
        start_idx = 0
        
        # If resuming, load previous results
        if args.resume_from and set_type in args.resume_from:
            results, start_idx = load_previous_results(args.resume_from)
            timestamp = results['timestamp']
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
        
        # Print basic statistics for this dataset
        accuracy = sum(np.array(true_labels) == np.array(pred_labels)) / len(true_labels)
        print(f"\n{set_type.capitalize()} dataset complete!")
        print(f"{set_type.capitalize()} accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
    """
    MNIST Classification with resume functionality.
    Example usage:
    python predict_MNIST_90B.py --resume-from /users/aczd097/archive/mnist/results/mnist_training_Llama-3.2-90B-Vision-Instruct_20250215_120000.npy
    """