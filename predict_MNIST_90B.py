import os
import torch
import numpy as np
from PIL import Image
from torchvision import datasets
from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import login
import re
import argparse

# Path configurations remain the same...
BASE_PATH = "/users/aczd097"
ARCHIVE_PATH = os.path.join(BASE_PATH, "archive")
MNIST_PATH = os.path.join(ARCHIVE_PATH, "mnist")
RESULTS_PATH = os.path.join(MNIST_PATH, "results")
RAW_PATH = os.path.join(MNIST_PATH, "raw")

def setup_argparse():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description='MNIST Classification with Llama Vision')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
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

from datetime import datetime

def save_results(results_dict, save_path, set_type, timestamp):
    """Save results to file with consistent timestamp."""
    save_file = os.path.join(
        save_path,
        f'mnist_{set_type}_Llama-3.2-90B-Vision-Instruct_{timestamp}.npy'
    )
    np.save(save_file, results_dict)

def process_dataset(dataset, model, processor, set_type, debug=False):
    """Process either training or testing dataset."""
    results = {
        'true_labels': [],
        'predicted_labels': [],
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    total_images = len(dataset)
    print(f"Processing {set_type} dataset ({total_images} images)...")
    
    for idx in range(total_images):
        if idx % 100 == 0 and not debug:
            print(f"Processing image {idx}/{total_images}")
            
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
            
        # Save results periodically
        if idx % 1000 == 0:
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
    
    # Process datasets
    if args.debug:
        print("Running in debug mode - will show detailed output for each prediction")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting processing with timestamp: {timestamp}")
    
    # Process training dataset
    train_true, train_pred = process_dataset(train_dataset, model, processor, 'training', args.debug)
    
    # Process testing dataset
    test_true, test_pred = process_dataset(test_dataset, model, processor, 'testing', args.debug)
    
    # Print basic statistics
    print("\nProcessing complete!")
    print(f"Training accuracy: {sum(np.array(train_true) == np.array(train_pred)) / len(train_true):.4f}")
    print(f"Testing accuracy: {sum(np.array(test_true) == np.array(test_pred)) / len(test_true):.4f}")

if __name__ == "__main__":
    main()