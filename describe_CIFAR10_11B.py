import os
import torch
import numpy as np
from PIL import Image
from torchvision import datasets
from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import login
import re
import argparse
# TODO, increase images to 720x720px
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
    return parser.parse_args()

def setup_model():
    """Initialize and return the model and processor."""
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    if not hf_token:
        raise ValueError("Please set the HUGGINGFACE_API_KEY environment variable")
    
    login(token=hf_token)
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    
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

from datetime import datetime

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
        f'cifar10_{set_type}_Llama-3.2-11B-Vision-Instruct_{timestamp}.npy'
    )
    np.save(save_file, data)

def process_dataset(dataset, model, processor, set_type, timestamp, debug=False):
    """Process either training or testing dataset."""
    results = {
        'true_labels': [],
        'predicted_labels': []
    }
    
    total_images = len(dataset)
    print(f"Processing {set_type} dataset ({total_images} images)...")
    
    # Create prompt with class options
    class_options = ", ".join(CIFAR10_CLASSES.keys())
    prompt = f"What object is shown in this image? Choose one from these options: {class_options}. Respond with only the class name."
    
    for idx in range(total_images):
        if idx % 100 == 0 and not debug:
            print(f"Processing image {idx}/{total_images}")
            
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
        if idx % 1000 == 0:
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
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting processing with timestamp: {timestamp}")
    
    # Process training dataset
    train_true, train_pred = process_dataset(train_dataset, model, processor, 'training', timestamp, args.debug)
    
    # Process testing dataset
    test_true, test_pred = process_dataset(test_dataset, model, processor, 'testing', timestamp, args.debug)
    
    # Print basic statistics
    print("\nProcessing complete!")
    print(f"Training accuracy: {sum(np.array(train_true) == np.array(train_pred)) / len(train_true):.4f}")
    print(f"Testing accuracy: {sum(np.array(test_true) == np.array(test_pred)) / len(test_true):.4f}")

if __name__ == "__main__":
    main()
