import os
import torch
import numpy as np
from PIL import Image
from torchvision import datasets
from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import login
import re
import argparse

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

def process_dataset(dataset, model, processor, set_type, debug=False):
    """Process either training or testing dataset."""
    true_labels = []
    predicted_labels = []
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
            
            true_labels.append(label)
            predicted_labels.append(predicted_class)
            
        except Exception as e:
            print(f"Error processing image {idx}: {str(e)}")
            true_labels.append(label)
            predicted_labels.append(10)
            
        # Save results periodically
        if idx % 1000 == 0:
            np.save(
                os.path.join(RESULTS_PATH, f'cifar10_{set_type}_iteration_Llama-3.2-90B-Vision-Instruct.npy'),
                {'true_labels': true_labels, 'predicted_labels': predicted_labels}
            )
    
    # Save final results
    np.save(
        os.path.join(RESULTS_PATH, f'cifar10_{set_type}_iteration_Llama-3.2-90B-Vision-Instruct.npy'),
        {'true_labels': true_labels, 'predicted_labels': predicted_labels}
    )
    
    return true_labels, predicted_labels

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
