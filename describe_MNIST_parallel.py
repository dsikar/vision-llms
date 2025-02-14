import os
import torch
import numpy as np
from PIL import Image
from torchvision import datasets
from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import login
import re
import argparse
from torch.multiprocessing import Process, Queue, Value, set_start_method
import time
from datetime import datetime, timedelta

# Path configurations
BASE_PATH = "/users/aczd097"
ARCHIVE_PATH = os.path.join(BASE_PATH, "archive")
MNIST_PATH = os.path.join(ARCHIVE_PATH, "mnist")
RESULTS_PATH = os.path.join(MNIST_PATH, "results")
RAW_PATH = os.path.join(MNIST_PATH, "raw")

def setup_argparse():
    parser = argparse.ArgumentParser(description='MNIST Classification with Llama Vision')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    return parser.parse_args()

def save_results(results_dict, save_path, set_type, timestamp):
    """Save results to file with consistent timestamp and image indices."""
    indices = sorted(results_dict.keys())
    
    data = {
        'indices': indices,
        'true_labels': [results_dict[idx][0] for idx in indices],
        'predicted_labels': [results_dict[idx][1] for idx in indices],
        'timestamp': timestamp
    }
    
    save_file = os.path.join(
        save_path,
        f'mnist_{set_type}_Llama-3.2-11B-Vision-Instruct_{timestamp}.npy'
    )
    np.save(save_file, data)

def extract_digit(response, debug=False):
    """Extract digit from model response."""
    try:
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

def process_image(rank, image, label, model_id, hf_token, debug=False):
    """Process a single image with the model."""
    try:
        # Initialize model for this process
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=f"cuda:{rank}",
            token=hf_token,
            local_files_only=True
        )
        
        processor = AutoProcessor.from_pretrained(
            model_id,
            token=hf_token,
            local_files_only=True
        )
        
        # Convert to RGB as model expects color images
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image.numpy(), mode='L')
        image = image.convert('RGB')
        
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "What digit (0-9) is shown in this image? Provide your answer in <answer> tags."}
            ]}
        ]
        
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
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clean up
        del model
        del processor
        
        return predicted_digit
        
    except Exception as e:
        print(f"Error in process {rank}: {str(e)}")
        return 10

def worker(rank, task_queue, result_queue, counter, total_images, model_id, hf_token, debug):
    """Worker process function."""
    try:
        while True:
            task = task_queue.get()
            if task is None:
                break
                
            idx, image, label = task
            
            predicted_digit = process_image(rank, image, label, model_id, hf_token, debug)
            result_queue.put((idx, label, predicted_digit))
            
            # Update counter
            with counter.get_lock():
                counter.value += 1
                current_count = counter.value
                
            # Calculate progress
            progress = (current_count / total_images) * 100
            print(f"\rProcessed: {current_count}/{total_images} ({progress:.1f}%) - GPU {rank} - Last pred: {predicted_digit}", end="")
            
    except Exception as e:
        print(f"\nCritical error in worker {rank}: {str(e)}")

def main():
    args = setup_argparse()
    
    # Create necessary directories
    for path in [MNIST_PATH, RESULTS_PATH, RAW_PATH]:
        os.makedirs(path, exist_ok=True)
    
    # Get model ID and token
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    if not hf_token:
        raise ValueError("Please set the HUGGINGFACE_API_KEY environment variable")
    
    # Determine number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")
    
    # Load datasets
    print("Loading MNIST datasets...")
    train_dataset = datasets.MNIST(RAW_PATH, train=True, download=True)
    test_dataset = datasets.MNIST(RAW_PATH, train=False, download=True)
    
    # Process datasets
    for dataset, set_type in [(train_dataset, 'training'), (test_dataset, 'testing')]:
        print(f"\nProcessing {set_type} dataset...")
        
        # Generate timestamp once for this dataset processing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Starting processing with timestamp: {timestamp}")
        
        # Initialize multiprocessing components
        task_queue = Queue()
        result_queue = Queue()
        counter = Value('i', 0)
        
        # Start worker processes
        processes = []
        for rank in range(num_gpus):
            p = Process(
                target=worker,
                args=(rank, task_queue, result_queue, counter, len(dataset),
                      model_id, hf_token, args.debug)
            )
            p.start()
            processes.append(p)
        
        # Add tasks to queue
        for idx in range(len(dataset)):
            image, label = dataset[idx]
            task_queue.put((idx, image, label))
        
        # Add termination signals
        for _ in range(num_gpus):
            task_queue.put(None)
        
        # Collect results
        results = {}
        completed = 0
        total_images = len(dataset)
        
        while completed < total_images:
            idx, label, pred = result_queue.get()
            results[idx] = (label, pred)
            completed += 1
            
            if completed % 50 == 0:
                save_results(results, RESULTS_PATH, set_type, timestamp)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Save final results
        save_results(results, RESULTS_PATH, set_type, timestamp)
        
        print(f"\nCompleted {set_type} dataset processing!")

if __name__ == "__main__":
    # Set start method to spawn
    set_start_method('spawn', force=True)
    main()