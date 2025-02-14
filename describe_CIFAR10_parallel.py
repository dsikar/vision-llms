import os
import torch
import numpy as np
from PIL import Image
from torchvision import datasets
from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import login
import argparse
from torch.multiprocessing import Process, Manager, Queue, Lock, Value
import time
from datetime import datetime, timedelta
import math
from tqdm import tqdm

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

class ProcessTracker:
    def __init__(self, total_images, save_path, set_type):
        self.manager = Manager()
        self.lock = Lock()
        self.processed_count = Value('i', 0)
        self.total_images = total_images
        self.start_time = time.time()
        self.save_path = save_path
        self.set_type = set_type
        self.results_dict = self.manager.dict()
        
    def update_progress(self, idx, true_label, pred_label):
        with self.lock:
            # Update results
            self.results_dict[idx] = (true_label, pred_label)
            
            # Update counter
            with self.processed_count.get_lock():
                self.processed_count.value += 1
                current_count = self.processed_count.value
            
            # Calculate progress metrics
            elapsed_time = time.time() - self.start_time
            images_per_second = current_count / elapsed_time
            remaining_images = self.total_images - current_count
            eta_seconds = remaining_images / images_per_second if images_per_second > 0 else 0
            
            # Format ETA
            eta = str(timedelta(seconds=int(eta_seconds)))
            
            # Print progress with detailed metrics
            print(f"\rProcessed: {current_count}/{self.total_images} | "
                  f"Progress: {(current_count/self.total_images)*100:.1f}% | "
                  f"Speed: {images_per_second:.2f} img/s | "
                  f"ETA: {eta}", end="")
            
            # Save results periodically (every 50 images)
            if current_count % 50 == 0:
                self.save_results()
    
    def save_results(self):
        with self.lock:
            # Convert manager dict to numpy arrays
            indices = sorted(self.results_dict.keys())
            true_labels = []
            predicted_labels = []
            
            for idx in indices:
                true_label, pred_label = self.results_dict[idx]
                true_labels.append(true_label)
                predicted_labels.append(pred_label)
            
            # Save to file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = os.path.join(
                self.save_path, 
                f'cifar10_{self.set_type}_iteration_Llama-3.2-11B-Vision-Instruct_{timestamp}.npy'
            )
            np.save(save_file, {
                'true_labels': true_labels,
                'predicted_labels': predicted_labels,
                'processed_count': self.processed_count.value
            })

def worker_process(process_id, image_queue, model_id, hf_token, tracker, debug=False):
    """Worker process for parallel image processing."""
    try:
        # Initialize model and processor for this process
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=f"cuda:{process_id}" if torch.cuda.is_available() else "cpu",
            token=hf_token,
            local_files_only=True
        )
        
        processor = AutoProcessor.from_pretrained(
            model_id,
            token=hf_token,
            local_files_only=True
        )
        
        class_options = ", ".join(CIFAR10_CLASSES.keys())
        prompt = f"What object is shown in this image? Choose one from these options: {class_options}. Respond with only the class name."
        
        while True:
            try:
                # Get next image from queue
                item = image_queue.get(timeout=5)  # 5 second timeout
                if item is None:  # Poison pill
                    break
                    
                idx, image, label = item
                
                # Process image
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(image)
                
                messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
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
                
                predicted_class = extract_class(response, debug)
                
                # Update progress
                tracker.update_progress(idx, label, predicted_class)
                
            except Queue.Empty:
                continue
            except Exception as e:
                print(f"\nError in worker {process_id} processing image {idx}: {str(e)}")
                tracker.update_progress(idx, label, 10)  # Use error class
                
    except Exception as e:
        print(f"\nCritical error in worker {process_id}: {str(e)}")

def parallel_process_dataset(dataset, model_id, hf_token, num_processes, set_type, debug=False):
    """Process dataset using multiple processes."""
    total_images = len(dataset)
    
    # Initialize progress tracker
    tracker = ProcessTracker(total_images, RESULTS_PATH, set_type)
    
    # Create image queue
    image_queue = Queue(maxsize=num_processes * 2)
    
    # Start worker processes
    processes = []
    for i in range(num_processes):
        p = Process(
            target=worker_process,
            args=(i, image_queue, model_id, hf_token, tracker, debug)
        )
        p.start()
        processes.append(p)
    
    # Feed images to queue
    print(f"\nProcessing {set_type} dataset with {num_processes} processes...")
    for idx in range(total_images):
        image, label = dataset[idx]
        image_queue.put((idx, image, label))
    
    # Add poison pills to stop workers
    for _ in range(num_processes):
        image_queue.put(None)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Save final results
    tracker.save_results()
    print(f"\nCompleted processing {set_type} dataset!")

def main():
    args = setup_argparse()
    
    # Create necessary directories
    for path in [CIFAR10_PATH, RESULTS_PATH, RAW_PATH]:
        os.makedirs(path, exist_ok=True)
    
    # Get model ID and token
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    if not hf_token:
        raise ValueError("Please set the HUGGINGFACE_API_KEY environment variable")
    
    # Determine number of processes based on available GPUs or CPU cores
    num_processes = torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count()
    print(f"Using {num_processes} processes")
    
    # Load datasets
    print("Loading CIFAR10 datasets...")
    train_dataset = datasets.CIFAR10(RAW_PATH, train=True, download=True)
    test_dataset = datasets.CIFAR10(RAW_PATH, train=False, download=True)
    
    # Process datasets in parallel
    parallel_process_dataset(train_dataset, model_id, hf_token, num_processes, 'training', args.debug)
    parallel_process_dataset(test_dataset, model_id, hf_token, num_processes, 'testing', args.debug)

if __name__ == "__main__":
    main()