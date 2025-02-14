import os
import torch
import numpy as np
from PIL import Image
from torchvision import datasets
from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import login
import re
import argparse
from torch.multiprocessing import Process, Manager, Queue, Lock, Value, set_start_method
import time
from datetime import datetime, timedelta

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
    return parser.parse_args()

class ProcessTracker:
    def __init__(self, total_images, save_path, set_type, timestamp):
        self.manager = Manager()
        self.lock = Lock()
        self.processed_count = Value('i', 0)
        self.total_images = total_images
        self.start_time = time.time()
        self.save_path = save_path
        self.set_type = set_type
        self.results_dict = self.manager.dict()
        self.timestamp = timestamp
        
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
                  f"ETA: {eta} | "
                  f"Last pred: {pred_label}", end="")
            
            # Save results periodically (every 50 images)
            if current_count % 50 == 0:
                self.save_results()
    
    def save_results(self):
        with self.lock:
            # Sort indices and create aligned arrays
            indices = sorted(self.results_dict.keys())
            true_labels = []
            predicted_labels = []
            
            for idx in indices:
                true_label, pred_label = self.results_dict[idx]
                true_labels.append(true_label)
                predicted_labels.append(pred_label)
            
            # Save to file with consistent timestamp
            save_file = os.path.join(
                self.save_path, 
                f'mnist_{self.set_type}_Llama-3.2-11B-Vision-Instruct_{self.timestamp}.npy'
            )
            
            # Save with complete information
            np.save(save_file, {
                'indices': indices,
                'true_labels': true_labels,
                'predicted_labels': predicted_labels,
                'processed_count': self.processed_count.value,
                'timestamp': self.timestamp
            })

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
        
        while True:
            try:
                # Get next image from queue
                item = image_queue.get(timeout=5)  # 5 second timeout
                if item is None:  # Poison pill
                    break
                    
                idx, image, label = item
                
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
                
                # Update progress
                tracker.update_progress(idx, label, predicted_digit)
                
                if debug:
                    print(f"\nProcess {process_id} - Image {idx}:")
                    print(f"True label: {label}")
                    print(f"Response: {response}")
                    print(f"Predicted: {predicted_digit}\n")
                
                # Clear CUDA cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
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
    
    # Generate timestamp for this processing run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting processing with timestamp: {timestamp}")
    
    # Initialize progress tracker with timestamp
    tracker = ProcessTracker(total_images, RESULTS_PATH, set_type, timestamp)
    
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
    # Set start method to spawn
    set_start_method('spawn', force=True)
    
    args = setup_argparse()
    
    # Create necessary directories
    for path in [MNIST_PATH, RESULTS_PATH, RAW_PATH]:
        os.makedirs(path, exist_ok=True)
    
    # Get model ID and token
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    if not hf_token:
        raise ValueError("Please set the HUGGINGFACE_API_KEY environment variable")
    
    # Determine number of processes based on available GPUs
    num_processes = torch.cuda.device_count()
    print(f"Using {num_processes} GPUs")
    
    # Load datasets
    print("Loading MNIST datasets...")
    train_dataset = datasets.MNIST(RAW_PATH, train=True, download=True)
    test_dataset = datasets.MNIST(RAW_PATH, train=False, download=True)
    
    # Process datasets in parallel
    parallel_process_dataset(train_dataset, model_id, hf_token, num_processes, 'training', args.debug)
    parallel_process_dataset(test_dataset, model_id, hf_token, num_processes, 'testing', args.debug)
    
    print("\nAll processing complete!")

if __name__ == "__main__":
    main()