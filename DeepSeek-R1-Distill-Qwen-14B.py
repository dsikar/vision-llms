import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Get token from environment variable and login
hf_token = os.getenv("HUGGINGFACE_API_KEY")
if not hf_token:
    raise ValueError("Please set the HUGGINGFACE_API_KEY environment variable")
login(token=hf_token)

# Model configuration
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=hf_token,
    trust_remote_code=True
)

# Initialize model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=hf_token,
    trust_remote_code=True
)

# Prepare the question
question = "What is the capital of Italy?"

# Create input for the model
inputs = tokenizer(
    question,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
).to(model.device)

# Generate response
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

# Decode and print the response
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)
