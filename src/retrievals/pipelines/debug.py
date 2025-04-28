import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import numpy as np
from copy import deepcopy
import torch
from transformers import AutoModel, AutoTokenizer
from mteb import MTEB
from datasets import disable_caching
from peft import PeftModel
import numpy as np
from copy import deepcopy


# Load the base model and tokenizer
base_model_name = "nomic-ai/nomic-embed-text-v1"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
base_model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)

# Load the LoRA fine-tuned weights
lora_path = "/root/workspace/example_data/output"

# Create a deep copy of the base model
base_model_copy = deepcopy(base_model)

# Apply LoRA to the copy
fine_tuned_model = PeftModel.from_pretrained(base_model_copy, lora_path)

# Wrap the fine-tuned model
import types

def new_forward(self, **kwargs):
    # Remove unsupported arguments
    kwargs.pop("output_attentions", None)
    kwargs.pop("output_hidden_states", None)
    
    # Call the fine-tuned model's forward method
    return self.model.forward(**kwargs)

fine_tuned_model.forward = types.MethodType(new_forward, fine_tuned_model)

# Ensure models are in evaluation mode
base_model.eval()
fine_tuned_model.eval()

# Move models to the appropriate device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model.to(device)
fine_tuned_model.to(device)

import random
# Define a function to encode texts using a model
def encode_texts(model, tokenizer, texts, device):
    all_embeddings = []
    for text in texts:
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items() if key in {"input_ids", "attention_mask"}}
        with torch.no_grad():
            outputs = model(**inputs)
        # Extract the last hidden state and apply mean pooling
        last_hidden_state = outputs.last_hidden_state
        embeddings = torch.mean(last_hidden_state, dim=1).cpu().numpy()
        all_embeddings.append(embeddings)
    return np.concatenate(all_embeddings, axis=0)

def manually_update_lora_weights(model, scale=100.0):
    for name, param in model.named_parameters():
        if "lora_A" in name:
            print(f"Updating {name} with scale {scale}")
            with torch.no_grad():
                param.data = torch.ones_like(param.data) * scale * random.random()  # Set weights to a large value

# Update LoRA weights in the fine-tuned model
# manually_update_lora_weights(fine_tuned_model, scale=10000000.0)
# Define a set of test texts
test_texts = [
    "This is a test sentence.",
    "Another example sentence.",
    "The quick brown fox jumps over the lazy dog.",
    "Nomic AI is a great company."
]

# Encode texts using the base model
base_embeddings = encode_texts(base_model, tokenizer, test_texts, device)
print("Base model embeddings shape:", base_embeddings.shape)

# Encode texts using the fine-tuned model
fine_tuned_embeddings = encode_texts(fine_tuned_model, tokenizer, test_texts, device)
print("Fine-tuned model embeddings shape:", fine_tuned_embeddings.shape)

# Compute the difference between the embeddings
difference = np.abs(base_embeddings - fine_tuned_embeddings)
print("Difference between embeddings:")
print(difference)

# Compute the mean and max difference
mean_difference = np.mean(difference)
max_difference = np.max(difference)
print(f"Mean difference: {mean_difference}")
print(f"Max difference: {max_difference}")

# Check if the embeddings are significantly different
threshold = 1e-5  # Define a threshold for significant difference
if mean_difference > threshold or max_difference > threshold:
    print("The embeddings are significantly different.")
else:
    print("The embeddings are not significantly different.")



# print(fine_tuned_model)

# for name, param in fine_tuned_model.named_parameters():
#     if "lora" in name:
#         print(name, param.requires_grad, param.data)
# base_model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
for name, param in base_model.named_parameters():
    if "lora" in name:
        print(name, param.data)

from typing import List  # Import List for type hints

class CustomEmbedder:
    def __init__(self, model, tokenizer, device):
        print("Initializing CustomEmbedder...")  # Debug statement

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def encode(self, texts, batch_size=32, **kwargs):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Extract the last hidden state
            last_hidden_state = outputs.last_hidden_state
            # Apply mean pooling over the sequence dimension
            embeddings = torch.mean(last_hidden_state, dim=1).cpu()
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0).numpy()


base_embedder = CustomEmbedder(model=base_model, tokenizer=tokenizer, device=device)
fine_tuned_embedder = CustomEmbedder(model=fine_tuned_model, tokenizer=tokenizer, device=device)

# Step 3: Load the MTEB benchmark and specify the task
task = "STS13"
benchmark = MTEB(tasks=[task])
new_benchmark = MTEB(tasks=[task],use_cache=False)
base_evaluation = benchmark.run(model=base_embedder)
fine_tuned_evaluation = new_benchmark.run(model=fine_tuned_embedder,overwrite_results=True)
# # Step 4: Run the benchmark on the base model
# print(f"Running benchmark on task: {task} for base model")
# base_evaluation = benchmark.run(model=base_embedder)

# # Step 5: Run the benchmark on the fine-tuned model
# print(f"Running benchmark on task: {task} for fine-tuned model")
# fine_tuned_evaluation = benchmark.run(model=fine_tuned_embedder)

# Step 6: Inspect the results
def inspect_results(evaluation, model_name):
    print(f"Inspecting results for {model_name}:")
    for result in evaluation:
        print("\nResult object attributes (dir):")
        print(dir(result))
        print("\nResult object internal dictionary (vars):")
        print(vars(result))
        print("-" * 40)

# Inspect base model results
inspect_results(base_evaluation, "Base Model")

# Inspect fine-tuned model results
inspect_results(fine_tuned_evaluation, "Fine-Tuned Model")