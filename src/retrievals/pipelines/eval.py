import torch
from transformers import AutoModel, AutoTokenizer
from mteb import MTEB
from datasets import disable_caching
from peft import PeftModel
import numpy as np
from copy import deepcopy

# Disable caching for datasets
disable_caching()

# Load base model and tokenizer
base_model_name = "nomic-ai/nomic-embed-text-v1"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
base_model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)

# Load the LoRA fine-tuned weights
lora_path = "/root/workspace/example_data/output"
base_model_copy = deepcopy(base_model)
fine_tuned_model = PeftModel.from_pretrained(base_model_copy, lora_path)
import random
def manually_update_lora_weights(model, scale=100.0):
    for name, param in model.named_parameters():
        if "lora_A" in name:
            print(f"Updating {name} with scale {scale}")
            with torch.no_grad():
                param.data = torch.ones_like(param.data) * scale * random.random()  # Set weights to a large value

# Update LoRA weights in the fine-tuned model

# Define a custom forward method for the fine-tuned model
def new_forward(self, **kwargs):
    kwargs.pop("output_attentions", None)
    kwargs.pop("output_hidden_states", None)
    return self.model.forward(**kwargs)

import types
fine_tuned_model.forward = types.MethodType(new_forward, fine_tuned_model)

# Ensure models are in evaluation mode
base_model.eval()
fine_tuned_model.eval()

# Move models to the appropriate device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model.to(device)
fine_tuned_model.to(device)

# Define the custom embedder
class CustomEmbedder:
    def __init__(self, model, tokenizer, device):
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
            # Extract the last hidden state and apply mean pooling
            last_hidden_state = outputs.last_hidden_state
            embeddings = torch.mean(last_hidden_state, dim=1).cpu()
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0).numpy()
manually_update_lora_weights(fine_tuned_model, scale=100.0)

# Initialize embedders for both models
base_embedder = CustomEmbedder(model=base_model, tokenizer=tokenizer, device=device)
fine_tuned_embedder = CustomEmbedder(model=fine_tuned_model, tokenizer=tokenizer, device=device)

# Step 3: Load the MTEB benchmark and specify the task
task = "STS12"
benchmark = MTEB(tasks=[task])

# Step 4: Run the benchmark on the base model
print(f"Running benchmark on task: {task} for base model")
base_evaluation = benchmark.run(model=base_embedder)

# Step 5: Run the benchmark on the fine-tuned model
print(f"Running benchmark on task: {task} for fine-tuned model")
fine_tuned_evaluation = benchmark.run(model=fine_tuned_embedder)

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