import os

import torch
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer

from data.dataset import PsychologyDataset  # Import your dataset

# Define training parameters
model_name = "mistralai/Mistral-Small-3.1"  # Updated to use Mistral Small 3.1 model
data_file = "/Users/alexpavlides/Documents/psychology_model/data/richard_summers_physchodynamictherapy.txt"  # Replace with your data file
max_length = 512
chunk_size = 1024
batch_size = 2
epochs = 1
learning_rate = 5e-5

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # setting the pad token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)  # Added torch_dtype check
model.resize_token_embeddings(len(tokenizer)) # Ensure model embeddings match tokenizer vocab size

# Create dataset and dataloader
dataset = PsychologyDataset(data_file, tokenizer, max_length, chunk_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Use GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(epochs):
    for idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Batch {idx+1}/{len(dataloader)}, Loss: {loss.item()}")

# Save the fine-tuned model
output_dir = "/Users/alexpavlides/Documents/psychology_model/fine_tuned_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Fine-tuned model saved to {output_dir}")
