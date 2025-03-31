# data/dataset.py
import os

import torch
from torch.utils.data import Dataset


class PsychologyDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length, chunk_size):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.data = self.load_and_chunk_data(file_path)

    def load_and_chunk_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Split the text into smaller chunks of chunk_size
        chunks = [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        return chunks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoded_text = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded_text['input_ids'].flatten()
        attention_mask = encoded_text['attention_mask'].flatten()

        # Create labels for next token prediction (shift input_ids by one)
        labels = torch.cat((input_ids[1:], torch.tensor([self.tokenizer.eos_token_id])))

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels  # Add labels for next token prediction
        }

# Example usage:
if __name__ == '__main__':
    # Create a dummy text file for testing
    dummy_file_path = 'dummy_psychology_text.txt'
    with open(dummy_file_path, 'w', encoding='utf-8') as f:
        f.write("This is the first example. This is the second example. And this is a longer sentence to test the chunking.")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # Replace with your model
    tokenizer.pad_token = tokenizer.eos_token # set pad token to eos token
    max_length = 128
    chunk_size = 512
    dataset = PsychologyDataset(dummy_file_path, tokenizer, max_length, chunk_size)
    sample = dataset[0]
    print("Input IDs shape:", sample['input_ids'].shape)
    print("Attention Mask shape:", sample['attention_mask'].shape)
    print("Labels shape:", sample['labels'].shape)

    # Clean up the dummy file
    os.remove(dummy_file_path)
