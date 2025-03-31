import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load fine-tuned model
model_path = "/Users/alexpavlides/Documents/psychology_model/fine_tuned_model"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Sample psychology prompts for testing
test_prompts = [
    "The therapeutic alliance is defined as",
    "Cognitive-behavioral therapy works by",
    "In psychodynamic theory, transference refers to"
]

# Generate completions
for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}\n")