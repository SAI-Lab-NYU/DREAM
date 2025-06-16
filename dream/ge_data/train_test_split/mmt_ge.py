import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
import os
import tqdm
from PIL import Image
import base64
from io import BytesIO
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import requests

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "models/llava-v1.6-vicuna-7b-hf"
processor = LlavaNextProcessor.from_pretrained(model_id)

model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16) 
model.to(DEVICE)

# Define paths
data_path = "datasets/test/mmt/MMT-Bench_ALL_MI.tsv"
output_dir = "processed_data"
train_output_path = os.path.join(output_dir, "train.jsonl")
test_output_path = os.path.join(output_dir, "test.jsonl")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def load_image_from_base64(base64_string):
    """Load image from base64 string"""
    base64_string = base64_string.replace("\n", "")
    missing_padding = len(base64_string) % 4
    if missing_padding:
        base64_string += "=" * (4 - missing_padding)
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image
    
def main():
    print("Loading dataset...")
    # Load data with pandas
    data = pd.read_csv(data_path, delimiter='\t')
    
    # Check total number of samples
    total_samples = len(data)
    print(f"Total samples in dataset: {total_samples}")
    
    # Calculate how many samples we need
    train_size = min(1000, total_samples - 100)
    test_size = min(3000, total_samples - train_size)
    
    print(f"Splitting into {train_size} training samples and {test_size} test samples")
    
    # Create indices for train/test split
    indices = np.arange(total_samples)
    
    # Perform the split
    train_indices, test_indices = train_test_split(
        indices, 
        train_size=train_size,
        test_size=test_size,
        random_state=42  # for reproducibility
    )
    
    # Process and save training data
    process_and_save_split(data, train_indices, train_output_path, "training")
    
    # Process and save test data
    process_and_save_split(data, test_indices, test_output_path, "test")
    
    print(f"Processing complete. Files saved to {train_output_path} and {test_output_path}")

def process_and_save_split(data, indices, output_path, split_name):
    """Process and save a data split to JSONL format"""
    print(f"Processing {len(indices)} {split_name} samples...")
    
    with open(output_path, 'w') as f:
        for i, idx in enumerate(tqdm.tqdm(indices)):
            # Get the row
            row = data.iloc[idx]

            # Format the conversation
            messages = [
                {"role": "user", 
                 "content": [
                    {"type": "text", "text": row['question']},
                    {"type": "image"}
                ]}
            ]

            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[load_image_from_base64(row['image'])], return_tensors="pt")
            inputs = inputs.to(DEVICE)

            # Generate outputs
            generated_ids = model.generate(**inputs, max_new_tokens=500, do_sample=False, use_cache=True, pad_token_id=processor.tokenizer.eos_token_id)
            generated_texts = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )

            full_response = generated_texts[0]
            # Find where the assistant's response starts
            assistant_prefix = "ASSISTANT:"
            if assistant_prefix in full_response:
                assistant_response = full_response.split(assistant_prefix)[1].strip()
            else:
                assistant_response = full_response  # Fallback if format is unexpected

            conversation = [
                {"role": "user", "content": [
                    {"type": "text", "text": row['question']},
                    {"type": "image"},
                ]},

                {"role": "assistant", "content": [
                    {"type": "text", "text": assistant_response}
                ]},
            ]

            # Create the JSONL entry
            entry = {
                "idx": int(idx),
                "image": row['image'],  # This is the raw base64 image from dataset
                "conversation": conversation
            }

            # Write to file
            f.write(json.dumps(entry) + '\n')


main()