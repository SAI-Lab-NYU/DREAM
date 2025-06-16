import io
import json
import base64
from io import BytesIO
from PIL import Image
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


#SPLIT TRAINING AND TESTING AT THE END OF THE FILE

target_model_id = "/models/llava-v1.6-vicuna-7b-hf"
# Load processor and model
processor = LlavaNextProcessor.from_pretrained("models/llava-v1.6-vicuna-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "models/llava-v1.6-vicuna-7b-hf", 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    device_map="auto"
)
model.eval()

# Load dataset (Hugging Face dataset)
with open("datasets/test/seed2/SEED-Bench_v2_level1_2_3.json", "r", encoding="utf-8") as f:
    records = json.load(f)
print(records["questions"][0])
ds = records["questions"]
ds = records["questions"][:10]

def load_image_path(base_path, image_path):
    try:
        img = Image.open(f"{base_path}/{image_path}")
    except Exception as e:
        print(f"Unable to open image {image_path}, error message: {e}")
        img = None  # Ensure img is assigned a value
    return img

def pil_image_to_base64_bytes(pil_image):
    """Convert PIL Image to base64 bytes for storage"""
    if pil_image is None:
        return None

    try:
        # Convert CMYK to RGB if necessary
        if pil_image.mode == 'CMYK':
            pil_image = pil_image.convert('RGB')
        
        buffer = BytesIO()
        # Save as PNG to preserve quality
        pil_image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        return image_bytes
    except Exception as e:
        print(f"Unable to process image, error message: {e}")
        return None

# Define function to decode base64 images
def load_image_base64(base64_string):
    # Note: This assumes the input is a base64 encoded string
    base64_string = base64_string.replace("\n", "")
    missing_padding = len(base64_string) % 4
    if missing_padding:
        base64_string += "=" * (4 - missing_padding)
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image

def load_image(image):
    img_file = io.BytesIO(image)
    # Open this file object with PIL
    img = Image.open(img_file)
    return img

# Define custom dataset class
class SEEDDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            # Get single data item from Hugging Face dataset (containing "image" and "question")
            record = self.dataset[idx]

            image = load_image_path("datasets/test/seed2/cc3m-image", record["data_id"])
            if image is None:
                # Skip this item if image loading failed
                return None, None, None
                
            question = record["question"]
            image_base64 = pil_image_to_base64_bytes(image)
            
            if image_base64 is None:
                # Skip this item if image processing failed
                return None, None, None

            return image, question, image_base64
        except Exception as e:
            print(f"Error processing data item {idx}: {e}")
            return None, None, None

# Custom collate function: batch construct prompts and encode
def collate_fn(batch):
    # Filter out None values from failed image processing
    valid_batch = [(img, q, img_b64) for img, q, img_b64 in batch if img is not None and q is not None and img_b64 is not None]
    
    if not valid_batch:
        return [], [], [], []
    
    # batch is a list, each element is (image, question, image_base64)
    images, questions, images_b64 = zip(*valid_batch)
    # Construct user conversations, note that each record needs to add "image" placeholder
    user_conversations = [[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": q},
                {"type": "image"}
            ]
        }
    ] for q in questions]
    # Use processor to construct generation prompts
    prompts = [processor.apply_chat_template(msg, add_generation_prompt=True) for msg in user_conversations]
    # Pass images and text to processor, return encoded model inputs
    
    return prompts, questions, list(images), list(images_b64)

# Instantiate dataset and DataLoader (set num_workers for multiprocess loading)
batch_size = 10
dataset = SEEDDataset(ds)
data_loader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=4,        # Can be adjusted based on CPU core count
    collate_fn=collate_fn,
    pin_memory=True
)

# Open output file
output_file = "no-sample-seeds-llava-v1.6-vicuna-7b.jsonl"

# Iterate through DataLoader to get data for each batch
for prompts, questions, images, images_b64 in tqdm(data_loader):
    # Skip empty batches
    if not prompts:
        continue
        
    torch.cuda.empty_cache()
    
    try:
        # Model generates responses
        inputs = processor(
            images=images, 
            text=prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            padding_side='left'
        )
        # Move data to GPU
        inputs = inputs.to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        
        # Iterate through each sample in the batch
        for j in range(len(questions)):
            try:
                decoded = processor.decode(outputs[j], skip_special_tokens=True)
                # Extract response text (if "ASSISTANT: " separator exists)
                if "ASSISTANT: " in decoded:
                    assistant_text = decoded.split("ASSISTANT: ")[1]
                else:
                    assistant_text = decoded
                final_conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": questions[j]},
                            {"type": "image"}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": assistant_text}
                        ]
                    }
                ]

                image_b64_string = base64.b64encode(images_b64[j]).decode('utf-8')
                # If you need to save the original image, you can directly save the image_base64 string
                record = {
                    "image": image_b64_string,
                    "conversation": final_conversation
                }
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Error processing output: {e}")
                continue
                
    except Exception as e:
        print(f"Error during model inference: {e}")
        continue


import json
import os

# Paths
input_file = "no-sample-seeds-llava-v1.6-vicuna-7b.jsonl"
output_dir = "processed_seed_data"
train_output_path = os.path.join(output_dir, "seed_train.jsonl")
test_output_path = os.path.join(output_dir, "seed_test.jsonl")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Read all data
with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line.strip()) for line in f if line.strip()]

# Split data
train_data = data[:1000]
test_data = data[1000:4000]

# Write train data
with open(train_output_path, "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Write test data
with open(test_output_path, "w", encoding="utf-8") as f:
    for item in test_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Split complete: {len(train_data)} train samples, {len(test_data)} test samples")