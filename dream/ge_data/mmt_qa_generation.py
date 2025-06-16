import json
import base64
from io import BytesIO
from PIL import Image
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

processor = LlavaNextProcessor.from_pretrained("/scratch/yh5961/models/llava-v1.6-vicuna-13b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "/scratch/yh5961/models/llava-v1.6-vicuna-13b-hf", 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    device_map="auto"
)
model.eval()

ds = load_dataset(
    path="/scratch/yh5961/data/MMT-Bench", 
    data_files="/scratch/yh5961/data/MMT-Bench/MMT-Bench_ALL.tsv", 
    split="train[4130:16000]"
)

def load_image(base64_string):
    base64_string = base64_string.replace("\n", "")
    missing_padding = len(base64_string) % 4
    if missing_padding:
        base64_string += "=" * (4 - missing_padding)
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image

class MMTBenchDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        record = self.dataset[idx]
        image_base64 = record["image"]
        question = record["question"]
        image = load_image(image_base64)
        return image, question, image_base64  

def collate_fn(batch):
    images, questions, images_b64 = zip(*batch)
    user_conversations = [[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": q},
                {"type": "image"}
            ]
        }
    ] for q in questions]
    prompts = [processor.apply_chat_template(msg, add_generation_prompt=True) for msg in user_conversations]
    
    return prompts, questions, list(images), list(images_b64)

batch_size = 10
dataset = MMTBenchDataset(ds)
data_loader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=4,       
    collate_fn=collate_fn,
    pin_memory=True
)

output_file = "no-sample-mmt-bench-llava-v1.6-vicuna-13b.jsonl"

for prompts, questions, images, images_b64 in tqdm(data_loader):
    torch.cuda.empty_cache()
    inputs = processor(
        images=images, 
        text=prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        padding_side='left'
    )
    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    with open(output_file, "a", encoding="utf-8") as f:
        for j in range(len(questions)):
            decoded = processor.decode(outputs[j], skip_special_tokens=True)
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
            record = {
                "image": images_b64[j],
                "conversation": final_conversation
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")