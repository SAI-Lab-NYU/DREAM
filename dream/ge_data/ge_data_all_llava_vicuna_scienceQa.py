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

# 加载 processor 和模型
processor = LlavaNextProcessor.from_pretrained("/home/asperger/models/llava-v1.6-vicuna-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "/home/asperger/models/llava-v1.6-vicuna-7b-hf", 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    device_map="auto"
)
model.eval()

# 加载数据集（Hugging Face 数据集）
ds = load_dataset(
    "parquet", 
    data_files={'train': '/home/asperger/datasets/ScienceQA/data/train-00000-of-00001-1028f23e353fbe3e.parquet', 'validation': '/home/asperger/datasets/ScienceQA/data/validation-00000-of-00001-6c7328ff6c84284c.parquet'},
    split='train'
)

ds = ds.filter(lambda record: record["image"] is not None)


# 定义解码 base64 图像的函数
def load_image_base64(base64_string):
    # 注意：这里假设传入的是 base64 编码的字符串
    base64_string = base64_string.replace("\n", "")
    missing_padding = len(base64_string) % 4
    if missing_padding:
        base64_string += "=" * (4 - missing_padding)
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image

def load_image(image):
    img_file = io.BytesIO(image)
    # 用 PIL 打开这个文件对象
    img = Image.open(img_file)
    return img

# 定义自定义数据集类
class MMTBenchDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 从 Hugging Face 数据集中获取单条数据（包含 "image" 和 "question"）
        record = self.dataset[idx]
        image_base64 = record["image"]["bytes"]
        image = load_image(image_base64)
        question = record["question"]
        # 将 base64 字符串转换为 PIL.Image 对象
        
        return image, question, image_base64  # image_base64 备用（写入文件时可能需要）

# 自定义 collate 函数：批量构造 prompt 并编码
def collate_fn(batch):
    # batch 是一个列表，每个元素为 (image, question, image_base64)
    images, questions, images_b64 = zip(*batch)
    # 构造用户对话，注意每条记录均需要添加 "image" 占位符
    user_conversations = [[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": q},
                {"type": "image"}
            ]
        }
    ] for q in questions]
    # 利用 processor 构造生成 prompt
    prompts = [processor.apply_chat_template(msg, add_generation_prompt=True) for msg in user_conversations]
    # 将图像和文本传入 processor，返回编码后的模型输入
    
    return prompts, questions, list(images), list(images_b64)

# 实例化数据集和 DataLoader（设置 num_workers 实现多进程加载）
batch_size = 10
dataset = MMTBenchDataset(ds)
data_loader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=4,        # 可根据 CPU 核数进行调整
    collate_fn=collate_fn,
    pin_memory=True
)

# 打开输出文件
output_file = "no-sample-scienceqa-bench-llava-v1.6-vicuna-7b.jsonl"

# 遍历 DataLoader 获取每个批次的数据
for prompts, questions, images, images_b64 in tqdm(data_loader):
    torch.cuda.empty_cache()
        # 模型生成回答
    inputs = processor(
        images=images, 
        text=prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        padding_side='left'
    )
    # 将数据移动到 GPU
    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    # 遍历批次中的每个样本
    for j in range(len(questions)):
        decoded = processor.decode(outputs[j], skip_special_tokens=True)
        # 提取回答文本（如果存在 "ASSISTANT: " 分隔符）
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
        # 如果需要保存原始图像，可以直接保存 image_base64 字符串
        record = {
            "image": images_b64[j],
            "conversation": final_conversation
        }
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")