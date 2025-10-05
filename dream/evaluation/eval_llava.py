"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import base64
import io
from io import BytesIO
import torch
import os
import io

#from fastchat.utils import str_to_torch_dtype
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time

import argparse

import importlib

from fastchat.model import get_conversation_template
from transformers import LlavaNextProcessor
from PIL import Image
import requests

import re

def truncate_list(lst, num):
    if num not in lst:
        return lst


    first_index = lst.index(num)


    return lst[:first_index + 1]


def find_next_non_image_token(input_ids, image_token_ids):
    image_indices = torch.where(torch.tensor([id in image_token_ids for id in input_ids]))[0]

    if len(image_indices) == 0:
        return -1 

    last_image_index = image_indices[-1].item()

    for i in range(last_image_index + 1, len(input_ids)):
        if input_ids[i].item() not in image_token_ids:
            return i 

    return -1

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ea-model-path",
    type=str, required=True,
    help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
)
parser.add_argument("--base-model-path", type=str, default="/home/asperger/models/llava-v1.6-vicuna-7b-hf",
                    help="path of basemodel, huggingface project or local path")
parser.add_argument(
    "--load-in-8bit", action="store_true", help="Use 8-bit quantization"
)
parser.add_argument(
    "--load-in-4bit", action="store_true", help="Use 4-bit quantization"
)
parser.add_argument("--model-type", type=str, default="llama-3-instruct",choices=["llama-2-chat","vicuna","mixtral","llama-3-instruct"])
parser.add_argument(
    "--total-token",
    type=int,
    default=60,
    help="The maximum number of new generated tokens.",
)
parser.add_argument(
    "--max-new-token",
    type=int,
    default=500,
    help="The maximum number of new generated tokens.",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=1.0,
    help="The maximum number of new generated tokens.",
)
parser.add_argument(
    "--depth",
    type=int,
    default=8,
    help="The maximum number of new generated tokens.",
)
parser.add_argument(
    "--topk",
    type=int,
    default=4,
    help="The maximum number of new generated tokens.",
)
parser.add_argument(
    "--version",
    type=str,
    default="default",
    help="The version of code.",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.0,
    help="The maximum number of new generated tokens.",
)
parser.add_argument(
    "--layer",
    type=str,
    default="mid_a",
    help="The maximum number of new generated tokens.",
)

parser.add_argument(
    "--dataset",
    type=str,
    default="MMT-Bench",
    help="The maximum number of new generated tokens.",
)
args = parser.parse_args()

version_map = {
    "default": "dream.model.ea_model",
}

module_name = version_map.get(args.version)
print(args.version)
if module_name is None:
    raise ValueError(f"unknown version: {args.version}")

module = importlib.import_module(module_name)

model = module.EaModel.from_pretrained(
    base_model_path=args.base_model_path,
    ea_model_path=args.ea_model_path,
    total_token=args.total_token,
    depth=args.depth,
    top_k=args.topk,
    threshold=args.threshold,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    load_in_4bit=args.load_in_4bit,
    load_in_8bit=args.load_in_8bit,
    device_map="cuda:0",
)
model.eval()
# warmup(model)

question_file = f"data/question.jsonl"

from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, LlavaNextForConditionalGeneration
from PIL import Image
import requests
from datasets import load_dataset
import time
import json

sample_num = 1000

if args.dataset == "MMT-Bench":
    ds = load_dataset(path="datasets", data_files = "datasets/processed_data/test.jsonl", split="train")
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(sample_num))
    
elif args.dataset == "ScienceQA":
    ds = load_dataset(
        "parquet", 
        data_files={'validation': 'datasets/ScienceQA/data/test-00000-of-00001-f0e719df791966ff.parquet'},
        split="validation"
    )
    ds = ds.filter(lambda record: record["image"] is not None)  
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(sample_num))

elif args.dataset == "ChartQA":
    ds = load_dataset(
        "parquet", 
        data_files={'validation': 'datasets/ChartQA/data/test-00000-of-00001.parquet'},
        split="validation"
    )
    ds = ds.filter(lambda record: record["image"] is not None)  
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(sample_num))

elif args.dataset == "MathVista":
    ds = load_dataset(
        "parquet", 
        data_files={'validation': 'datasets/MathVista/data/test-00001-of-00002-6a611c71596db30f.parquet'},
        split="validation"
    )
    #ds = ds.filter(lambda record: record["image"] is not None)  
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(sample_num))

elif args.dataset == "OCRBench":
    ds = load_dataset(
        "parquet", 
        # TEST SET: test-00000-of-00011.parquet
        data_files={'validation': 'datasets/OCRBench-v2/data/test-00000-of-00011.parquet'},
        split="validation"
    )
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(sample_num))

elif args.dataset == "seed":
    # TEST SET : datasets/processed_seed_data/test.jsonl (Split in ge_data/seed_train_test_split.py)
    ds = load_dataset(path="datasets", data_files = "datasets/processed_seed_data/test.jsonl", split="train")
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(sample_num))

def load_bs64_image(base64_data):
    base64_data = base64_data.replace("\n", "") 
    missing_padding = len(base64_data) % 4
    if missing_padding:
        base64_data += "=" * (4 - missing_padding) 

    image_data = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_data))
    return image

def load_image(image):
    img_file = io.BytesIO(image)
    img = Image.open(img_file)
    return img

def load_image_path(base_path, image_path):
    try:
        img = Image.open(f"{base_path}/{image_path}")
    except Exception as e:
        print(f"Cannot open {image_path}, error info: {e}")
        img = None  
    return img

os.makedirs(f"data/mmt/llava-vicuna-7b", exist_ok=True)
answer_file = f"data/mmt/llava-vicuna-7b/{args.dataset}-{args.version}-top{args.topk}-d{args.depth}-total{args.total_token}-temp{args.temperature}.jsonl"
speed_up_avg = []
ar = []
sd = []
adl = []
accept_avg = []
temperature = args.temperature
top_p = 0.6
for i in range(sample_num):
    if args.dataset == "MMT-Bench":
        messages = ds[i]['conversation'][0]
        imagebase64 = ds[i]["image"]
        image = load_bs64_image(imagebase64)
    elif args.dataset == "ScienceQA":
        imagebase64 = ds[i]["image"]["bytes"]
        image = load_image(imagebase64)

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": ds["question"][i]}
            ]}
        ]
    elif args.dataset == "seed":
        messages = ds[i]['conversation'][0]
        imagebase64 = ds[i]["image"]
        image = load_bs64_image(imagebase64)
    elif args.dataset == "ChartQA":
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": ds["question"][i]}
            ]}
        ]
        image = ds["image"][i]
    elif args.dataset == "MathVista":
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": ds[i]["question"]}
            ]}
        ]
        image = load_image_path("/data/MathVista",ds[i]["image"])

    elif args.dataset == "OCRBench":
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": ds["question"][i]}
            ]}
        ]
        image = ds["image"][i]

    prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = model.processor(images=image, text=prompt, truncation=True, return_tensors="pt").to(model.base_model.device)

    input_ids = inputs.input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    input_len = input_ids.shape[1]
    naive_text = []
    cu_len = input_len
    totaltime=0
    start_time=time.time()
    total_ids=0

    start = time.time()
    for output_ids in model.naive_generate(inputs, temperature=temperature, top_p=top_p,
                                        max_new_tokens=args.max_new_token,is_llama3=args.model_type=="llama-3-instruct"):
        totaltime += (time.time() - start_time)
        total_ids+=1
        decode_ids = output_ids[0, input_len:].tolist()
        decode_ids = truncate_list(decode_ids, model.tokenizer.eos_token_id)
        text = model.tokenizer.decode(decode_ids, skip_special_tokens=True, spaces_between_special_tokens=False,
                                        clean_up_tokenization_spaces=True, )
        cu_len = output_ids.shape[1]
        start_time = time.time()
    ar_time = totaltime
    print(text)
    print('ar_time: ', ar_time)

    totaltime=0
    start_time=time.time()
    total_ids=0
    cu_len=input_len
    ea_text=[]
    for output_ids, _, _ in model.ea_generate(inputs, temperature=temperature, top_p=top_p,
                                        max_new_tokens=args.max_new_token,is_llama3=args.model_type=="llama-3-instruct", ):
        totaltime+=(time.time()-start_time)
        total_ids+=1
        decode_ids = output_ids[0, input_len:].tolist()
        decode_ids = truncate_list(decode_ids, model.tokenizer.eos_token_id)
        if args.model_type == "llama-3-instruct":
            decode_ids = truncate_list(decode_ids, model.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        text = model.tokenizer.decode(decode_ids, skip_special_tokens=True, spaces_between_special_tokens=False,
                                        clean_up_tokenization_spaces=True, )

        cu_len = output_ids.shape[1]
        new_tokens = cu_len - input_len
        start_time = time.time()
        
    sd_time = totaltime
    print(text)
    print('sd_time: ', sd_time)
    print("average draft length: ", f"{new_tokens/total_ids:.2f}")

    decoded_output = text

    record = {
        "question_id": i,
        "decoded_output": decoded_output,
        # "accept_length_list": accept_length_list,
        "average_accept_length": f"{new_tokens/total_ids:.2f}",
        "speedup": ar_time / sd_time
    }

    print('record: ', record, flush=True)
    with open(answer_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    speed_up_avg.append(ar_time / sd_time)
    ar.append(ar_time)
    sd.append(sd_time)
    adl.append(new_tokens/total_ids)
    # accept_avg.append(avg_accept_length)
print('average speed up: ', sum(ar)/sum(sd))
# print('average accept length: ', sum(accept_avg)/len(accept_avg))
print('max speedup: ', max(speed_up_avg))
print(f'average draft length: {sum(adl)/len(adl):.2f}, max draft length: {max(adl):.2f}')
