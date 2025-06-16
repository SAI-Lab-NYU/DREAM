import argparse
import base64
from io import BytesIO
from tqdm import tqdm

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=10000)
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
parser.add_argument('--outdir', type=str, default='outdir0')
args = parser.parse_args()
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from torch.utils.data import Dataset, DataLoader



base_dataset_path = "/scratch/yh5961/data/llava-v1.5-instruct/download/llava-v1.5-instruct"

target_model_id = "/scratch/yh5961/models/llava-v1.6-vicuna-7b-hf"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True
)

bigmodel = LlavaNextForConditionalGeneration.from_pretrained(
    target_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
    quantization_config=quantization_config
)
processor = AutoProcessor.from_pretrained(target_model_id)

if "vicuna" in target_model_id:
    assist_tokens = processor.tokenizer.encode("ASSISTANT:", add_special_tokens=False)
    end_tokens = processor.tokenizer.encode("USER:", add_special_tokens=False)
    image_tokens = processor.tokenizer.encode("<image>", add_special_tokens=False)
elif "mistral" in target_model_id:
    assist_tokens = processor.tokenizer.encode("[/INST]:", add_special_tokens=False)
    end_tokens = processor.tokenizer.encode("[INST]:", add_special_tokens=False)
    image_tokens = processor.tokenizer.encode("<image>", add_special_tokens=False)

assist_len = len(assist_tokens)
end_len = len(end_tokens)

def load_image(base_path, image_path):
    try:
        img = Image.open(f"{base_path}/{image_path}")
    except Exception as e:
        print(f"Unable to open image {image_path}, error message: {e}")
        img = None 
    return img

def convert_conversation2(c3):
    conversation2 = []
    for msg in c3:
        role = "user" if msg.get("from") == "human" else "assistant"
        content_list = []

        if role == "user":
            value = msg.get("value", "")
            if "\n<image>" in value:
                text_part, _ = value.split("\n<image>", 1)
                text_part = text_part.strip()
                if text_part:
                    content_list.append({"type": "text", "text": text_part})
                content_list.append({"type": "image"})
            else:
                content_list.append({"type": "text", "text": value})
        else:
            content_list.append({"type": "text", "text": msg.get("value", "")})

        conversation2.append({
            "role": role,
            "content": content_list
        })

    return conversation2




class MMTBenchDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        record = self.dataset[idx]
        image_path = record["image"]
        if image_path == None:
            raise Exception(f"{image_path} no image")
        conversation = convert_conversation2(record["conversations"])
        image = load_image(base_path=base_dataset_path, image_path=image_path)
        return image, conversation

def collate_fn(batch):
    images, conversations = zip(*batch)
    print(conversations)

    prompts = [processor.apply_chat_template(msg, add_generation_prompt=True) for msg in conversations]
    
    return prompts, list(images)

def compute_attention_entropy(attn_weights, eps=1e-8):
    attn_weights = attn_weights.to(torch.bfloat16)
    entropy = - (attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)  # (B, head, qlen)
    attn_entropy = entropy.mean(dim=1)  # (B, qlen)
    return attn_entropy

def mid_feature_collect(features_tuple, attn_tuple, eps=1e-8):
    L = int(0.4 * (len(features_tuple)-1))
    B, s, d = features_tuple[0].shape
    print(L)

    features_stack = torch.stack(features_tuple[:L+1], dim=0).to(torch.bfloat16)

    att_entropy_list = []
    for l in range(L):
        att_entropy = compute_attention_entropy(attn_tuple[l].to(features_stack.device), eps=eps)  # (B, qlen)
        att_entropy_list.append(att_entropy)
    att_entropy_stack = torch.stack(att_entropy_list, dim=0)  # (L, B, s)

    total_metric = att_entropy_stack[1:L-1]# (L, B, s)


    best_layer_idx = total_metric.argmin(dim=0) + 1  # (B, s)

    best_layer_idx_expanded = best_layer_idx.unsqueeze(0).unsqueeze(-1)  # (1, B, s, 1)
    best_features_b = torch.gather(features_stack, dim=0, 
                                 index=best_layer_idx_expanded.expand(1, B, s, d)).squeeze(0)  # (B, s, d)
    return best_features_b.cpu()



ds = load_dataset("mrm8488/llava_v1_5_mix665k", split="train")
batch_size = 1
ds = ds.shuffle(seed=42)
if len(ds) < args.end:
    args.end = len(ds)
ds = ds.select(range(args.start, args.end))
ds = ds.filter(lambda x: x["image"] is not None and x["conversations"] is not None)
dataset = MMTBenchDataset(ds)
data_loader = DataLoader(
    dataset, 
    batch_size=1,
    shuffle=True, 
    num_workers=1,        
    collate_fn=collate_fn,
    pin_memory=True
)

import gc

@torch.no_grad()
def ge(data):
    prompts, images = data
    inputs = processor(
        images=images, 
        text=prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=5120, 
        padding_side='left'
    )
    inputs = inputs.to(bigmodel.device)
    outs_big = bigmodel(**inputs, output_hidden_states=True, output_attentions=True)
    loss_mask = torch.zeros_like(inputs.input_ids)

    for i in range(inputs.input_ids.size(0)):
        tokens = inputs.input_ids[i].cpu()
        start_idx = None
        j = 0
        while j < tokens.size(0):
            if start_idx is None and j <= tokens.size(0) - assist_len and tokens[j:j+assist_len].tolist() == assist_tokens:
                start_idx = j  
                j += assist_len  
                continue
            if start_idx is not None and j <= tokens.size(0) - end_len and tokens[j:j+end_len].tolist() == end_tokens:
                loss_mask[i, start_idx+assist_len:j] = 1
                start_idx = None  
                j += end_len  
                continue
            j += 1
        loss_mask[i, start_idx+assist_len:-assist_len] = 1
            

    td={"loss_mask":loss_mask.cpu()}
    td["attention_mask"]=inputs.attention_mask.cpu()
    # early exit layer 
    # exit at layer2 for vicuna-7B and layer3 for vicuna-13B 
    td[f"inputs_embeds"] = outs_big.hidden_states[0].cpu()
    td[f"hidden_state_mid"] = mid_feature_collect(outs_big.hidden_states, outs_big.attentions)

    td[f"target"] = outs_big.hidden_states[-1].cpu()

    return td
    
outdir = f'{args.outdir}/{args.index}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

def writedata(name,data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')


for data in tqdm(data_loader):
    torch.cuda.empty_cache()
    outdata = ge(data)
    writedata(outdir,outdata)


