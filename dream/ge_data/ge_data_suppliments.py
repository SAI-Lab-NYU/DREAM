import argparse
import base64
import os
from io import BytesIO

import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
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
        conversation = record["conversation"]
        image = load_image(image_base64)
        return image, conversation


def collate_fn(batch):
    images, conversations = zip(*batch)
    prompts = [processor.apply_chat_template(
        msg, add_generation_prompt=True) for msg in conversations]
    return prompts, list(images)


def compute_cosine_distance(f1, f2, eps=1e-8):
    """
    Computes the cosine distance between two feature vectors:
      f1, f2: Tensor, shape = (B, s, d)
    Returns: cosine distance, shape = (B, s)
    """
    cosine_sim = F.cosine_similarity(f1, f2, dim=-1, eps=eps)
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def compute_attention_entropy(attn_weights, eps=1e-8):
    """
    Computes entropy for each layer's attention weights:
      attn_weights: Tensor, shape = (B, head, qlen, kvlen)
    Here, entropy is computed along the kvlen dimension, then averaged over the heads, returning shape = (B, qlen)
    """
    attn_weights = attn_weights.to(torch.float16).cpu()
    entropy = - (attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)
    attn_entropy = entropy.mean(dim=1)
    return attn_entropy


def compute_mutual_attention_entropy(attn_weights1, attn_weights2, eps=1e-8):
    entropy1 = compute_attention_entropy(attn_weights1, eps)
    entropy2 = compute_attention_entropy(attn_weights2, eps)
    attn_weights1 = attn_weights1.to(torch.float16).cpu()
    attn_weights2 = attn_weights2.to(torch.float16).cpu()

    joint = attn_weights1 * attn_weights2
    joint = joint / (joint.sum(dim=-1, keepdim=True) + eps)

    joint_entropy = - (joint * torch.log(joint + eps)).sum(dim=-1)
    joint_entropy = joint_entropy.mean(dim=1)

    mutual_entropy = entropy1 + entropy2 - joint_entropy
    return mutual_entropy


def mid_feature_collect_a(features_tuple, attn_tuple, eps=1e-8):
    L = int(0.4 * (len(features_tuple)-1))
    B, s, d = features_tuple[0].shape

    features_stack = torch.stack(features_tuple[:L+1], dim=0).cpu()

    att_entropy_list = []
    for l in range(L):
        att_entropy = compute_attention_entropy(
            attn_tuple[l].to(features_stack.device), eps=eps)
        att_entropy_list.append(att_entropy)
    att_entropy_stack = torch.stack(att_entropy_list, dim=0)

    att_mutual_entropy_list = []
    for l in range(1, L-1):
        cd = torch.zeros((B, s), device=features_stack.device)
        cd += compute_mutual_attention_entropy(
            attn_tuple[l].to(features_stack.device), attn_tuple[l-1].to(features_stack.device))
        cd += compute_mutual_attention_entropy(
            attn_tuple[l].to(features_stack.device), attn_tuple[l+1].to(features_stack.device))
        att_mutual_entropy_list.append(cd)
    att_mutual_entropy_stack = torch.stack(att_mutual_entropy_list, dim=0)

    total_metric = att_entropy_stack[1:L-1] + att_mutual_entropy_stack
    best_layer_idx = total_metric.argmin(dim=0) + 1
    best_layer_idx_expanded = best_layer_idx.unsqueeze(0).unsqueeze(-1)
    best_features_a = torch.gather(features_stack, dim=0,
                                   index=best_layer_idx_expanded.expand(1, B, s, d)).squeeze(0)

    total_metric = att_entropy_stack[1:L-1]
    best_layer_idx = total_metric.argmin(dim=0) + 1
    best_layer_idx_expanded = best_layer_idx.unsqueeze(0).unsqueeze(-1)
    best_features_b = torch.gather(features_stack, dim=0,
                                   index=best_layer_idx_expanded.expand(1, B, s, d)).squeeze(0)

    total_metric = att_mutual_entropy_stack
    best_layer_idx = total_metric.argmin(dim=0) + 1
    best_layer_idx_expanded = best_layer_idx.unsqueeze(0).unsqueeze(-1)
    best_features_c = torch.gather(features_stack, dim=0,
                                   index=best_layer_idx_expanded.expand(1, B, s, d)).squeeze(0)

    return best_features_a.cpu(), best_features_b.cpu(), best_features_c.cpu()


def compute_attention_scores(self_attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Generate token scores:
      self_attn_weights: [B, H, Q, K] (Q==K)
    Returns:
      scores: [B, K], score[b, j] = mean_{h,i} attn[b,h,i,j]
    """
    scores = self_attn_weights.mean(dim=1)
    scores = scores.mean(dim=1)
    return scores


def prune_image_tokens(
    input_ids: torch.LongTensor,
    embeds: torch.Tensor,
    features: torch.Tensor,
    attentions: torch.Tensor,
    image_token_id: int = 32000,
    keep_ratio: float = 0.7,
    start: int = None,
    end: int = None,
    pad_token_id: int = 0
):
    B, L = input_ids.shape
    D = embeds.size(-1)
    device = input_ids.device
    scores = compute_attention_scores(attentions)

    start = 0 if start is None else start
    end = L if end is None else end

    new_ids, new_embs, new_target_attention = [], [], []
    for b in range(B):
        keep_mask = (input_ids[b] != image_token_id)

        if start is not None and end is not None:
            in_range = torch.zeros(L, dtype=torch.bool, device=device)
            in_range[start:end] = True
            drop_mask = (input_ids[b] == image_token_id) & in_range
            keep_mask[drop_mask] = False

            img_indices = drop_mask.nonzero(as_tuple=False).squeeze(1)
            if img_indices.numel() > 0:
                img_scores = scores[b, img_indices]
                k = max(1, int(img_indices.numel() * keep_ratio))
                topk_rel = torch.topk(img_scores, k, largest=True).indices
                topk_idx = img_indices[topk_rel]
                keep_mask[topk_idx] = True

        new_ids.append(input_ids[b][keep_mask])
        new_embs.append(embeds[b][keep_mask])
        new_target_attention.append(features[b][keep_mask])

    if B == 1:
        new_input_ids = new_ids[0].unsqueeze(0)
        new_embeds = new_embs[0].unsqueeze(0)
        new_target_attention = new_target_attention[0].unsqueeze(0)

    return new_input_ids, new_embeds, new_target_attention


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
    inputs = inputs.to("cuda:0")
    outs_big = bigmodel(**inputs, output_hidden_states=True,
                        output_attentions=True)

    new_input_ids, new_embeds, new_target = prune_image_tokens(
        inputs.input_ids, outs_big.hidden_states[0], outs_big.hidden_states[-1], outs_big.attentions[-1])

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

    td = {"loss_mask": loss_mask.cpu()}
    td["attention_mask"] = inputs.attention_mask.cpu()
    td[f"inputs_embeds"] = new_embeds.cpu()
    td[f"hidden_state_layer8"] = outs_big.hidden_states[8].cpu()
    td[f"hidden_state_layer16"] = outs_big.hidden_states[16].cpu()
    td[f"hidden_state_layer24"] = outs_big.hidden_states[24].cpu()
    td[f"hidden_state_mid_a"], td[f"hidden_state_mid_b"], td[f"hidden_state_mid_c"] = mid_feature_collect_a(
        outs_big.hidden_states, outs_big.attentions)

    td[f"hidden_state_big"] = outs_big.hidden_states[-1].cpu()
    td[f"target"] = new_target.cpu()

    return td


def writedata(name, data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length = len(os.listdir(name))
    idx = current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sp')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--index', type=int, default=1)
    parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
    parser.add_argument('--outdir', type=str, default='outdir0')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]

    target_model_id = "/home/asperger/models/llava-v1.6-vicuna-7b-hf"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    bigmodel = LlavaNextForConditionalGeneration.from_pretrained(
        target_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
        quantization_config=quantization_config
    )
    processor = AutoProcessor.from_pretrained(target_model_id)

    if "vicuna" in target_model_id:
        assist_tokens = processor.tokenizer.encode(
            "ASSISTANT:", add_special_tokens=False)
        end_tokens = processor.tokenizer.encode(
            "ASSISTANT:", add_special_tokens=False)
        image_tokens = processor.tokenizer.encode(
            "<image>", add_special_tokens=False)
    elif "mistral" in target_model_id:
        assist_tokens = processor.tokenizer.encode(
            "[/INST]:", add_special_tokens=False)
        end_tokens = processor.tokenizer.encode(
            "[INST]:", add_special_tokens=False)
        image_tokens = processor.tokenizer.encode(
            "<image>", add_special_tokens=False)

    assist_len = len(assist_tokens)
    end_len = len(end_tokens)

    ds = load_dataset("ge_data/processed_data",
                      data_files="ge_data/processed_data/train.jsonl", split="train")
    batch_size = 1
    ds = ds.shuffle(seed=42)
    if len(ds) < args.end:
        args.end = len(ds)
    ds = ds.select(range(args.start, args.end))
    dataset = MMTBenchDataset(ds)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn,
        pin_memory=True
    )

    outdir = f'{args.outdir}/{args.index}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for data in tqdm(data_loader):
        outdata = ge(data)
        writedata(outdir, outdata)
