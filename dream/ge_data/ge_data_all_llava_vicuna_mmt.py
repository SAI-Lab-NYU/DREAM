import argparse
import base64
from io import BytesIO
from tqdm import tqdm

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
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
from fastchat.model.model_adapter import get_conversation_template
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from torch.utils.data import Dataset, DataLoader


#bigname="/home/apc/models/vicuna-7b-v1.3/"

target_model_id = "/home/asperger/models/llava-v1.6-vicuna-7b-hf"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # 开启 4-bit 量化
    bnb_4bit_compute_dtype=torch.float16,  # 计算时使用的数值类型，可选：torch.float16, torch.bfloat16 等
    bnb_4bit_quant_type="nf4",             # 量化类型，常见的选项有 "nf4" 或 "fp4"
    bnb_4bit_use_double_quant=True         # 是否使用 double quantization（一般会带来更好的精度）
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
    assist_tokens = processor.tokenizer.encode("ASSISTANT:", add_special_tokens=False)
    end_tokens = processor.tokenizer.encode("ASSISTANT:", add_special_tokens=False)
    image_tokens = processor.tokenizer.encode("<image>", add_special_tokens=False)
elif "mistral" in target_model_id:
    assist_tokens = processor.tokenizer.encode("[/INST]:", add_special_tokens=False)
    end_tokens = processor.tokenizer.encode("[INST]:", add_special_tokens=False)
    image_tokens = processor.tokenizer.encode("<image>", add_special_tokens=False)

assist_len = len(assist_tokens)
end_len = len(end_tokens)

# 定义解码 base64 图像的函数
def load_image(base64_string):
    # 注意：这里假设传入的是 base64 编码的字符串
    base64_string = base64_string.replace("\n", "")
    missing_padding = len(base64_string) % 4
    if missing_padding:
        base64_string += "=" * (4 - missing_padding)
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image

def find_next_non_image_token(input_ids, image_token_ids):
    """
    找到 input_ids 中最后一个 image token，并返回下一个非-image token 的索引。
    
    Args:
        input_ids (torch.Tensor): 形状为 (seq_len,) 的 token ID 张量。
        image_token_ids (set): 包含所有 image token ID 的集合。

    Returns:
        int: 下一个非-image token 的索引（如果找到），否则返回 -1
    """
    # 找到所有 image token 的索引
    image_indices = torch.where(torch.tensor([id in image_token_ids for id in input_ids]))[0]
    
    if len(image_indices) == 0:
        return -1  # 没有找到 image token

    # 获取最后一个 image token 的索引
    last_image_index = image_indices[-1].item()

    # 找到下一个非-image token
    for i in range(last_image_index + 1, len(input_ids)):
        if input_ids[i].item() not in image_token_ids:
            return i  # 返回第一个非 image token 的索引

    return -1  # 没有找到下一个非-image token

# 定义自定义数据集类
class MMTBenchDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 从 Hugging Face 数据集中获取单条数据（包含 "image" 和 "question"）
        record = self.dataset[idx]
        image_base64 = record["image"]
        conversation = record["conversation"]
        # 将 base64 字符串转换为 PIL.Image 对象
        image = load_image(image_base64)
        return image, conversation  # image_base64 备用（写入文件时可能需要）

# 自定义 collate 函数：批量构造 prompt 并编码
def collate_fn(batch):
    # batch 是一个列表，每个元素为 (image, question, image_base64)
    images, conversations = zip(*batch)
    print(conversations)

    # 利用 processor 构造生成 prompt
    prompts = [processor.apply_chat_template(msg, add_generation_prompt=True) for msg in conversations]
    # 将图像和文本传入 processor，返回编码后的模型输入
    
    return prompts, list(images)

def mid_feature_collect(hidden_states):
    hid = torch.stack(hidden_states, dim=0)
    L, B, S, D = hid.shape

    # 定义候选层范围
    candidate_start = 3
    candidate_end = int(0.75 * (L - 1))
    if candidate_end <= candidate_start:
        raise ValueError("候选层范围无效")

    # 候选层数量
    candidate_count = candidate_end - candidate_start

    # 提取候选层及其相邻层
    # 注意：候选层的索引为 candidate_start 到 candidate_end - 1
    candidate = hid[candidate_start:candidate_end, :, :, :]       # shape: (candidate_count, B, S, D)
    candidate_prev = hid[candidate_start - 1:candidate_end - 1, :, :, :]  # shape: (candidate_count, B, S, D)
    candidate_next = hid[candidate_start + 1:candidate_end + 1, :, :, :]  # shape: (candidate_count, B, S, D)

    # 全局 anchor 层：第一层和最后一层
    anchor_left = hid[0].unsqueeze(0)    # shape: (1, B, S, D)
    anchor_right = hid[-1].unsqueeze(0)   # shape: (1, B, S, D)

    def cosine_distance(x, y, eps=1e-9):
        """
        计算余弦距离：
        d(x,y) = 1 - (x · y) / (||x|| ||y|| + eps)
        输入 x, y 的形状为 (..., D)
        返回形状为 x 和 y 除最后一维外的张量
        """
        dot_xy = (x * y).sum(dim=-1)
        norm_x = x.norm(dim=-1)
        norm_y = y.norm(dim=-1)
        return 1 - dot_xy / (norm_x * norm_y + eps)

    # 计算全局距离
    # 分别计算候选层与 anchor_left 和 anchor_right 的距离
    global_left = cosine_distance(anchor_left, candidate)   # shape: (candidate_count, B, S)
    global_right = cosine_distance(anchor_right, candidate)  # shape: (candidate_count, B, S)
    global_dis = torch.abs(global_left - global_right)       # shape: (candidate_count, B, S)

    # 计算局部距离
    local_left = cosine_distance(candidate_prev, candidate)   # shape: (candidate_count, B, S)
    local_right = cosine_distance(candidate_next, candidate)  # shape: (candidate_count, B, S)
    local_dis = torch.abs(local_left - local_right)           # shape: (candidate_count, B, S)

    # 综合距离
    total_dis = global_dis + local_dis  # shape: (candidate_count, B, S)

    # 对候选层维度（dim=0）取 argmin 得到最佳候选索引 (范围在 [0, candidate_count-1])
    best_candidate_idx = torch.argmin(total_dis, dim=0)  # shape: (B, S)

    # 转换为原始 hidden_states 的层 id
    best_layer = best_candidate_idx + candidate_start  # shape: (B, S)
    print(best_layer)


    hidden_states_trans = hid.permute(1, 2, 0, 3)  # (B, S, L, D)

    # best_layer: (B, S) → 扩展为 (B, S, 1) 用于索引
    best_layer_unsq = best_layer.unsqueeze(-1)  # (B, S, 1)

    # 利用 torch.gather 在第 2 维（即 L 维度）上获取对应层的特征
    # 首先需要将 best_layer_unsq 扩展到和最后一个特征维 D 对齐，形状变为 (B, S, 1, D)
    index = best_layer_unsq.unsqueeze(-1).expand(-1, -1, -1, hidden_states_trans.shape[-1])
    # 使用 gather 得到 (B, S, 1, D) 的结果
    selected_features = torch.gather(hidden_states_trans, dim=2, index=index)

    # squeeze 掉第 2 维（长度为 1）即可得到 (B, S, D)
    selected_features = selected_features.squeeze(2)
    return selected_features

def compute_cosine_distance(f1, f2, eps=1e-8):
    """
    计算两个 feature 向量之间的余弦距离：
      f1, f2: Tensor, shape = (B, s, d)
    返回: cosine distance, shape = (B, s)
    """
    cosine_sim = F.cosine_similarity(f1, f2, dim=-1, eps=eps)  # (B, s)
    cosine_dist = 1 - cosine_sim  # (B, s)

    return cosine_dist

def compute_l2_distance(f1, f2, eps=1e-8):
    """
    计算两个 feature 向量之间的 L2 距离：
      f1, f2: Tensor, shape = (B, s, d)
    返回: L2 distance, shape = (B, s)
    """
    # 计算两个向量的差值
    diff = f1 - f2  # (B, s, d)
    # 对差值的每个维度平方后求和，然后取平方根
    l2_dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + eps)  # (B, s)
    return l2_dist

def compute_attention_entropy(attn_weights, eps=1e-8):
    """
    对每层的 attention 权重计算熵：
      attn_weights: Tensor, shape = (B, head, qlen, kvlen)
    这里对 kvlen 维度计算熵，再对 head 取平均，返回 shape = (B, qlen)
    """
    attn_weights = attn_weights.to(torch.float16).cpu()
    # 沿着 kvlen 计算熵
    entropy = - (attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)  # (B, head, qlen)
    # 对 head 平均
    attn_entropy = entropy.mean(dim=1)  # (B, qlen)
    return attn_entropy

def compute_mutual_attention_entropy(attn_weights1, attn_weights2, eps=1e-8):
    """
    计算两个 attention weight 的 mutual entropy（互信息）。
    
    参数：
      attn_weights1: Tensor, shape = (B, head, qlen, kvlen)
      attn_weights2: Tensor, shape = (B, head, qlen, kvlen)
      eps: 防止 log(0) 的小常数
      
    计算步骤：
      1. 分别调用 compute_attention_entropy 计算两个 attention 分布的边际熵（形状为 (B, qlen)）。
      2. 计算联合 attention 分布：逐元素相乘后沿 kvlen 维度归一化，再计算熵，最后对 head 维度取平均，得到形状为 (B, qlen)。
      3. mutual entropy 定义为：entropy1 + entropy2 - joint_entropy
      
    返回：
      Tensor，形状为 (B, qlen)
    """
    # 计算两个 attention 分布各自的边际熵
    entropy1 = compute_attention_entropy(attn_weights1, eps)  # (B, qlen)
    entropy2 = compute_attention_entropy(attn_weights2, eps)  # (B, qlen)
    attn_weights1 = attn_weights1.to(torch.float16).cpu()
    attn_weights2 = attn_weights2.to(torch.float16).cpu()
    
    # 计算联合 attention 分布：逐元素乘积，然后归一化
    joint = attn_weights1 * attn_weights2  # (B, head, qlen, kvlen)
    joint = joint / (joint.sum(dim=-1, keepdim=True) + eps)  # 归一化 joint 分布
    
    # 计算联合熵（对 kvlen 维度求和，再对 head 取平均）
    joint_entropy = - (joint * torch.log(joint + eps)).sum(dim=-1)  # (B, head, qlen)
    joint_entropy = joint_entropy.mean(dim=1)  # (B, qlen)
    
    # mutual entropy 定义为边际熵之和减去联合熵
    mutual_entropy = entropy1 + entropy2 - joint_entropy
    return mutual_entropy

def mid_feature_collect_a(features_tuple, attn_tuple, eps=1e-8):
    """
    输入:
      features_tuple: tuple，每个元素为 tensor，形状为 (B, s, d)
        总层数 L = len(features_tuple)
    实现步骤：
      1. 对每层的每个 token 计算 feature entropy
      2. 计算每层与相邻层（前一层、后一层）的余弦距离和
      3. 预先堆叠后，基于向量化操作归一化和求和组合度量
      4. 对于每个 token，在所有层中选择度量最小的层作为最佳层
    返回:
      best_layer_idx: Tensor, shape = (B, s)，每个 token 对应的最佳层索引
      best_features: Tensor, shape = (B, s, d)，对应层的 feature
    """
    L = int(0.4 * (len(features_tuple)-1))  # 总层数
    B, s, d = features_tuple[0].shape
    print(L)

    # 预先堆叠各层 feature，得到 (L, B, s, d)
    features_stack = torch.stack(features_tuple[:L+1], dim=0).cpu()

    # 1. 计算各层的 feature entropy
    #entropy_stack = compute_feature_entropy(features_stack, eps=eps)[1:]  # (L, B, s)
    #print(entropy_stack)

    # 2. 计算余弦距离（与相邻层之间的和），提前堆叠有助于利用张量操作
    # cosine_list = []
    # for l in range(1, L):
    #     cd = torch.zeros((B, s), device=features_stack.device)
    #     cd += compute_cosine_distance(features_tuple[l], features_tuple[l-1])
    #     cd += compute_cosine_distance(features_tuple[l], features_tuple[l+1])
    #     cosine_list.append(cd)
    # cosine_stack = torch.stack(cosine_list, dim=0)  # (L, B, s)

    att_entropy_list = []
    for l in range(L):
        # 对应层的 attention 权重 shape = (B, head, qlen, kvlen)
        att_entropy = compute_attention_entropy(attn_tuple[l].to(features_stack.device), eps=eps)  # (B, qlen)
        att_entropy_list.append(att_entropy)
    att_entropy_stack = torch.stack(att_entropy_list, dim=0)  # (L, B, s)

    att_mutual_entropy_list = []
    for l in range(1, L-1):
        cd = torch.zeros((B, s), device=features_stack.device)
        cd += compute_mutual_attention_entropy(attn_tuple[l].to(features_stack.device), attn_tuple[l-1].to(features_stack.device))
        cd += compute_mutual_attention_entropy(attn_tuple[l].to(features_stack.device), attn_tuple[l+1].to(features_stack.device))
        att_mutual_entropy_list.append(cd)
    att_mutual_entropy_stack = torch.stack(att_mutual_entropy_list, dim=0)  # (L, B, s)

    # 3. 对两个度量在层维度上归一化（利用向量化操作），归一化到 [0,1]
    def normalize_metric(metric):
        min_val, _ = metric.min(dim=0, keepdim=True)  # (1, B, s)
        max_val, _ = metric.max(dim=0, keepdim=True)  # (1, B, s)
        normalized = (metric - min_val) / (max_val - min_val + eps)
        return normalized

    entropy_norm = normalize_metric(att_mutual_entropy_stack)
    #cosine_norm = normalize_metric(cosine_stack)
    attn_entropy_norm = normalize_metric(att_entropy_stack)
    #print("Cosine Dis:", cosine_stack)

    #print("Normalized Feature Entropy:", entropy_norm)
    #print("Normalized Cosine Distance:", cosine_norm)
    #print("Normalized Attention Entropy:", attn_entropy_norm)


    # 合并度量
    # total_metric = matrix_entropy_stack + cosine_stack + att_entropy_stack # (L, B, s)
    total_metric = att_entropy_stack[1:L-1] + att_mutual_entropy_stack# (L, B, s)
    #total_metric = att_mutual_entropy_stack# (L, B, s)


    # 4. 对每个 token 选择总得分最小的层
    best_layer_idx = total_metric.argmin(dim=0) + 1  # (B, s)

    # 根据最佳层索引选择对应的 feature（利用提前堆叠的 features_stack）
    best_layer_idx_expanded = best_layer_idx.unsqueeze(0).unsqueeze(-1)  # (1, B, s, 1)
    best_features_a = torch.gather(features_stack, dim=0, 
                                 index=best_layer_idx_expanded.expand(1, B, s, d)).squeeze(0)  # (B, s, d)
    print(best_layer_idx)

    total_metric = att_entropy_stack[1:L-1]# (L, B, s)
    #total_metric = att_mutual_entropy_stack# (L, B, s)


    # 4. 对每个 token 选择总得分最小的层
    best_layer_idx = total_metric.argmin(dim=0) + 1  # (B, s)

    # 根据最佳层索引选择对应的 feature（利用提前堆叠的 features_stack）
    best_layer_idx_expanded = best_layer_idx.unsqueeze(0).unsqueeze(-1)  # (1, B, s, 1)
    best_features_b = torch.gather(features_stack, dim=0, 
                                 index=best_layer_idx_expanded.expand(1, B, s, d)).squeeze(0)  # (B, s, d)
    print(best_layer_idx)

    total_metric = att_mutual_entropy_stack# (L, B, s)
    #total_metric = att_mutual_entropy_stack# (L, B, s)


    # 4. 对每个 token 选择总得分最小的层
    best_layer_idx = total_metric.argmin(dim=0) + 1  # (B, s)

    # 根据最佳层索引选择对应的 feature（利用提前堆叠的 features_stack）
    best_layer_idx_expanded = best_layer_idx.unsqueeze(0).unsqueeze(-1)  # (1, B, s, 1)
    best_features_c = torch.gather(features_stack, dim=0, 
                                 index=best_layer_idx_expanded.expand(1, B, s, d)).squeeze(0)  # (B, s, d)

    print(best_layer_idx)
    return best_features_a.cpu(), best_features_b.cpu(), best_features_c.cpu()

def mid_feature_collect_b(features_tuple, attn_tuple, eps=1e-8):
    """
    输入:
      features_tuple: tuple，每个元素为 tensor，形状为 (B, s, d)
        总层数 L = len(features_tuple)
    实现步骤：
      1. 对每层的每个 token 计算 feature entropy
      2. 计算每层与相邻层（前一层、后一层）的余弦距离和
      3. 预先堆叠后，基于向量化操作归一化和求和组合度量
      4. 对于每个 token，在所有层中选择度量最小的层作为最佳层
    返回:
      best_layer_idx: Tensor, shape = (B, s)，每个 token 对应的最佳层索引
      best_features: Tensor, shape = (B, s, d)，对应层的 feature
    """
    L = int(0.4 * (len(features_tuple)-1))  # 总层数
    B, s, d = features_tuple[0].shape
    print(L)

    # 预先堆叠各层 feature，得到 (L, B, s, d)
    features_stack = torch.stack(features_tuple[:L+1], dim=0)

    # 1. 计算各层的 feature entropy
    #entropy_stack = compute_feature_entropy(features_stack, eps=eps)[1:]  # (L, B, s)
    #print(entropy_stack)

    # 2. 计算余弦距离（与相邻层之间的和），提前堆叠有助于利用张量操作
    cosine_list = []
    for l in range(1, L):
        cd = torch.zeros((B, s), device=features_stack.device)
        cd += compute_cosine_distance(features_tuple[l], features_tuple[l-1])
        cd += compute_cosine_distance(features_tuple[l], features_tuple[l+1])
        cosine_list.append(cd)
    cosine_stack = torch.stack(cosine_list, dim=0)  # (L, B, s)

    att_entropy_list = []
    for l in range(L):
        # 对应层的 attention 权重 shape = (B, head, qlen, kvlen)
        att_entropy = compute_attention_entropy(attn_tuple[l], eps=eps)  # (B, qlen)
        att_entropy_list.append(att_entropy)
    att_entropy_stack = torch.stack(att_entropy_list, dim=0)  # (L, B, s)

    att_mutual_entropy_list = []
    for l in range(1, L-1):
        cd = torch.zeros((B, s), device=features_stack.device)
        cd += compute_mutual_attention_entropy(attn_tuple[l], attn_tuple[l-1])
        cd += compute_mutual_attention_entropy(attn_tuple[l], attn_tuple[l+1])
        att_mutual_entropy_list.append(cd)
    att_mutual_entropy_stack = torch.stack(att_mutual_entropy_list, dim=0)  # (L, B, s)

    # 3. 对两个度量在层维度上归一化（利用向量化操作），归一化到 [0,1]
    def normalize_metric(metric):
        min_val, _ = metric.min(dim=0, keepdim=True)  # (1, B, s)
        max_val, _ = metric.max(dim=0, keepdim=True)  # (1, B, s)
        normalized = (metric - min_val) / (max_val - min_val + eps)
        return normalized

    entropy_norm = normalize_metric(att_mutual_entropy_stack)
    cosine_norm = normalize_metric(cosine_stack)
    attn_entropy_norm = normalize_metric(att_entropy_stack)
    #print("Normalized Feature Entropy:", entropy_norm)
    #print("Normalized Cosine Distance:", cosine_norm)
    #print("Normalized Attention Entropy:", attn_entropy_norm)


    # 合并度量
    # total_metric = matrix_entropy_stack + cosine_stack + att_entropy_stack # (L, B, s)
    total_metric = att_entropy_stack[1:L-1]# (L, B, s)
    #total_metric = att_mutual_entropy_stack# (L, B, s)


    # 4. 对每个 token 选择总得分最小的层
    best_layer_idx = total_metric.argmin(dim=0) + 1  # (B, s)

    # 根据最佳层索引选择对应的 feature（利用提前堆叠的 features_stack）
    best_layer_idx_expanded = best_layer_idx.unsqueeze(0).unsqueeze(-1)  # (1, B, s, 1)
    best_features = torch.gather(features_stack, dim=0, 
                                 index=best_layer_idx_expanded.expand(1, B, s, d)).squeeze(0)  # (B, s, d)
    print(best_layer_idx)

    return best_features

def mid_feature_collect_c(features_tuple, attn_tuple, eps=1e-8):
    """
    输入:
      features_tuple: tuple，每个元素为 tensor，形状为 (B, s, d)
        总层数 L = len(features_tuple)
    实现步骤：
      1. 对每层的每个 token 计算 feature entropy
      2. 计算每层与相邻层（前一层、后一层）的余弦距离和
      3. 预先堆叠后，基于向量化操作归一化和求和组合度量
      4. 对于每个 token，在所有层中选择度量最小的层作为最佳层
    返回:
      best_layer_idx: Tensor, shape = (B, s)，每个 token 对应的最佳层索引
      best_features: Tensor, shape = (B, s, d)，对应层的 feature
    """
    L = int(0.4 * (len(features_tuple)-1))  # 总层数
    B, s, d = features_tuple[0].shape
    print(L)

    # 预先堆叠各层 feature，得到 (L, B, s, d)
    features_stack = torch.stack(features_tuple[:L+1], dim=0)

    # 1. 计算各层的 feature entropy
    #entropy_stack = compute_feature_entropy(features_stack, eps=eps)[1:]  # (L, B, s)
    #print(entropy_stack)

    # 2. 计算余弦距离（与相邻层之间的和），提前堆叠有助于利用张量操作
    cosine_list = []
    for l in range(1, L):
        cd = torch.zeros((B, s), device=features_stack.device)
        cd += compute_cosine_distance(features_tuple[l], features_tuple[l-1])
        cd += compute_cosine_distance(features_tuple[l], features_tuple[l+1])
        cosine_list.append(cd)
    cosine_stack = torch.stack(cosine_list, dim=0)  # (L, B, s)

    att_entropy_list = []
    for l in range(L):
        # 对应层的 attention 权重 shape = (B, head, qlen, kvlen)
        att_entropy = compute_attention_entropy(attn_tuple[l], eps=eps)  # (B, qlen)
        att_entropy_list.append(att_entropy)
    att_entropy_stack = torch.stack(att_entropy_list, dim=0)  # (L, B, s)

    att_mutual_entropy_list = []
    for l in range(1, L-1):
        cd = torch.zeros((B, s), device=features_stack.device)
        cd += compute_mutual_attention_entropy(attn_tuple[l], attn_tuple[l-1])
        cd += compute_mutual_attention_entropy(attn_tuple[l], attn_tuple[l+1])
        att_mutual_entropy_list.append(cd)
    att_mutual_entropy_stack = torch.stack(att_mutual_entropy_list, dim=0)  # (L, B, s)

    # 3. 对两个度量在层维度上归一化（利用向量化操作），归一化到 [0,1]
    def normalize_metric(metric):
        min_val, _ = metric.min(dim=0, keepdim=True)  # (1, B, s)
        max_val, _ = metric.max(dim=0, keepdim=True)  # (1, B, s)
        normalized = (metric - min_val) / (max_val - min_val + eps)
        return normalized

    entropy_norm = normalize_metric(att_mutual_entropy_stack)
    cosine_norm = normalize_metric(cosine_stack)
    attn_entropy_norm = normalize_metric(att_entropy_stack)
    print("Cosine Dis:", cosine_stack)
    print("Attention Entropy:", att_entropy_stack)
    print("matrix Entropy:", att_mutual_entropy_stack)

    #print("Normalized Feature Entropy:", entropy_norm)
    #print("Normalized Cosine Distance:", cosine_norm)
    #print("Normalized Attention Entropy:", attn_entropy_norm)


    # 合并度量
    # total_metric = matrix_entropy_stack + cosine_stack + att_entropy_stack # (L, B, s)
    total_metric = att_mutual_entropy_stack# (L, B, s)
    #total_metric = att_mutual_entropy_stack# (L, B, s)


    # 4. 对每个 token 选择总得分最小的层
    best_layer_idx = total_metric.argmin(dim=0) + 1  # (B, s)

    # 根据最佳层索引选择对应的 feature（利用提前堆叠的 features_stack）
    best_layer_idx_expanded = best_layer_idx.unsqueeze(0).unsqueeze(-1)  # (1, B, s, 1)
    best_features = torch.gather(features_stack, dim=0, 
                                 index=best_layer_idx_expanded.expand(1, B, s, d)).squeeze(0)  # (B, s, d)
    print(best_layer_idx)
    return best_features

def compute_attention_scores(self_attn_weights: torch.Tensor) -> torch.Tensor:
    """
    生成 token 分数：
      self_attn_weights: [B, H, Q, K] (Q==K)
    返回：
      scores: [B, K]，score[b, j] = mean_{h,i} attn[b,h,i,j]
    """
    # 对 heads 维度求均值 -> [B, Q, K]
    scores = self_attn_weights.mean(dim=1)
    # 对 query 维度再求均值 -> [B, K]
    scores = scores.mean(dim=1)
    return scores

def prune_image_tokens(
    input_ids: torch.LongTensor,      # [B, L]
    embeds: torch.Tensor,             # [B, L, D]
    features: torch.Tensor,  
    attentions: torch.Tensor,       # [B, H, Q, K]
    image_token_id: int = 32000,
    keep_ratio: float = 0.7,
    start: int = None,
    end: int = None,
    pad_token_id: int = 0
):
    """
    在 image token 上做 pruning，并同步处理 embeds：
      - input_ids:    [B, L]
      - embeds:       [B, L, D]
      - attn_weights: [B, H, Q, K] (Q==K==L)
      - image_token_id: 用于区分 image token 的特殊 id
      - keep_ratio:   在 image token 中保留多少比例
      - start,end:    可选，仅在 [start,end) 区间内做 pruning
    返回：
      new_input_ids: [B, L_new]
      new_embeds:    [B, L_new, D]
    """
    B, L = input_ids.shape
    D = embeds.size(-1)
    device = input_ids.device
    scores = compute_attention_scores(attentions)

    start = 0 if start is None else start
    end = L if end is None else end
    print(scores)

    # 2. 找出每个 batch 中 image token 的位置 mask
    new_ids, new_embs, new_target_attention = [], [], []
    for b in range(B):
        # 初始保留所有非 image token
        keep_mask = (input_ids[b] != image_token_id)

        # 如果指定区间，先全部丢弃区间内的 image token，再选 top-k 放回
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

        new_ids.append (input_ids[b][keep_mask])
        new_embs.append(embeds[b][keep_mask])
        new_target_attention.append(features[b][keep_mask])

    # 2) 根据 B 决定 pad 还是 unsqueeze
    if B == 1:
        # 直接取第一个样本，添加 batch 维度
        new_input_ids = new_ids[0].unsqueeze(0)   # [1, L']
        new_embeds    = new_embs[0].unsqueeze(0)  # [1, L', D]
        new_target_attention = new_target_attention[0].unsqueeze(0)

    return new_input_ids, new_embeds, new_target_attention



ds = load_dataset("/home/asperger/EfficientMultimodalSpeculativeDecoding/eagle/ge_data", data_files="/home/asperger/EfficientMultimodalSpeculativeDecoding/eagle/ge_data/mmt-bench-llava-v1.6-vicuna-7b.jsonl", split="train")
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
    num_workers=1,        # 可根据 CPU 核数进行调整
    collate_fn=collate_fn,
    pin_memory=True
)

@torch.no_grad()
def ge(data):
    print(data)
    prompts, images = data
    # 模型生成回答
    inputs = processor(
        images=images, 
        text=prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=5120, 
        padding_side='left'
    )
    # 将数据移动到 GPU
    inputs = inputs.to("cuda:0")
    print(processor.tokenizer.decode(inputs.input_ids[0]))
    outs_big = bigmodel(**inputs, output_hidden_states=True, output_attentions=True)

    new_input_ids, new_embeds, new_target = prune_image_tokens(inputs.input_ids, outs_big.hidden_states[0], outs_big.hidden_states[-1], outs_big.attentions[-1])

    loss_mask = torch.zeros_like(new_input_ids)

    for i in range(new_input_ids.size(0)):
        tokens = new_input_ids[i].cpu()
        start_idx = None
        j = 0
        while j < tokens.size(0):
            # 如果还未找到起始位置，检查当前 slice 是否等于 assist_tokens
            if start_idx is None and j <= tokens.size(0) - assist_len and tokens[j:j+assist_len].tolist() == assist_tokens:
                start_idx = j  # 记录 "ASSISTANT:" 起始位置
                j += assist_len  # 跳过这段 token 序列
                continue
            # 如果已找到起始位置，检查是否匹配 end_tokens
            if start_idx is not None and j <= tokens.size(0) - end_len and tokens[j:j+end_len].tolist() == end_tokens:
                # 设置 loss mask：从 "ASSISTANT:" 序列之后到结束 token 之前
                loss_mask[i, start_idx+assist_len:j] = 1
                start_idx = None  # 重置起始标记
                j += end_len  # 跳过结束 token 序列
                continue
            j += 1
        loss_mask[i, start_idx+assist_len:-assist_len] = 1
        print(loss_mask[i][-200:])

    loss_mask = torch.zeros_like(inputs.input_ids)

    for i in range(inputs.input_ids.size(0)):
        tokens = inputs.input_ids[i].cpu()
        start_idx = None
        j = 0
        while j < tokens.size(0):
            # 如果还未找到起始位置，检查当前 slice 是否等于 assist_tokens
            if start_idx is None and j <= tokens.size(0) - assist_len and tokens[j:j+assist_len].tolist() == assist_tokens:
                start_idx = j  # 记录 "ASSISTANT:" 起始位置
                j += assist_len  # 跳过这段 token 序列
                continue
            # 如果已找到起始位置，检查是否匹配 end_tokens
            if start_idx is not None and j <= tokens.size(0) - end_len and tokens[j:j+end_len].tolist() == end_tokens:
                # 设置 loss mask：从 "ASSISTANT:" 序列之后到结束 token 之前
                loss_mask[i, start_idx+assist_len:j] = 1
                print(loss_mask[i], start_idx, j)
                start_idx = None  # 重置起始标记
                j += end_len  # 跳过结束 token 序列
                continue
            j += 1
            

    td={"loss_mask":loss_mask.cpu()}
    td["attention_mask"]=inputs.attention_mask.cpu()
    # early exit layer 
    # exit at layer2 for vicuna-7B and layer3 for vicuna-13B 
    td[f"inputs_embeds"] = new_embeds.cpu()
    td[f"hidden_state_layer8"] = outs_big.hidden_states[8].cpu()
    td[f"hidden_state_layer16"] = outs_big.hidden_states[16].cpu()
    td[f"hidden_state_layer24"] = outs_big.hidden_states[24].cpu()
    #td[f"hidden_state_mid_a"], td[f"hidden_state_mid_b"], td[f"hidden_state_mid_c"] = mid_feature_collect_a(outs_big.hidden_states, outs_big.attentions)

    td[f"hidden_state_big"] = outs_big.hidden_states[-1].cpu()
    td[f"target"] = new_target.cpu()

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
    outdata = ge(data)
    writedata(outdir,outdata)


