import argparse
import deepspeed

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='/scratch/yh5961/models/llava-v1.6-vicuna-7b-hf')
parser.add_argument('--configpath', type=str, default="/scratch/yh5961/EfficientMultimodalSpeculativeDecoding/DREAM/dream/train/vicuna_7B_config.json")
parser.add_argument('--tmpdir', type=str,
                    default='/vast/yh5961/llava_vicuna_7B_mid_mix665k/only_mid')
parser.add_argument('--cpdir', type=str, default='0')
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to resume checkpoint")

parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()
import json
import wandb

wandb.login()

train_config = {
    "lr": 1e-5,
    "bs": 4,
    "gradient_accumulation_steps": 1,
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    "num_epochs": 20,
    "num_warmup_steps": 2000,
    "total_steps": 800000,
    "p_w": 1.0,
    "v_w": 0.4,
    "head_w": 0.1,
    "num_workers": 4,
    "embeding": True,
    "act": "No",
    "data_noise": True,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "residual": "true,norm",
    "max_len": 5120,
    "config_path": args.configpath,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
}

from safetensors import safe_open
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(0)
accelerator = Accelerator(mixed_precision="bf16")
from cnets import Model
from configs import EConfig
from typing import Any, Dict, List

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# import accelerate
from torch.autograd import Variable
import numpy as np


deepspeed.init_distributed()
rank = torch.distributed.get_rank()
if rank == 0:
    import wandb
    wandb.init(project="Bingle", name="hyh209127306-bilibili", config=train_config)
try:
    with open(os.path.join(args.basepath, "model.safetensors.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["language_model.lm_head.weight"]
    with safe_open(os.path.join(args.basepath, head_path),
                   framework="pt",
                   device="cpu") as f:
        tensor_slice = f.get_slice("language_model.lm_head.weight")
        vocab_size, hidden_dim = tensor_slice.get_shape()
        tensor = tensor_slice[:, :hidden_dim].float()
except:
    with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    weights = torch.load(os.path.join(args.basepath, head_path))
    tensor = weights["lm_head.weight"].float()
    

head = torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False).to(torch.bfloat16)
head.weight.data = tensor


def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path, followlinks=True):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.data = datapath
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # try:
        data = torch.load(self.data[index])
        new_data = {}
        hidden_state = data['target'][:train_config["max_len"]]
        #input_ids = data['input_ids'][:train_config["max_len"]][None, :]
        inputs_embeds = data['inputs_embeds'][:train_config["max_len"]]
        hidden_state_mid = data['hidden_state_mid_a'][:train_config["max_len"]]
        loss_mask = data["loss_mask"][:train_config["max_len"]]


        length = hidden_state.shape[1]
        # length_q = data['query_ids'].shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        # loss_mask[-1] = 0

        #input_ids_target = input_ids[:, 1:]
        #zeropadding = torch.tensor([[0]])
        #input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        target = hidden_state
        
        hidden_state = hidden_state[:,:-1,:]
        #loss_mask[-1] = 0
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["hidden_state_mid"] = hidden_state_mid
        #new_data["input_ids"] = input_ids_target
        new_data["inputs_embeds"] = inputs_embeds


        if self.transform:
            new_data = self.transform(new_data)

        return new_data


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_mid'].shape[1] for item in features)
        # batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_inputs_embeds = torch.cat([self.paddingtensor(item['inputs_embeds'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length-1) for item in features])
        batch_hidden_states_mid = torch.cat([self.paddingtensor(item['hidden_state_mid'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        # batch_loss_mask = torch.ones_like(batch_loss_mask)
        # batch_attention_mask=torch.ones_like(batch_attention_mask)
        batch = {
            #"input_ids": batch_input_ids,
            "inputs_embeds": batch_inputs_embeds,
            "hidden_states": batch_hidden_states,
            "hidden_states_mid": batch_hidden_states_mid,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch

def top_accuracy(output, target, topk=(1,)):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res

def compute_loss(target, target_p, predict, loss_mask):
    out_head = head_engine(predict)
    out_logp = nn.LogSoftmax(dim=2)(out_head)
    plogp = target_p * out_logp
    ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.shape[0] * loss_mask.shape[1] + 1e-5)
    vloss = criterion(predict, target.to(rank))
    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.shape[0] * loss_mask.shape[1] + 1e-5)
    kldloss = F.kl_div(out_logp, target_p, reduction='none')
    kldloss = torch.sum(torch.mean(loss_mask * kldloss, 2)) / (loss_mask.sum() + 1e-5)
    return vloss, kldloss, out_head

def compute_mid_loss(target, predict, loss_mask):
    vloss = criterion(predict, target.to(rank))
    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.shape[0] * loss_mask.shape[1] + 1e-5)
    return vloss

if train_config["data_noise"]:
    if train_config["noise"] == "uniform":
        aug = AddUniformNoise(std=train_config["std"])
    else:
        aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
else:
    aug = None

datapath = list_files(train_config["datapath"])

traindatapath = datapath[:int(len(datapath) * 0.95)]
testdatapath = datapath[int(len(datapath) * 0.95):]
traindataset = CustomDataset(traindatapath, transform=aug)
testdataset = CustomDataset(testdatapath)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False,
                         collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)


if rank == 0:
    if not os.path.exists(args.cpdir):
        os.makedirs(args.cpdir)

config = EConfig.from_pretrained(train_config["config_path"])
model = Model(config, path=args.basepath, load_emb=True)

criterion = nn.SmoothL1Loss(reduction="none")

num_epochs = train_config["num_epochs"]
num_warmup_steps = train_config["num_warmup_steps"]
total_steps = train_config["total_steps"]
is_warmup = train_config["is_warmup"]

model_engine, optimizer, train_loader, _ = deepspeed.initialize(args=args,
                                                                model=model,
                                                                model_parameters=model.parameters(),
                                                                training_data=traindataset,
                                                                collate_fn=DataCollatorWithPadding()
                                                                )

head_engine, _, test_loader, _ = deepspeed.initialize(args=args,
                                                      model=head,
                                                      model_parameters=head.parameters(),
                                                      training_data=testdataset,
                                                      collate_fn=DataCollatorWithPadding()
                                                      )

if args.resume_from_checkpoint:
    ckpt_path, _ = model_engine.load_checkpoint(args.resume_from_checkpoint)
    if ckpt_path is None:
        raise ValueError(f"Failed to load checkpoint from {args.resume_from_checkpoint}")
    else:
        print(f"Resumed training from {ckpt_path}")

for param in head.parameters():
    param.requires_grad = False

for epoch in range(0, num_epochs):
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.train()
    for batch_idx, data in enumerate(train_loader):

        model.zero_grad()

        inputs_embeds = Variable(data["inputs_embeds"].to(torch.bfloat16), requires_grad=True)
        last_hidden_state = Variable(data["hidden_states"].to(torch.bfloat16), requires_grad=True)
        predict, all_hidden_states = model_engine(last_hidden_state.to(rank), inputs_embeds=inputs_embeds.to(rank),
                               attention_mask=data["attention_mask"].to(rank), output_hidden_states=True)
        mid_predict = all_hidden_states[1]
        with torch.no_grad():
            target_head = head_engine(data["target"].to(torch.bfloat16).to(rank))
            target_p = nn.Softmax(dim=2)(target_head)
            target_p = target_p.detach()

        loss_mask = data["loss_mask"][:, :, None].to(rank)
        vloss, ploss, out_head = compute_loss(data["target"], target_p, predict, loss_mask)
        mid_vloss = compute_mid_loss(data["hidden_states_mid"], mid_predict, loss_mask)
        loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss + mid_vloss
        # loss.backward()
        model_engine.backward(loss)
        # accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])

        model_engine.step()

        with torch.no_grad():
            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            ct = loss_mask.sum().item()
            cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
            out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            target = target.view(-1)[loss_mask.view(-1) == 1]
            topkacc = top_accuracy(out_head, target, (1, 2, 3))
            for top_i in range(len(topkacc)):
                top_3acc[top_i] += topkacc[top_i]
            total += ct
            correct += cc
        if rank == 0 and ct != 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"], "train/vloss": vloss.item(),
                       "train/ploss": ploss.item(), "train/loss": loss.item(), "train/acc": cc / ct}
            for id, i in enumerate(top_3acc):
                logdict[f'train/top_{id + 1}_acc'] = topkacc[id].item() / ct
            for id,i in enumerate(top_3acc):
                 wandb.log({f'train/top_{id+1}_acc':topkacc[id].item()/ct})

        del ploss, vloss
        epoch_loss += loss.item()
        num_batches += 1

        model.zero_grad()
        
        last_hidden_state = Variable(data["hidden_states"].to(torch.bfloat16), requires_grad=True).to(rank)
        last_hidden_state = torch.where(loss_mask[:,:-1,:] == 1, predict[:, :-1, :].detach(), last_hidden_state)
        inputs_embeds = Variable(data["inputs_embeds"].to(torch.bfloat16), requires_grad=True)
        predict, all_hidden_states = model_engine(last_hidden_state.to(rank), inputs_embeds=inputs_embeds.to(rank),
                               attention_mask=data["attention_mask"].to(rank), output_hidden_states=True)
        mid_predict = all_hidden_states[1]
        with torch.no_grad():
            target_head = head_engine(data["target"].to(torch.bfloat16).to(rank))
            target_p = nn.Softmax(dim=2)(target_head)
            target_p = target_p.detach()

        loss_mask = data["loss_mask"][:, :, None].to(rank)
        vloss, ploss, out_head = compute_loss(data["target"], target_p, predict, loss_mask)
        #mid_vloss = compute_mid_loss(data["hidden_states_mid"], mid_predict, loss_mask)
        loss = ploss
        # loss.backward()
        model_engine.backward(loss)
        # accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])

        model_engine.step()

        with torch.no_grad():
            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            ct = loss_mask.sum().item()
            cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
            out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            target = target.view(-1)[loss_mask.view(-1) == 1]
            topkacc = top_accuracy(out_head, target, (1, 2, 3))
            for top_i in range(len(topkacc)):
                top_3acc[top_i] += topkacc[top_i]
            total += ct
            correct += cc
        if rank == 0 and ct != 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"], "train/vloss": vloss.item(),
                       "train/ploss": ploss.item(), "train/loss": loss.item(), "train/acc": cc / ct}
            for id, i in enumerate(top_3acc):
                logdict[f'train/adapter_top_{id + 1}_acc'] = topkacc[id].item() / ct
            for id,i in enumerate(top_3acc):
                 wandb.log({f'train/adapter_top_{id+1}_acc':topkacc[id].item()/ct})

        del ploss, vloss
        epoch_loss += loss.item()
        num_batches += 1

    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    epoch_loss /= num_batches
    top_3acc = accelerator.gather_for_metrics(top_3acc)
    if accelerator.is_local_main_process:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        print('Train Accuracy: {:.2f}%'.format(100 * correct / (total + 1e-5)))
        wandb.log(
            {"train/epochacc": correct / (total + 1e-5), "train/epochloss": epoch_loss})

    model_engine.save_16bit_model(f"{args.cpdir}/state_{epoch}")
    if epoch % 1 == 0:
        deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=f"{args.cpdir}/state_{epoch}")
