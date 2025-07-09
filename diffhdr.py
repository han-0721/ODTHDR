import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse

import os
import sys
sys.path.append(os.getcwd())
import yaml
import copy
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from models.autoencoder_kl import AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from peft import LoraConfig

         
def initialize_vae(args): # pretrain的VAE初始化
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.train()
    
    l_target_modules_encoder = []
    l_grep = ["conv1","conv2","conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n: 
            continue
        for pattern in l_grep:
            if pattern in n and ("encoder" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
            elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
    
    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder) # LoRA
    vae.add_adapter(lora_conf_encoder, adapter_name="default_encoder")

    return vae, l_target_modules_encoder


def initialize_unet(args, return_lora_module_names=False, pretrained_model_name_or_path=None):
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break

    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder)
    lora_conf_others = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_modules_others)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")

    return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others

                
class OSEDiff_gen(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer") # 分词化
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").cuda() # 将分词化文本内容转为嵌入
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler") # 噪声调度器，决定扩散过程中的噪声分布和步长
        self.noise_scheduler.set_timesteps(1, device="cuda") # 只进行一次噪声注入 # 扩散过程中总共的去噪步数
        # self.noise_scheduler.set_timesteps(5, device="cuda") # 只进行一次噪声注入 # 扩散过程中总共的去噪步数
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda() # 作为预计算的噪声缩放因子
        self.args = args

        self.vae, self.lora_vae_modules_encoder = initialize_vae(self.args) # 初始化 VAE（变分自编码器）， lora_vae_modules_encoder 存储 LoRA 相关的 VAE 编码器参数
        self.unet, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = initialize_unet(self.args) # 初始化 UNet（常用于扩散模型的主网络）
        self.lora_rank_unet = self.args.lora_rank # 设置 LoRA 低秩参数的秩（rank），用于减少训练参数量，提高训练效率
        self.lora_rank_vae = self.args.lora_rank

        self.unet.to("cuda")
        self.vae.to("cuda")
        self.timesteps = torch.tensor([999], device="cuda").long() # Unet的默认时间步， 指定当前正在执行去噪的时间步， 由于上面噪声调度器只进行1次噪声注入，故只存在 999——>0 一步，也即onestep
        self.text_encoder.requires_grad_(False) # 不训练

    def set_train(self):
        self.unet.train()
        self.vae.train()
        
        # for n, _p in self.unet.named_parameters():
        #     if "lora" in n:
        #         _p.requires_grad = True
        # self.unet.conv_in.requires_grad_(True)
        # for n, _p in self.vae.named_parameters():
        #     if "lora" in n:
        #         _p.requires_grad = True

        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n or "mid_block" in n:
                p.requires_grad = True
        # for n, p in self.unet.named_parameters():
        #     p.requires_grad = True
        for n, p in self.vae.named_parameters():
            if "lora" in n or "post_quant_conv" in n:
                p.requires_grad = True

    def encode_prompt(self, prompt_batch): # Prompt Extractor
        prompt_embeds_list = [] # 将输入的prompt_batch转为文本嵌入
        with torch.no_grad():
            for caption in prompt_batch: # 逐条处理文本prompt
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids # 分词化
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0] # 文本编码
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    def forward(self, c_t, prompt, args=None): # c_t为输入图像

        assert not torch.isnan(c_t).any(), "Input contains NaN!"

        encoded_control = self.vae.encode(c_t).latent_dist.sample() # 用 VAE 编码 c_t，得到潜在表示 encoded_control，并乘以缩放因子
        if torch.isnan(encoded_control).any():
            print("encoded_control包含NaN，使用nan_to_num修复")
            encoded_control = torch.nan_to_num(encoded_control) 
        encoded_control *= self.vae.config.scaling_factor

        # calculate prompt_embeddings and neg_prompt_embeddings
        prompt_embeds = self.encode_prompt(prompt) # 正样本的文本嵌入
        # neg_prompt_embeds = self.encode_prompt(batch["neg_prompt"]) # 负样本的文本嵌入
        model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=prompt_embeds.to(torch.float32),).sample # unet处理去噪
        
        # if torch.isnan(model_pred).any():
        #     print("model_pred包含NaN，使用nan_to_num修复")

        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample # DDPMScheduler噪声调度器处理去噪过程

        # if torch.isnan(x_denoised).any():
        #     print("x_denoised包含NaN，使用nan_to_num修复")
            
        out = x_denoised / self.vae.config.scaling_factor
        # if torch.isnan(out).any():
        #     print("out包含NaN，使用nan_to_num修复")
            # out = torch.nan_to_num(out)
        output_image = (self.vae.decode(out).sample).clamp(-1, 1) # VAE解码

        # return output_image, x_denoised, prompt_embeds, neg_prompt_embeds
        return output_image, x_denoised

    def save_model(self, outf): # 主干网络不进行更新，只更新unet的conv_in与lora模块和vae的lora模块
        sd = {}
        sd["vae_lora_encoder_modules"] = self.lora_vae_modules_encoder
        sd["unet_lora_encoder_modules"], sd["unet_lora_decoder_modules"], sd["unet_lora_others_modules"] = self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k} # 只保存 lora 相关的参数，减少存储开销
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k}
        torch.save(sd, outf)