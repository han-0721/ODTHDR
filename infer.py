import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datasets.dataloader import Load_Data
from diffhdr import OSEDiff_gen
from mainhdr import MainHDR_gen
from options import Options
from torchvision import transforms
import torchvision.transforms.functional as F
from util import (
    load_checkpoint,
    make_required_directories,
    drago_tonemap,
    save_checkpoint,
    save_hdr_image,
    save_ldr_image,
    save_rgb_image,
    update_lr,
    normalize,
)

opt = Options().parse()
dataset = Load_Data(mode="infer", opt=opt)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
max_num = 1000000
num = min(len(dataset), max_num)
print("Testing num: ", num)

diffmodel = OSEDiff_gen(args=opt)
diffmodel.vae.set_adapter(['default_encoder'])
diffmodel.unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])
mainmodel = MainHDR_gen(dim=opt.dim, window_size=opt.window_size, num_heads=opt.num_heads)

print("model is set for inferring")

str_ids = opt.gpu_ids.split(",")
opt.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opt.gpu_ids.append(id)
if len(opt.gpu_ids) > 0:
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= len(opt.gpu_ids)
    torch.cuda.set_device(opt.gpu_ids[0])
    if len(opt.gpu_ids) > 1:
        diffmodel = torch.nn.DataParallel(diffmodel, device_ids=opt.gpu_ids)
        mainmodel = torch.nn.DataParallel(mainmodel, device_ids=opt.gpu_ids)
    diffmodel.cuda()
    mainmodel.cuda()

diffmodel.load_state_dict(torch.load(opt.diffhdr_ckpt_path))
mainmodel.load_state_dict(torch.load(opt.mainhdr_ckpt_path))

from ram.models.ram_lora import ram
from ram import inference_ram as inference
ram_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

model_vlm = ram(pretrained=opt.ram_path,
    pretrained_condition=None,
    image_size=384,
    vit='swin_l')
model_vlm.eval()
model_vlm.to("cuda", dtype=torch.float32)

make_required_directories(mode="test")

npy_dir = "./test_results/npy/"
os.makedirs(npy_dir, exist_ok=True)
print(f"Created directory for npy files: {npy_dir}")

print("Starting evaluation. Results will be saved in '/test_results' directory")

with torch.no_grad():
    for batch, data in enumerate(tqdm(data_loader, desc="Testing %")):

        input = data["ldr_image"].data.cuda() # 0-1
        
        file_path = data["path"][0] 
        filename = os.path.splitext(os.path.basename(file_path))[0] 

        input = F.normalize(input, mean=[0.5], std=[0.5])
        
        gt_ram = ram_transforms(input*0.5+0.5)
        caption = inference(gt_ram.to(dtype=torch.float32), model_vlm)
        prompt = [f'{each_caption}' for each_caption in caption]
        print("Prompt:{}".format(prompt))

        sdr_out, _ = diffmodel(input, prompt) # output > 0 !!
        hdr_out = mainmodel(input, sdr_out) # -1-1

        sdr_out = sdr_out * 0.5 + 0.5
        hdr_out = (hdr_out - hdr_out.min()) / (hdr_out.max() - hdr_out.min())  # 将值标准化到 [0, 1]
        #hdr_out = hdr_out * 255  # 将值缩放到 [0, 255]
        #hdr_out = torch.clamp(hdr_out, 0, 255)  # 确保值在 [0, 255] 范围内

        hdr_drago_tonemap = drago_tonemap(hdr_out)
        input = input * 0.5 + 0.5

        hdr_npy = hdr_out.squeeze(0).cpu().numpy()  # 形状: [C, H, W]
        np.save(os.path.join(npy_dir, f"{filename}.npy"), hdr_npy)
        print(f"Saved HDR as npy: {npy_dir}{filename}.npy")

        save_rgb_image(
            img_tensor=hdr_drago_tonemap,
            batch=0,
            path=f"./test_results/ldr_hdr_{filename}.png",
        )
        save_hdr_image(
            img_tensor=hdr_out,
            batch=0,
            path=f"./test_results/generate/{filename}.hdr",
        )          
        if batch == max_num:
            break

print("Inferringing Finished!")