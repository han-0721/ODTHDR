import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datasets.dataloader import Load_Data
from diffhdr import OSEDiff_gen
from options import Options
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn.functional as Fct
import pytorch_msssim

from common.perceptual_loss import PerceptualLoss
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
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

writer = SummaryWriter('./logger')

opt = Options().parse()
       
model = OSEDiff_gen(args=opt)
model.set_train()
print("model is set for training")
# net_lpips = lpips.LPIPS(net='vgg').cuda()
# net_lpips.requires_grad_(False)
model.vae.set_adapter(['default_encoder'])
model.unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])

dataset = Load_Data(mode="train", opt=opt)
# data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
print("Train Num: ", len(dataset))

# set GPU
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
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    model.cuda()

layers_to_opt = []
for n, _p in model.unet.named_parameters():
    if "lora" in n:
        layers_to_opt.append(_p)
layers_to_opt += list(model.unet.conv_in.parameters())
for n, _p in model.vae.named_parameters():
    if "lora" in n:
        layers_to_opt.append(_p)

optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
# optimizer = torch.optim.AdamW(layers_to_opt, lr=opt.lr, betas=(opt.adam_beta1, opt.adam_beta2), weight_decay=opt.adam_weight_decay, eps=opt.adam_epsilon,)
# optimizer = torch.optim.AdamW(layers_to_opt, lr=opt.lr, betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-8)

           
l1 = torch.nn.L1Loss()
perceptual_loss = PerceptualLoss().to(opt.gpu_ids[0]) 

# make_required_directories(mode="train")###

#load checkpoint
if opt.continue_train:
    try:
        start_epoch, model = load_checkpoint(model, opt.ckpt_path)
    except Exception as e:
        print(e)
        print("Checkpoint is empty! Training reset!")
        start_epoch = 1
        # model.apply(weights_init)
else:
    start_epoch = 1
    # model.apply(weights_init)

from ram.models.ram_lora import ram
from ram import inference_ram as inference
ram_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# model_vlm = ram(pretrained=opt.ram_path,
#     pretrained_condition=args.ram_ft_path,
#     image_size=384, #######################################
#     vit='swin_l')
# model_vlm.eval()
# model_vlm.to("cuda", dtype=torch.float16)

model_vlm = ram(pretrained=opt.ram_path,
    pretrained_condition=None,
    image_size=384,
    vit='swin_l')
model_vlm.eval()
model_vlm.to("cuda", dtype=torch.float32)

#  begin to train
for epoch in range(start_epoch, opt.epochs + 1):

    epoch_start = time.time()
    running_loss = 0

    if epoch > opt.lr_decay_after:
        update_lr(optimizer, epoch, opt)
    print("Epoch: ", epoch)

    l1_loss_sum = 0
    # mse_loss_sum = 0
    perceptual_loss_sum = 0
    ssim_loss_sum = 0
    idx = 0

    for batch, data in enumerate(tqdm(data_loader, desc="Batch %")):
        optimizer.zero_grad()
        input = data["ldr_image"].data.cuda()
        gt = data["hdr_image"].data.cuda()

        drago_tonemap_gt = drago_tonemap(gt)
        input = F.normalize(input, mean=[0.5], std=[0.5])
        drago_tonemap_gt = F.normalize(drago_tonemap_gt, mean=[0.5], std=[0.5])
        
        gt_ram = ram_transforms(drago_tonemap_gt*0.5+0.5)
        caption = inference(gt_ram.to(dtype=torch.float32), model_vlm)
        prompt = [f'{each_caption}' for each_caption in caption]
        # print("Prompt:{}".format(prompt))

        output, _ = model(input, prompt) # output > 0 !!
        # print("Input range:", input.min().item(), input.max().item())
        # print("GT range:", drago_tonemap_gt.min().item(), drago_tonemap_gt.max().item())

        l1_loss = 0
        # mse_loss = 0
        vgg_loss = 0
        ssim_loss = 0

        # print(input)
        # print(output.shape)
        # print(drago_tonemap_gt.shape)
        
        output = output * 0.5 + 0.5  # 归一化到 [0, 1]
        drago_tonemap_gt = drago_tonemap_gt * 0.5 + 0.5
        output.retain_grad() 
        
        l1_loss = l1(output, drago_tonemap_gt)
        l1_loss = torch.mean(l1_loss)
        vgg_loss = perceptual_loss(output, drago_tonemap_gt)[0]
        vgg_loss = torch.mean(vgg_loss)
        ssim_loss = 1 - pytorch_msssim.ssim(output, drago_tonemap_gt, data_range=1.0)
        ssim_loss = torch.mean(ssim_loss)
        

        if torch.isnan(l1_loss).any():
            print(output)
            print(drago_tonemap_gt)
            assert not torch.isnan(l1_loss).any(), "l1_loss is NaN, terminating training."
        assert not torch.isnan(l1_loss).any(), "l1_loss is NaN, terminating training."
        # assert not torch.isnan(vgg_loss).any(), "vgg_loss is NaN, terminating training."

        l1_loss_sum += l1_loss
        # mse_loss_sum += mse_loss
        perceptual_loss_sum += vgg_loss
        ssim_loss_sum += ssim_loss
        num = batch + 1

        # loss = l1_loss + 0.5 * mse_loss
        # loss = l1_loss + 0.25 * vgg_loss
        loss = l1_loss + 0.5 * ssim_loss + 0.1 * vgg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        idx = idx + 1

        print(output.grad)  # 看看是否是 None 或过小

        # print(drago_tonemap_gt[:, :, 505:512])
        # print(output[:, :, 505:512])

        if (batch+1) % opt.log_after == 0:
            print(
                "Epoch: {} ; Batch: {} ; Training loss: {}".format(epoch, batch + 1, running_loss / opt.log_after)
            )
            running_loss = 0

        if (batch+1) % opt.save_results_after == 0:  # save image results
            save_rgb_image(
                img_tensor=output,
                batch=0,
                path="./training_results/sdr_gen_e_{}_b_{}.png".format(epoch, batch + 1),
            )
            save_rgb_image(
                img_tensor=input*0.5+0.5,
                batch=0,
                path="./training_results/ldr_e_{}_b_{}.png".format(epoch, batch + 1),
            )
            save_rgb_image(
                img_tensor=drago_tonemap_gt,
                batch=0,
                path="./training_results/gt_hdr_e_{}_b_{}.png".format(epoch, batch + 1),
            )
            # save_hdr_image(
            #     img_tensor=output,
            #     batch=0,
            #     path="./training_results/hdr_generate_e_{}_b_{}.hdr".format(epoch, batch + 1),
            # )

        if (batch+1) % opt.batch_num == 0:
            break

    writer.add_scalar('l1_loss', l1_loss_sum/idx, epoch)
    # writer.add_scalar('mse_loss', mse_loss_sum/idx, epoch)
    # writer.add_scalar('l1_tonemap_loss', l1_tonemap_loss_sum, epoch)
    writer.add_scalar('perceptual_loss', perceptual_loss_sum/idx, epoch)
    writer.add_scalar('ssim_loss', ssim_loss_sum/idx, epoch)


    epoch_finish = time.time()
    time_taken = (epoch_finish - epoch_start) // 60

    print("epoch: {}. Time: {} minutes.".format(epoch, int(time_taken)))

    if epoch % opt.save_ckpt_after == 0:
        save_checkpoint(epoch, model)

writer.close()
print("Training Finished!")