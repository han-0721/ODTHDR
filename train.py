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
import torch.nn.functional as Fct
import pytorch_msssim
from common.perceptual_loss import PerceptualLoss
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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.1)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.01)
        
diffmodel = OSEDiff_gen(args=opt)
diffmodel.set_train()
print("diffmodel is set for training")

diffmodel.vae.set_adapter(['default_encoder'])
diffmodel.unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])

mainmodel = MainHDR_gen(dim=opt.dim, window_size=opt.window_size, num_heads=opt.num_heads)
# mainmodel.set_train()
print("mainmodel is set for training")

# model.apply(weights_init)
dataset = Load_Data(mode="train", opt=opt)
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
        diffmodel = torch.nn.DataParallel(diffmodel, device_ids=opt.gpu_ids)
        mainmodel = torch.nn.DataParallel(mainmodel, device_ids=opt.gpu_ids)
    diffmodel.cuda()
    mainmodel.cuda()

layers_to_opt = []
for n, _p in diffmodel.unet.named_parameters():
    if "lora" in n:
        layers_to_opt.append(_p)
layers_to_opt += list(diffmodel.unet.conv_in.parameters())
for n, _p in diffmodel.vae.named_parameters():
    if "lora" in n:
        layers_to_opt.append(_p)

optimizer_a = torch.optim.Adam(diffmodel.parameters(), lr=opt.lr, betas=(0.9, 0.999))
optimizer_b = torch.optim.Adam(mainmodel.parameters(), lr=opt.lr, betas=(0.9, 0.999))
# optimizer = torch.optim.AdamW(layers_to_opt, lr=opt.lr, betas=(opt.adam_beta1, opt.adam_beta2), weight_decay=opt.adam_weight_decay, eps=opt.adam_epsilon,)
# optimizer = torch.optim.AdamW(layers_to_opt, lr=opt.lr, betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-8)

#load checkpoint
if opt.continue_train:
    try:
        start_epoch_, diffmodel = load_checkpoint(diffmodel, opt.diffhdr_ckpt_path)
        start_epoch, mainmodel = load_checkpoint(mainmodel, opt.mainhdr_ckpt_path)
    except Exception as e:
        print(e)
        print("Checkpoint is empty! Training reset!")
        start_epoch = 1
        start_epoch_, diffmodel = load_checkpoint(diffmodel, opt.diffhdr_ckpt_path)
        mainmodel.apply(weights_init)
else:
    start_epoch = 1
    start_epoch_, diffmodel = load_checkpoint(diffmodel, opt.diffhdr_ckpt_path)
    mainmodel.apply(weights_init)

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

l1 = torch.nn.L1Loss()
perceptual_loss = PerceptualLoss().to(opt.gpu_ids[0]) 

make_required_directories(mode="train")

#  begin to train
for epoch in range(start_epoch, opt.epochs + 1):

    epoch_start = time.time()
    running_loss = 0

    if epoch > opt.lr_decay_after:
        update_lr(optimizer_a, epoch, opt)
        update_lr(optimizer_b, epoch, opt)
    print("Epoch: ", epoch)

    l1_sdr_loss_sum = 0
    l1_hdr_loss_sum = 0
    l1_hdr_drago_tonemap_loss_sum = 0
    mse_sdr_loss_sum = 0
    mse_hdr_loss_sum = 0
    ssim_sdr_loss_sum = 0
    ssim_hdr_loss_sum = 0
    perceptual_loss_sum = 0
    idx = 0

    for batch, data in enumerate(tqdm(data_loader, desc="Batch %")):

        optimizer_a.zero_grad()
        optimizer_b.zero_grad()

        input = data["ldr_image"].data.cuda() # 0-1
        gt = data["hdr_image"].data.cuda() # # 0-2^8

        drago_tonemap_gt = drago_tonemap(gt)
        input = F.normalize(input, mean=[0.5], std=[0.5])
        drago_tonemap_gt = F.normalize(drago_tonemap_gt, mean=[0.5], std=[0.5])
        
        gt_ram = ram_transforms(drago_tonemap_gt*0.5+0.5)
        caption = inference(gt_ram.to(dtype=torch.float32), model_vlm)
        prompt = [f'{each_caption}' for each_caption in caption]
        # print("Prompt:{}".format(prompt))

        sdr_out, _ = diffmodel(input, prompt) # output > 0 !!
        hdr_out = mainmodel(input, sdr_out) # -1-1

        sdr_out = sdr_out * 0.5 + 0.5
        drago_tonemap_gt = drago_tonemap_gt * 0.5 + 0.5

        hdr_out = (hdr_out * 0.5 + 0.5) * (2**8) # 0-2^8
        hdr_drago_tonemap = drago_tonemap(hdr_out)

        sdr_out.retain_grad()
        hdr_out.retain_grad()

        # loss init
        l1_sdr_loss = 0
        l1_hdr_loss = 0
        l1_hdr_drago_tonemap_loss = 0
        mse_sdr_loss = 0
        mse_hdr_loss = 0
        ssim_sdr_loss = 0
        ssim_hdr_loss = 0

        l1_sdr_loss = torch.mean(l1(sdr_out, drago_tonemap_gt))
        # l1_hdr_loss = torch.mean(l1(hdr_out, gt)) ###############################################消融！
        l1_hdr_loss = torch.mean(l1(hdr_out/hdr_out.max(), gt/gt.max()))
        l1_hdr_drago_tonemap_loss = torch.mean(l1(hdr_drago_tonemap, drago_tonemap_gt))


        # ssim_sdr_loss = 1 - pytorch_msssim.ssim(sdr_out, drago_tonemap_gt, data_range=1.0)
        # ssim_hdr_loss = 1 - pytorch_msssim.ssim(hdr_drago_tonemap, drago_tonemap_gt, data_range=1.0)
        vgg_loss = torch.mean(perceptual_loss(hdr_drago_tonemap, drago_tonemap_gt)[0])


        # assert not torch.isnan(l1_sdr_loss).any(), "l1_sdr_loss is NaN, terminating training."
        assert not torch.isnan(l1_hdr_loss).any(), "l1_hdr_loss is NaN, terminating training."
        assert not torch.isnan(l1_hdr_drago_tonemap_loss).any(), "l1_hdr_drago_tonemap_loss is NaN, terminating training."
        # assert not torch.isnan(mse_sdr_loss).any(), "mse_sdr_loss is NaN, terminating training."
        # assert not torch.isnan(mse_hdr_loss).any(), "mse_hdr_loss is NaN, terminating training."

        l1_sdr_loss_sum += l1_sdr_loss
        l1_hdr_loss_sum += l1_hdr_loss
        l1_hdr_drago_tonemap_loss_sum += l1_hdr_drago_tonemap_loss
        # mse_sdr_loss_sum += mse_sdr_loss
        # mse_hdr_loss_sum += mse_hdr_loss
        ssim_sdr_loss_sum += ssim_sdr_loss
        ssim_hdr_loss_sum += ssim_hdr_loss
        perceptual_loss_sum += vgg_loss

        num = batch + 1

        # loss_sdr = l1_sdr_loss + 0.5 * ssim_sdr_loss
        loss_sdr = l1_sdr_loss
        # loss_hdr = l1_hdr_loss + 0.5 * l1_hdr_drago_tonemap_loss + 0.4 * ssim_hdr_loss + 0.2 * vgg_loss
        loss_hdr = l1_hdr_loss + 0.5 * l1_hdr_drago_tonemap_loss + 0.2 * vgg_loss
        # loss = loss_hdr
        loss = loss_hdr + 0.5 * loss_sdr

        # loss_sdr.backward()
        # loss_hdr.backward()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diffmodel.parameters(), max_norm=1.0)

        optimizer_a.step()
        optimizer_b.step()
        running_loss += loss.item()
        idx = idx + 1

        # sdr_out = sdr_out * 0.5 + 0.5  # 归一化到 [0, 1]
        # drago_tonemap_gt = drago_tonemap_gt * 0.5 + 0.5
        # hdr_drago_tonemap = hdr_drago_tonemap * 0.5 + 0.5
        

        if (batch+1) % opt.log_after == 0:
            print(
                "Epoch: {} ; Batch: {} ; Training loss: {}".format(epoch, batch + 1, running_loss / opt.log_after)
            )
            running_loss = 0

        if (epoch+1) % 10 == 0:
            if (batch+1) % opt.save_results_after == 0:  # save image results
                save_rgb_image(
                    img_tensor=sdr_out,
                    batch=0,
                    path="./training_results/sdr_gen_e_{}_b_{}.png".format(epoch, batch + 1),
                )
                save_rgb_image(
                    img_tensor=hdr_drago_tonemap,
                    batch=0,
                    path="./training_results/hdr_drago_tonemap_e_{}_b_{}.png".format(epoch, batch + 1),
                )
                save_rgb_image(
                    img_tensor=drago_tonemap_gt,
                    batch=0,
                    path="./training_results/gt_hdr_e_{}_b_{}.png".format(epoch, batch + 1),
                )
                save_rgb_image(
                    img_tensor=(input*0.5+0.5),
                    batch=0,
                    path="./training_results/ldr_e_{}_b_{}.png".format(epoch, batch + 1),
                )
                # save_hdr_image(
                #     img_tensor=hdr_out,
                #     batch=0,
                #     path="./training_results/hdr_generate_e_{}_b_{}.hdr".format(epoch, batch + 1),
                # )

        if (batch+1) % opt.batch_num == 0:
            break

    writer.add_scalar('l1_sdr_loss', l1_sdr_loss_sum/idx, epoch)
    writer.add_scalar('l1_hdr_loss', l1_hdr_loss_sum/idx, epoch)
    writer.add_scalar('l1_hdr_drago_tonemap_loss', l1_hdr_drago_tonemap_loss_sum/idx, epoch)
    # writer.add_scalar('ssim_sdr_loss', ssim_sdr_loss_sum/idx, epoch)
    # writer.add_scalar('ssim_hdr_loss', ssim_hdr_loss_sum/idx, epoch)
    writer.add_scalar('perceptual_loss', perceptual_loss_sum/idx, epoch)


    epoch_finish = time.time()
    time_taken = (epoch_finish - epoch_start) // 60

    print("epoch: {}. Time: {} minutes.".format(epoch, int(time_taken)))

    if epoch % opt.save_ckpt_after == 0:
        save_checkpoint(epoch, diffmodel, "../diffhdr/")
        save_checkpoint(epoch, mainmodel, "../mainhdr/")

writer.close()
print("Training Finished!")