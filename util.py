import os
import cv2
import numpy as np
import torch
import imageio


def load_checkpoint(model, ckpt_path):
    start_epoch = np.loadtxt("../mainhdr/state.txt", dtype=int)
    model.load_state_dict(torch.load(ckpt_path))
    print("Resuming from epoch ", start_epoch)
    return start_epoch, model

def make_required_directories(mode): #create file
    if mode == "train":
        if not os.path.exists("./training_checkpoints"):
            print("Making checkpoints directory")
            os.makedirs("./training_checkpoints")

        if not os.path.exists("./training_results"):
            print("Making training_results directory")
            os.makedirs("./training_results")
    elif mode == "test":
        if not os.path.exists("./test_results"):
            print("Making test_results directory")
            os.makedirs("./test_results")
            os.makedirs("./test_results/generate")
            os.makedirs("./test_results/gt")


# def mu_tonemap(img):
#     """ tonemapping HDR images using μ-law before computing loss """
#     MU = 5000.0
#     return torch.log(1.0 + MU * (img + 1.0) / 2.0) / np.log(1.0 + MU)

def drago_tonemap(tensor, gamma=2.2, saturation=0.8, bias=0.85, delta=1e-3):
    tensor = torch.clamp(tensor, min=0.0)
    # L = 0.2126 R + 0.7152 G + 0.0722 B
    luminance = 0.2126 * tensor[:, 0, :, :] + 0.7152 * tensor[:, 1, :, :] + 0.0722 * tensor[:, 2, :, :]
    # luminance = tensor[:, 0, :, :] + tensor[:, 1, :, :] + tensor[:, 2, :, :]
    luminance = luminance.unsqueeze(1)
    log_luminance = torch.log(luminance + delta)
    log_avg = torch.exp(torch.mean(log_luminance.view(tensor.size(0), -1), dim=1))
    scale = bias / log_avg 
    scale = scale.view(tensor.size(0), 1, 1, 1)
    scaled_luminance = scale * luminance
    L_max = torch.max(scaled_luminance.view(tensor.size(0), -1), dim=1, keepdim=True)[0].view(tensor.size(0), 1, 1, 1)
    scale = torch.clamp(scale, min=1e-6, max=1e6)
    L_max = torch.clamp(L_max, min=1e-6, max=1e6)
    tonemapped_luminance = torch.log1p(scaled_luminance) / torch.log1p(L_max)
    epsilon = 1e-6
    scaling = tonemapped_luminance / (luminance + epsilon)
    tonemapped = tensor * scaling 
    tonemapped = torch.clamp(tonemapped, min=1e-6).pow(1.0 / gamma)
    gray = 0.2126 * tonemapped[:, 0, :, :] + 0.7152 * tonemapped[:, 1, :, :] + 0.0722 * tonemapped[:, 2, :, :]
    # gray = tonemapped[:, 0, :, :] + tonemapped[:, 1, :, :] + tonemapped[:, 2, :, :]
    gray = gray.unsqueeze(1)
    tonemapped = gray + saturation * (tonemapped - gray)
    tonemapped = torch.clamp(tonemapped, 0.0, 1.0)
    # tonemapped = tonemapped * 255.0
    return tonemapped


def write_hdr(path, hdr_image): # rgbe
    """ Writing HDR image in radiance (.hdr) format """
    rgb_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)
    imageio.plugins.freeimage.download()
    imageio.imwrite(path, rgb_image, format='HDR')
    # norm_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB) # convert to RGB (h,w,c)
    # norm_image = (norm_image - norm_image.min()) / (norm_image.max() - norm_image.min()) # normlization
    # with open(path, "wb") as f:
    #     f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n") # write the hdr head information
    #     f.write(b"-Y %d +X %d\n" % (norm_image.shape[0], norm_image.shape[1]))
    #     brightest = np.maximum(np.maximum(norm_image[..., 0], norm_image[..., 1]), norm_image[..., 2]) # the brightest value
    #     mantissa, exponent = np.frexp(brightest) # brightest = mantissa × 2^exponent , mantissa:[0.5, 1)
    #     scaled_mantissa = mantissa * 255.0 / brightest # to (0, 255)
    #     rgbe = np.zeros((norm_image.shape[0], norm_image.shape[1], 4), dtype=np.uint8)
    #     rgbe[..., 0:3] = np.around(norm_image[..., 0:3] * scaled_mantissa[..., None])
    #     rgbe[..., 3] = np.around(exponent + 128) # place the middle of the 256
    #     rgbe.flatten().tofile(f) # RGB * 2^(E-128)
    #     f.close()


# def write_hdr(path, hdr_image): # rgbe
#     """ Writing HDR image in radiance (.hdr) format """
#     cv2.imwrite(path, hdr_image)

def normalize(tensor):
    max = tensor.max()
    min = tensor.min()
    output = (tensor - min)/(max - min + 1e-8)
    return output

def save_hdr_image(img_tensor, batch, path):
    img = img_tensor[batch].detach().cpu().float().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    write_hdr(path, img.astype(np.float32))


def save_ldr_image(img_tensor, batch, path):
    img = img_tensor[batch].detach().cpu().float().numpy()
    img = 255 * (np.transpose(img, (1, 2, 0)) + 1) / 2
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # imread and imwrite using BRG
    cv2.imwrite(path, img)


def save_rgb_image(img_tensor, batch, path): # .png
    img = img_tensor[batch].detach().cpu().numpy().transpose(1, 2, 0) * 255
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)

def save_raw_gray_image(img_tensor, batch, path):
    assert img_tensor.shape[0] == 1, "raw channel needs to be 1!"
    img = img_tensor[batch].squeeze(0).detach().cpu().numpy()
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img = np.zeros_like(img, dtype=np.uint8)
    cv2.imwrite(path, img)
    

# def save_checkpoint(epoch, model):
#     """ Saving model checkpoint """
#     # checkpoint_path = os.path.join("./training_checkpoints", "epoch_" + str(epoch) + ".ckpt")
#     # latest_path = os.path.join("./training_checkpoints", "latest.ckpt")
#     latest_path = os.path.join("../autodl-tmp/", "latest.ckpt")
#     # torch.save(model.state_dict(), checkpoint_path)
#     torch.save(model.state_dict(), latest_path)
#     np.savetxt("./training_checkpoints/state.txt", [epoch + 1], fmt="%d")
#     print("Saved checkpoint for epoch ", epoch)

def save_checkpoint(epoch, model, ckpt_path):
    """ Saving model checkpoint """
    # checkpoint_path = os.path.join(ckpt_path, "epoch_" + str(epoch) + ".ckpt")
    latest_path = os.path.join(ckpt_path, "latest.ckpt")
    # torch.save(model.state_dict(), checkpoint_path)
    torch.save(model.state_dict(), latest_path)
    np.savetxt(os.path.join(ckpt_path, "state.txt"), [epoch + 1], fmt="%d")
    print("Saved checkpoint for epoch ", epoch)


def update_lr(optimizer, epoch, opt):
    """ Linearly decaying model learning rate after specified (opt.lr_decay_after) epochs """
    new_lr = opt.lr - opt.lr * (epoch - opt.lr_decay_after) / (opt.epochs - opt.lr_decay_after)

    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    print("Learning rate decayed. Updated LR is: %.6f" % new_lr)
