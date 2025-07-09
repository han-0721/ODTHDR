import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class Load_Data(Dataset):

    def __init__(self, mode, opt):

        self.batch_size = opt.batch_size
        self.mode = mode
        if mode == "train":
            self.dataset_path = os.path.join("../HDR-Real")
            self.ldr_data_path = os.path.join(self.dataset_path, "LDR")
            self.hdr_data_path = os.path.join(self.dataset_path, "HDR")
            self.ldr_image_names = sorted(os.listdir(self.ldr_data_path))
            self.hdr_image_names = sorted(os.listdir(self.hdr_data_path))
        elif mode == "test":
            # self.dataset_path = os.path.join("../autodl-tmp/HDREye")
            # self.dataset_path = os.path.join("../autodl-tmp/Test_HDR_Raise")
            self.dataset_path = os.path.join("../autodl-tmp/Test")
            # self.dataset_path = os.path.join("../autodl-tmp/Test_HDR_SIHDR")
            # self.dataset_path = os.path.join("../autodl-tmp/Test_HDR_Synth")
            # self.dataset_path = os.path.join("../autodl-tmp/test")
            self.ldr_data_path = os.path.join(self.dataset_path, "LDR")
            self.hdr_data_path = os.path.join(self.dataset_path, "HDR")
            self.ldr_image_names = sorted(os.listdir(self.ldr_data_path))
            self.hdr_image_names = sorted(os.listdir(self.hdr_data_path))
        else: # infer
            # self.dataset_path = os.path.join("./Test_HDR_Raise")
            # self.dataset_path = os.path.join("./SI-HDR/")
            # self.dataset_path = os.path.join("../autodl-tmp/HDR_Real")
            self.dataset_path = os.path.join("../AIM_TEST")
            # self.dataset_path = os.path.join("./HDR_Synth")
            # self.dataset_path = os.path.join("./Raise")
            self.ldr_data_path = os.path.join(self.dataset_path, "LDR")
            self.ldr_image_names = sorted(os.listdir(self.ldr_data_path))

    def __getitem__(self, index):

        # LDR load
        self.ldr_image_path = os.path.join(
            self.ldr_data_path, self.ldr_image_names[index]
        )
        ldr_sample = cv2.imread(self.ldr_image_path, -1).astype(np.float32)
        ldr_sample = cv2.cvtColor(ldr_sample, cv2.COLOR_BGR2RGB)
        ldr_sample = ldr_sample/255.0
        transform_list = [
            transforms.ToTensor(),
        ]
        transform_ldr = transforms.Compose(transform_list)
        ldr_tensor = transform_ldr(ldr_sample)

        if (self.mode == "train") or (self.mode == "test"):
            self.hdr_image_path = os.path.join(
                self.hdr_data_path, self.hdr_image_names[index]
            )
            if self.hdr_image_path.split(".")[-1] == "exr":
                os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
                hdr_sample = cv2.imread(self.hdr_image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
            elif self.hdr_image_path.split(".")[-1] == "hdr":
                hdr_sample = cv2.imread(self.hdr_image_path, -1).astype(np.float32)
                
            hdr_sample = cv2.cvtColor(hdr_sample, cv2.COLOR_BGR2RGB)
                
            hdr_sample = (hdr_sample/ hdr_sample.max() * (2**8)).astype(np.float32) # 0-2^8
            transform_list = [
                transforms.Lambda(lambda img: torch.from_numpy(img.transpose((2, 0, 1)))),  # can't use the ToTensor
            ]
            transform_hdr = transforms.Compose(transform_list)
            hdr_tensor = transform_hdr(hdr_sample)

        #if self.mode == "train":
            #pass
        #if self.mode == "test":
            #resize = transforms.Resize((512, 512))
            #ldr_tensor = resize(ldr_tensor)
            #hdr_tensor = resize(hdr_tensor)
        #else: # infer
            #print(ldr_tensor.shape)
            #c, h, w = ldr_tensor.shape
            #resize = transforms.Resize((h//8*8, w//8*8))
            # resize = transforms.Resize((512, 512))
            #ldr_tensor = resize(ldr_tensor)

        if (self.mode == "train") or (self.mode == "test"):
            data_dict = {
                "ldr_image": ldr_tensor,
                "hdr_image": hdr_tensor,
                "path": self.ldr_image_path,
            }
        else:
            data_dict = {
                "ldr_image": ldr_tensor,
                "path": self.ldr_image_path,
            }
        return data_dict

    def __len__(self):
        return len(self.ldr_image_names) // self.batch_size * self.batch_size
