'''
Author: Six_God_K
Date: 2024-04-30 20:39:38
LastEditors: Six_God_K
LastEditTime: 2024-05-02 20:52:11
FilePath: \ComfyUI\custom_nodes\comfyui-sixgod-sweeper\sweeper.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''

import os,subprocess,sys
import folder_paths
from PIL import Image
from PIL import Image, ImageOps, ImageSequence
import numpy as np
import torch

comfy_path = os.path.dirname(folder_paths.__file__)
MODEL_PATH = os.path.join(comfy_path, 'custom_nodes','comfyui-sixgod-sweeper','models')


# def install_requirements():
#     comfy_path = os.path.dirname(folder_paths.__file__)
#     req_path =os.path.join(comfy_path, 'custom_nodes','comfyui-sixgod-sweeper','requirements.txt')
#     if not os.path.isfile(req_path):
#         return  
#     try:
#         subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', req_path])
#         print("依赖安装完成")
#     except subprocess.CalledProcessError as e:
#         print(f"安装依赖时出错: {e}")

# install_requirements()

try:
    from litelama import LiteLama
    from litelama.model import download_file  
except:
     subprocess.run([sys.executable, '-m', 'pip', 'install', 'litelama'])

from litelama import LiteLama
from litelama.model import download_file 
 
def tensorTo_image(images):
    imgs=[]
    for (batch_number, image) in enumerate(images):
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        imgs.append(img)
    return  imgs

def imageTo_tensor(img):
    output_image_tensors = []
    for i in ImageSequence.Iterator(img):
        i = ImageOps.exif_transpose(i)
        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]    
        output_image_tensors.append(image)
    if len(output_image_tensors) > 1:
        output_image = torch.cat(output_image_tensors, dim=0)
       
    else:
        output_image = output_image_tensors[0]
      
 
    return output_image

 
 
 

def clean_object(image,mask):
    device = "cuda:0"
    Lama = LiteLama2() 
    init_images = tensorTo_image(image)
    mask_image = tensorTo_image(mask)

    res_tensors=[]
    try:
        Lama.to(device)
        for (index, image) in enumerate(init_images):     
           if len(mask_image)==1:
               _mask=mask_image[0]
           else:
               _mask=mask_image[index]          
           predictImg = Lama.predict(image,_mask) 
           tensorImg=imageTo_tensor(predictImg)
           tensorImg=tensorImg.squeeze(dim=0)     
           res_tensors.append(tensorImg)
        newdata= torch.stack(res_tensors, dim=0)
        return newdata
    except:
        pass
    finally:
        Lama.to("cpu")
 


class LiteLama2(LiteLama):
    
    _instance = None
    
    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance
        
    def __init__(self, checkpoint_path=None, config_path=None):
        self._checkpoint_path = checkpoint_path
        self._config_path = config_path
        self._model = None
        
        if self._checkpoint_path is None:        
            checkpoint_path = os.path.join(MODEL_PATH, "big-lama.safetensors")
            if  os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path):
                pass
            else:
                download_file("https://huggingface.co/anyisalin/big-lama/resolve/main/big-lama.safetensors", checkpoint_path)
                
            self._checkpoint_path = checkpoint_path
        
        self.load(location="cpu")