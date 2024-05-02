'''
Author: Six_God_K
Date: 2024-04-30 20:30:24
LastEditors: Six_God_K
LastEditTime: 2024-05-02 20:46:12
FilePath: \ComfyUI\custom_nodes\comfyui-sixgod-sweeper\__init__.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''

from .sweeper import clean_object


class Cleaner:
    def __init__(self):
        pass    
    @classmethod
    def INPUT_TYPES(s): 
       return {
            "required": { 
                "image": ("IMAGE",),
                "mask_image": ("IMAGE",),
             },
                      
        }       
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "clear"
    CATEGORY = "image"
    def clear(self,image,mask_image): 
        imgs=clean_object(image,mask_image)
        return (imgs,)
    
 
 

# WEB_DIRECTORY = "./javascript"


NODE_CLASS_MAPPINGS = {
    "SixGod_Sweeper": Cleaner,
    
}

 
NODE_DISPLAY_NAME_MAPPINGS = {
    "SixGod_Sweeper": "SixGod_清道夫 🀞",
}


 