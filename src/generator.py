# src/generator.py
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import MLSDdetector

class StagingEngine:
    def __init__(self):
        # 1. Load the MLSD 'Architectural Line' detector
        self.mlsd = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
        
        # 2. Load the ControlNet Model (Specific for MLSD)
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_mlsd", torch_dtype=torch.float16
        )
        
        # 3. Load the Stable Diffusion Pipeline
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
        ).to("cuda") # Use "cpu" if you don't have a GPU
        
        # Speed up generation
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

    def generate_staging(self, image_path, prompt):
        # Load and process the image into straight lines
        input_image = Image.open(image_path)
        mlsd_image = self.mlsd(input_image)
        
        # Generate the new room!
        output = self.pipe(
            prompt,
            image=mlsd_image,
            num_inference_steps=20,
            controlnet_conditioning_scale=0.8 # Higher = stick closer to room shape
        ).images[0]
        
        return output, mlsd_image