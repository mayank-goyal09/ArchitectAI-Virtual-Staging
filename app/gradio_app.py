import os
import sys
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DPMSolverMultistepScheduler
from diffusers import StableDiffusionLatentUpscalePipeline

# === Hugging Face ZeroGPU Support ===
# ZeroGPU allows free users to access NVIDIA A100 GPUs for free!
# We import 'spaces' and use @spaces.GPU to decorate our GPU function.
try:
    import spaces
    print("✨ Hugging Face ZeroGPU environment detected!")
except ImportError:
    # Fallback mock decorator for local running on CPU or GPU
    print("💻 Local environment detected. Using local hardware.")
    class spaces:
        @staticmethod
        def GPU(func):
            return func


# === 1. SYSTEM INITIALIZATION & DEVICE DETECTOR ===
# Auto-detect hardware to ensure deployment works on CPU or GPU seamlessly
if torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.float16
    print("🚀 Running on NVIDIA GPU (CUDA Enabled)!")
else:
    device = "cpu"
    torch_dtype = torch.float32
    print("⚠️ CUDA GPU not found. Running on CPU. Generation may be slow.")

# === 2. LOAD AI ENGINES & MODEL CACHING ===
print("📦 Loading ControlNet Canny Model...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", 
    torch_dtype=torch_dtype
)

print("📦 Loading Stable Diffusion Inpaint Pipeline...")
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch_dtype
).to(device)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Load the Latent Upscaler (Optional, can be toggled on/off to save memory)
print("📦 Loading Super-Resolution Latent Upscaler...")
try:
    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
        "stabilityai/sd-x2-latent-upscaler",
        torch_dtype=torch_dtype
    ).to(device)
    has_upscaler = True
except Exception as e:
    print(f"⚠️ Could not load upscaler: {e}. Running without upscaler support.")
    has_upscaler = False


# === 3. CORE IMAGE PROCESSING FUNCTIONS ===
def get_canny_skeleton_from_pil(pil_image):
    """
    Directly processes a PIL Image using the perfected high-contrast logic.
    No need to save to disk!
    """
    # Convert PIL Image to OpenCV BGR format
    img_np = np.array(pil_image)
    if len(img_np.shape) == 2:  # Grayscale check
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 4:  # RGBA check
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 1. Convert to grayscale and equalize histogram
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # 2. Reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Canny edge detection
    edges = cv2.Canny(blurred, 30, 100)

    # 4. Thicken edges for the AI to see better
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Convert back to PIL RGB Image
    return Image.fromarray(cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2RGB))


def create_floor_mask(image_shape):
    """
    Creates a binary floor mask focusing the AI staging on the bottom 50% of the room.
    """
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[int(height * 0.5):, :] = 255
    return Image.fromarray(mask)


# === 4. GRADIO INFERENCE HANDLER ===
@spaces.GPU
def ai_interior_designer(input_img, custom_prompt, creativity_level, style_strength, enable_upscale):
    if input_img is None:
        return None, None

    # Resize input image to standard size for generation (512x512)
    original_resized = input_img.convert("RGB").resize((512, 512))
    
    # 1. Get high-contrast line structure map
    skeleton_img = get_canny_skeleton_from_pil(original_resized)
    
    # 2. Create the floor mask
    mask_img = create_floor_mask(np.array(original_resized).shape)

    # 3. Combine custom prompt with design terms
    full_prompt = f"{custom_prompt}, professional interior design, highly detailed, 8k, photorealistic"

    print(f"🎨 Designing room with prompt: '{full_prompt}'...")
    
    # 4. Generate the staged room
    low_res_result = pipe(
        prompt=full_prompt,
        image=original_resized,
        mask_image=mask_img,
        control_image=skeleton_img,
        num_inference_steps=30,
        controlnet_conditioning_scale=style_strength,  # User controls 'lines'
        strength=creativity_level,                    # User controls 'change'
        guidance_scale=10.0
    ).images[0]

    # 5. Optional super-resolution upscale
    if enable_upscale and has_upscaler:
        print("💎 Polishing and upscaling image...")
        try:
            final_output = upscaler(
                prompt=full_prompt,
                image=low_res_result,
                num_inference_steps=20
            ).images[0]
            return final_output, skeleton_img
        except Exception as e:
            print(f"⚠️ Upscaling failed: {e}. Returning base staged room.")
            return low_res_result, skeleton_img
    
    return low_res_result, skeleton_img


# === 5. BUILD GRADIO INTERFACE ===
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏗️ AI Virtual Interior Designer (Production Engine)
    Transform empty rooms into professionally staged spaces using Stable Diffusion and ControlNet.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Upload Your Empty Room")
            prompt = gr.Textbox(
                placeholder="e.g., A minimalist Scandinavian living room with light wood furniture", 
                label="Describe Your Vision"
            )

            with gr.Accordion("Advanced Designer Settings", open=False):
                creativity = gr.Slider(0.1, 1.0, value=0.8, label="Creativity (How much to change the floor?)")
                strictness = gr.Slider(0.1, 1.0, value=0.6, label="Structural Strictness (Follow wall outlines?)")
                upscale = gr.Checkbox(value=False, label="Enable Super-Resolution Upscaling (Requires more memory)")

            btn = gr.Button("🎨 Design My Room", variant="primary")

        with gr.Column(scale=1):
            output_img = gr.Image(label="Your Professionally Staged Room")
            skeleton_out = gr.Image(label="Detected Room Outlines")

    # Hook the button click to the designer logic
    btn.click(
        fn=ai_interior_designer, 
        inputs=[input_img, prompt, creativity, strictness, upscale], 
        outputs=[output_img, skeleton_out]
    )

# === 6. RUN THE SERVER ===
if __name__ == "__main__":
    # Get port from environment variable or default to 7860
    port = int(os.environ.get("PORT", 7860))
    # In production, we don't share=True, we bind to all interfaces (0.0.0.0)
    demo.launch(
        server_name="0.0.0.0", 
        server_port=port,
        share=False
    )
