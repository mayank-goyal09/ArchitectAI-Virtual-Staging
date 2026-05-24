import os
import sys
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DPMSolverMultistepScheduler

# === Hugging Face ZeroGPU Support ===
# ZeroGPU dynamically assigns a GPU ONLY when a @spaces.GPU function is called.
# At module level, PyTorch CUDA is *emulated*, so .to("cuda") works without a real GPU.
try:
    import spaces
    ZERO_GPU = True
    print("✨ Hugging Face ZeroGPU environment detected!")
except ImportError:
    ZERO_GPU = False
    print("💻 Local environment detected. Using local hardware.")
    class spaces:
        @staticmethod
        def GPU(func):
            return func


# === 1. MODEL LOADING AT MODULE LEVEL ===
# ZeroGPU best practice: load models at module level with float16 + "cuda".
# The CUDA emulation layer handles this during startup without needing a real GPU.
# The real GPU is only assigned when a @spaces.GPU decorated function is called.
device = "cuda" if (ZERO_GPU or torch.cuda.is_available()) else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

print(f"🔧 Target device: {device}, dtype: {torch_dtype}")

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

# Memory optimization — critical for staying within ZeroGPU VRAM limits
pipe.enable_attention_slicing()

print("✅ All models loaded and ready!")


# === 2. CORE IMAGE PROCESSING FUNCTIONS ===
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


# === 3. GRADIO INFERENCE HANDLER ===
@spaces.GPU
def ai_interior_designer(input_img, custom_prompt, creativity_level, style_strength):
    if input_img is None:
        raise gr.Error("Please upload an image of your empty room first!")

    if not custom_prompt or not custom_prompt.strip():
        raise gr.Error("Please describe your design vision in the text box!")

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
    result = pipe(
        prompt=full_prompt,
        image=original_resized,
        mask_image=mask_img,
        control_image=skeleton_img,
        num_inference_steps=25,
        controlnet_conditioning_scale=style_strength,
        strength=creativity_level,
        guidance_scale=10.0
    ).images[0]

    print("✅ Room design complete!")
    return result, skeleton_img


# === 4. BUILD GRADIO INTERFACE ===
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏗️ AI Virtual Interior Designer (Production Engine)
    Transform empty rooms into professionally staged spaces using Stable Diffusion and ControlNet.
    
    **Powered by ZeroGPU** — runs on NVIDIA A100 for free!
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

            btn = gr.Button("🎨 Design My Room", variant="primary")

        with gr.Column(scale=1):
            output_img = gr.Image(label="Your Professionally Staged Room")
            skeleton_out = gr.Image(label="Detected Room Outlines")

    # Hook the button click to the designer logic
    btn.click(
        fn=ai_interior_designer,
        inputs=[input_img, prompt, creativity, strictness],
        outputs=[output_img, skeleton_out]
    )

# === 5. RUN THE SERVER ===
if __name__ == "__main__":
    # Get port from environment variable or default to 7860
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )
