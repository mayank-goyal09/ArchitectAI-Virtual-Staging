import os
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DPMSolverMultistepScheduler

# === ZeroGPU Support ===
try:
    import spaces
    ZERO_GPU = True
    print("✨ ZeroGPU environment detected!")
except ImportError:
    ZERO_GPU = False
    print("💻 Local environment detected.")
    class spaces:
        @staticmethod
        def GPU(func):
            return func

# === LOAD MODELS ON CPU (works everywhere) ===
# Models are loaded in float16 on CPU. They get moved to GPU inside @spaces.GPU.
print("📦 Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

print("📦 Loading Stable Diffusion Pipeline...")
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()

# Only move to CUDA if a real GPU exists (not ZeroGPU — that happens inside @spaces.GPU)
if not ZERO_GPU and torch.cuda.is_available():
    pipe = pipe.to("cuda")
    print("🚀 Moved pipeline to local GPU.")
else:
    print("✅ Models loaded on CPU. Will use GPU inside @spaces.GPU calls.")


# === IMAGE PROCESSING ===
def get_canny_skeleton(pil_image):
    img_np = np.array(pil_image)
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.equalizeHist(cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY))
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return Image.fromarray(cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB))


def create_floor_mask(shape):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[int(h * 0.5):, :] = 255
    return Image.fromarray(mask)


# === INFERENCE (GPU requested here) ===
@spaces.GPU
def ai_interior_designer(input_img, custom_prompt, creativity_level, style_strength):
    if input_img is None:
        raise gr.Error("Please upload an image first!")
    if not custom_prompt or not custom_prompt.strip():
        raise gr.Error("Please describe your design vision!")

    # Move pipeline to GPU at inference time (ZeroGPU gives us a real GPU here)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    original = input_img.convert("RGB").resize((512, 512))
    skeleton = get_canny_skeleton(original)
    mask = create_floor_mask(np.array(original).shape)
    prompt = f"{custom_prompt}, professional interior design, highly detailed, 8k, photorealistic"

    print(f"🎨 Generating: '{prompt}'...")
    result = pipe(
        prompt=prompt,
        image=original,
        mask_image=mask,
        control_image=skeleton,
        num_inference_steps=25,
        controlnet_conditioning_scale=style_strength,
        strength=creativity_level,
        guidance_scale=10.0
    ).images[0]

    print("✅ Done!")
    return result, skeleton


# === GRADIO UI ===
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏗️ AI Virtual Interior Designer
    Transform empty rooms into professionally staged spaces using Stable Diffusion + ControlNet.
    """)
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Upload Your Empty Room")
            prompt = gr.Textbox(
                placeholder="e.g., A minimalist Scandinavian living room with light wood furniture",
                label="Describe Your Vision"
            )
            with gr.Accordion("Advanced Settings", open=False):
                creativity = gr.Slider(0.1, 1.0, value=0.8, label="Creativity")
                strictness = gr.Slider(0.1, 1.0, value=0.6, label="Structural Strictness")
            btn = gr.Button("🎨 Design My Room", variant="primary")
        with gr.Column(scale=1):
            output_img = gr.Image(label="Your Professionally Staged Room")
            skeleton_out = gr.Image(label="Detected Room Outlines")

    btn.click(
        fn=ai_interior_designer,
        inputs=[input_img, prompt, creativity, strictness],
        outputs=[output_img, skeleton_out]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
