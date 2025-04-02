import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import os
from dotenv import load_dotenv
from tenacity import retry, stop_after_delay, wait_fixed

# Load environment variables
load_dotenv()

# Get authentication token from environment variable
auth_token = os.getenv("HUGGINGFACE_TOKEN")
if not auth_token:
    st.error("HUGGINGFACE_TOKEN not found in environment. Please check your .env file.")
    st.stop()

# Load the model with optimizations
@st.cache_resource
def load_model():
    try:
        st.write("Loading model...")
        model_id = "stabilityai/stable-diffusion-2-1-base"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.write(f"Using device: {device}")
        
        # Use torch.float16 for CUDA, float32 for CPU
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Load the pipeline with several optimizations
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=dtype,
            use_auth_token=auth_token,
            revision="fp16" if dtype == torch.float16 else None,
            safety_checker=None,  # Disable safety checker for speed
            feature_extractor=None  # Not needed for generation
        )
        
        # Apply optimizations
        if device == "cuda":
            pipe.enable_xformers_memory_efficient_attention()  # Faster attention
            pipe.enable_sequential_cpu_offload()  # Better memory management
        else:
            pipe.enable_attention_slicing()  # For CPU/low-memory GPUs
            
        pipe.to(device)
        
        # Warm up the model (optional but can help with first-run speed)
        with torch.inference_mode():
            _ = pipe("warmup", num_inference_steps=1)
            
        st.write("Model loaded successfully!")
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

pipe = load_model()

# Streamlit UI
st.title("Stable Bud - Image Generator")
st.markdown("Enter a prompt and generate stunning AI-generated images.")

# Input text prompt
prompt = st.text_input("Enter your prompt:", "A photo of an astronaut riding a horse on Mars")

# Image dimensions
width = st.slider("Image Width", min_value=256, max_value=512, value=512, step=64)
height = st.slider("Image Height", min_value=256, max_value=512, value=512, step=64)

# Quality settings
quality_mode = st.radio(
    "Generation Quality",
    ["Draft (Faster)", "Normal", "High Quality (Slower)"],
    index=1  # Default to Normal
)

# Button to generate the image
if st.button("Generate Image") and pipe is not None:
    try:
        # Set parameters based on quality mode
        quality_settings = {
            "Draft (Faster)": {"steps": 15, "guidance_scale": 7.0},
            "Normal": {"steps": 25, "guidance_scale": 8.5},
            "High Quality (Slower)": {"steps": 40, "guidance_scale": 9.0}
        }
        settings = quality_settings[quality_mode]
        
        with st.spinner(f"Generating image ({quality_mode})..."):
            with torch.inference_mode():  # More efficient than autocast
                image = pipe(
                    prompt,
                    width=width,
                    height=height,
                    num_inference_steps=settings["steps"],
                    guidance_scale=settings["guidance_scale"]
                ).images[0]
        
        # Display the image
        st.image(image, caption=f'"{prompt}"', use_column_width=True)
        
        # Add download button
        img_bytes = image.tobytes()
        st.download_button(
            label="Download Image",
            data=img_bytes,
            file_name="generated_image.png",
            mime="image/png"
        )
        
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
