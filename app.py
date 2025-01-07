import streamlit as st
from PIL import Image
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

auth_token = "enter your hugging face token"

# Load the model
@st.cache_resource
def load_model():
    try:
        st.write("Loading model...")
        model_id = "enter your own model"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.write(f"Using device: {device}")
        
        # Modify dtype based on device
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=dtype,
            use_auth_token=auth_token
        )
        pipe.enable_attention_slicing()
        pipe.to(device)
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

# After the prompt input
width = st.slider("Image Width", min_value=256, max_value=768, value=512, step=64)
height = st.slider("Image Height", min_value=256, max_value=768, value=512, step=64)

# Add before the generate button
quality_mode = st.radio(
    "Generation Quality",
    ["Draft (Faster)", "Normal", "High Quality (Slower)"]
)

# Button to generate the image
if st.button("Generate Image"):
    try:
        steps = {
            "Draft (Faster)": 15,
            "Normal": 30,
            "High Quality (Slower)": 50
        }[quality_mode]
        
        with autocast("cuda" if torch.cuda.is_available() else "cpu"):
            image = pipe(
                prompt,
                guidance_scale=8.5,
                num_inference_steps=steps
            ).images[0]
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        
    
    # Save and display the image
    image.save("generated_image.png")
    st.image(image, caption="Generated Image", use_column_width=True)

