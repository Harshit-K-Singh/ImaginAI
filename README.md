# Stable Bud - AI-Powered Image Generator 🌌

Stable Bud is a web-based application that uses **Stable Diffusion** to generate stunning AI-generated images based on user-provided text prompts. Built with **Streamlit**, this project provides an intuitive interface for customizing image generation quality and dimensions.

---

## Features
- **Prompt-based Image Generation**: Generate unique images from custom text prompts.
- **Customizable Dimensions**: Select width and height for the generated image (256x256 to 768x768).
- **Quality Modes**: Choose between Draft, Normal, and High-Quality settings for faster or more refined outputs.
- **Device Optimization**: Automatically uses GPU (if available) or CPU for efficient processing.

---

## Tech Stack
- **Streamlit**: Interactive UI and application framework.
- **PyTorch**: Machine learning framework for model handling.
- **Diffusers**: Stable Diffusion implementation for image generation.
- **Pillow**: Image handling and manipulation library.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/stable-bud.git
   cd stable-bud
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add Your Hugging Face Auth Token**:
   - Replace `auth_token` in the script with your **Hugging Face API Token**. 
   - You can obtain the token [here](https://huggingface.co/settings/tokens).

5. **Run the App**:
   ```bash
   streamlit run app.py
   ```

---

## Usage

1. Enter a text prompt (e.g., *"A photo of an astronaut riding a horse on Mars"*).
2. Adjust the image dimensions using the sliders.
3. Select the generation quality: Draft, Normal, or High Quality.
4. Click **Generate Image** to create and display your masterpiece.

---

## File Structure

```
stable-bud/
│
├── app.py              # Main application script
├── requirements.txt    # List of dependencies
├── generated_image.png # Placeholder for the output image
└── README.md           # Project documentation
```

---

## Dependencies

- Python 3.8+
- torch
- diffusers
- Pillow
- streamlit

Install all dependencies via:
```bash
pip install -r requirements.txt
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push the branch:
   ```bash
   git push origin feature-name
   ```
4. Open a pull request and describe your changes.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing pre-trained models.
- [Streamlit](https://streamlit.io/) for making app development seamless.
