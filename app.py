import streamlit as st
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os


# Paths (portable)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_EMB = os.path.join(BASE_DIR, "image_embeddings.npy")
IMAGE_PATHS = os.path.join(BASE_DIR, "image_paths.npy")


# Load embeddings
image_embeddings = np.load(IMAGE_EMB)
image_paths = np.load(IMAGE_PATHS, allow_pickle=True)


# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

st.set_page_config(page_title="X-Ray Image Search", layout="wide")
st.title("🩻 X-Ray Image Search Engine")

mode = st.radio("Choose search mode:", ["Text Search", "Image Search"])

def show_results(paths, scores):
    cols = st.columns(5)
    for i, (p, s) in enumerate(zip(paths, scores)):
        with cols[i % 5]:
            full_path = os.path.join(BASE_DIR, p)

            if os.path.exists(full_path):
                st.image(
                    full_path,
                    caption=f"{os.path.basename(p)}\nScore: {float(s):.3f}",
                    use_container_width=True
                )
            else:
                st.warning(f"Missing: {p}")

    

if mode == "Text Search":
    query = st.text_input("Enter text query (e.g., 'chest xray', 'bone fracture', 'spine xray')")

    if st.button("Search") and query.strip():
        inputs = processor(text=[query], return_tensors="pt").to(device)
        with torch.no_grad():
            text_emb = model.get_text_features(**inputs)
            text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
            text_emb = text_emb.cpu().numpy()

        sims = (image_embeddings @ text_emb.T).squeeze()
        top_idx = np.argsort(sims)[::-1][:10]

        st.subheader("Top Results")
        show_results(image_paths[top_idx], sims[top_idx])

elif mode == "Image Search":
    uploaded = st.file_uploader("Upload an X-ray image", type=["jpg", "png", "jpeg"])

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Query Image", width=300)

        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            img_emb = model.get_image_features(**inputs)
            img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
            img_emb = img_emb.cpu().numpy()

        sims = (image_embeddings @ img_emb.T).squeeze()
        top_idx = np.argsort(sims)[::-1][:10]

        st.subheader("Top Similar Images")
        show_results(image_paths[top_idx], sims[top_idx])
