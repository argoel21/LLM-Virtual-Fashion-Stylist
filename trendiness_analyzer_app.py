import streamlit as st
from PIL import Image
import random
from transformers import BlipProcessor, BlipForConditionalGeneration
from openai import OpenAI
import pandas as pd
import torch

# --- Load models ---
@st.cache_resource
def load_blip():
    model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to("cpu")
    return processor, model

processor, blip_model = load_blip()

# --- Load Pinterest trends CSV ---
@st.cache_data
def load_trends():
    df = pd.read_csv("data/pinterest_trends.csv", skiprows=10, engine="python", encoding="utf-8", on_bad_lines='warn')
    df = df[df.columns[:5]]  # Simplify for display
    return df

trends_df = load_trends()

# --- Caption with BLIP ---
def caption_image(image):
    inputs = processor(image, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = blip_model.generate(**inputs, max_length=50)
    return processor.decode(outputs[0], skip_special_tokens=True)

# --- Trend analysis with Mistral ---
def analyze_trendiness(caption, trend_examples):
    trend_context = ", ".join(trend_examples)
    prompt = f"""
This outfit was detected in a photo: "{caption}".
Based on current Pinterest fashion trends — like {trend_context} — analyze whether this outfit is trendy or not, and explain why.
"""
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-fake-key")
    response = client.chat.completions.create(
        model="ConfidentialMind/Mistral-Small-24B-Instruct-2501_GPTQ_G32_W4A16",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# --- Streamlit UI ---
st.set_page_config(page_title="Is This Outfit Trendy?", page_icon="👗")
st.title("🌟 Is This Outfit Trendy?")
st.caption("Upload an outfit photo and get trend insights based on Pinterest trends and a vision-language model.")

uploaded_file = st.file_uploader("Upload an outfit image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing outfit..."):
        caption = caption_image(image)
        sample_trends = trends_df["Trend"].dropna().sample(5).tolist()
        analysis = analyze_trendiness(caption, sample_trends)

    st.subheader("📅 Outfit Description")
    st.markdown(f"`{caption}`")

    st.subheader("🧐 Trend Insight")
    st.markdown(analysis)

    st.subheader("💡 Trends Referenced")
    for trend in sample_trends:
        st.markdown(f"- {trend}")