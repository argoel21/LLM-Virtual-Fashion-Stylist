# 👗 Virtual Fashion Stylist: LLM-Based Outfit Recommender

## 📌 Project Overview
This project aims to create a **virtual fashion stylist** that leverages **large language models (LLMs)** and **fashion image data** to provide personalized outfit recommendations. Users will be able to input their wardrobe (via images or descriptions), receive outfit suggestions tailored to their **preferences, the weather**, and **current trends**, and interact with the system via a chatbot or web interface.

## ✨ Key Features
- 👕 **Wardrobe Intake:** Accepts user-uploaded images or text descriptions of clothing items.
- 📸 **Image Recognition:** Uses the Fashion MNIST dataset to identify and label basic clothing categories.
- 🌤️ **Context Awareness:** Integrates **weather data** (via National Weather Services API) to suggest seasonally appropriate outfits.
- 📈 **Trend Integration:** Uses **Pinterest Trends** to reflect real-time fashion preferences.
- 💬 **LLM-Powered Recommendations:** Uses OpenAI’s GPT models to generate personalized, context-aware outfit suggestions.

## 📊 Datasets & APIs
- **Fashion MNIST** – for training the model to identify clothing types.
- **Pinterest Trends** – to capture fashion trends.
- **National Weather Service API** – to bring in real-time weather context.

> **Note:** Due to API limitations, personal Pinterest boards are no longer used.

## 🔧 Tech Stack
- BLIP-2 and CLIP Models (for clothing captioning + text embedding)
- ChromaDB
- OpenAI Chat Models
- Open-Meteo Weather API
- TensorFlow/Keras (for Fashion MNIST)
- React (planned for user interface)

## 🚧 Current Status
- ✅ Defined project scope
- ✅ Acquired API key from OpenAI
- ✅ Outlined agent workflow
- ✅ Set up RAG workflow & prompting logic
- 🔜 Building user interface

## 🔜 Next Steps
1. Build chatbot/web interface for user interaction.
