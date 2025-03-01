import os
import time
from typing import Any

import requests
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain.prompts import PromptTemplate
from utils.custom import css_code

# Load environment variables
load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")


def progress_bar(amount_of_time: int) -> None:
    """Simulates a progress bar."""
    progress_text = "Please wait, Generative models hard at work"
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(amount_of_time):
        time.sleep(0.04)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()


def generate_text_from_image(url: str) -> str:
    """Generates text from an image using Hugging Face's BLIP model."""
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    generated_text = image_to_text(url)[0]["generated_text"]

    print(f"IMAGE INPUT: {url}")
    print(f"GENERATED TEXT OUTPUT: {generated_text}")
    return generated_text


def generate_story_from_text(scenario: str) -> str:
    """Generates a detailed story (at least 200 words) from text using an open-source model."""

    prompt_template = f"""
    Create a well-structured, engaging, and meaningful story of at least 200 simple words based on the given image. The story should feel natural, relatable, and include a real-world scenario that aligns with the image, keeping a human-centered approach.

    CONTEXT: {scenario}

    STORY:
    """

    text_generator = pipeline(
        "text-generation",
        model="databricks/dolly-v2-3b",
        max_new_tokens=300,  # Increased to generate a longer output
        trust_remote_code=True
    )

    generated_story = text_generator(prompt_template, do_sample=True, temperature=0.7)[0]["generated_text"]

    print(f"TEXT INPUT: {scenario}")
    print(f"GENERATED STORY OUTPUT: {generated_story}")
    return generated_story


def generate_speech_from_text(message: str) -> None:
    """Converts text to speech using Hugging Face's ESPnet model."""
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    payloads = {"inputs": message}

    response = requests.post(API_URL, headers=headers, json=payloads)
    with open("generated_audio.flac", "wb") as file:
        file.write(response.content)


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(page_title="IMAGE TO STORY GENERATOR", page_icon="🖼️")
    st.markdown(css_code, unsafe_allow_html=True)

    with st.sidebar:
        st.image("img/text-to-speech.webp")
        st.write("---")
        st.write("AI App created by @ Piku")

    st.header("Image To Story Generator")
    uploaded_file = st.file_uploader("Please choose a file to upload", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        progress_bar(100)

        scenario = generate_text_from_image(uploaded_file.name)
        story = generate_story_from_text(scenario)
        generate_speech_from_text(story)

        with st.expander("Generated Image scenario"):
            st.write(scenario)
        with st.expander("Generated short story"):
            st.write(story)

        st.audio("generated_audio.flac")


if __name__ == "__main__":
    main()
