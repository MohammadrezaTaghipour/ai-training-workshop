from email.mime import audio

#region Part 1 . Voice to Text
# https://huggingface.co/
# current model count: 3,093,637


#endregion


#region
import streamlit as st
from transformers import pipeline

MODELS = {
    "persian-v4" : r"C:\Src\ai-training-workshop\models\whisper-persian-v4",
    "persian-large-fa-v1" : r"C:\Src\ai-training-workshop\models\whisper-large-fa-v1"
}

st.sidebar.title("settings")
model_name = st.sidebar.selectbox("Model", MODELS.keys(), 0)
audio_file = st.file_uploader("Audio file", type=["mp3", "wav", "m4a", "ogg"])

if audio_file and st.button("Transcribe"):
    progress_bar = st.progress(0, "0%")
    progress_bar.progress(30, "Uploading...")

    pipe = pipline(
        "automatic-speech-recognition",
        MODELS[model_name],
    )
    progress_bar.progress(60, "60%")

    result = pipe(
        audio_file.read(),
        return_timestamps = True,
        generate_kwargs = {"language_code": "fa"},
    )
    text = result["text"]
    progress_bar.progress(100, "100%")
    st.text_area("Text", text)

#endregion