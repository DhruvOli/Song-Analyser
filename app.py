import streamlit as st
import speech_recognition as sr
import google.generativeai as gemini  # Assuming GeminiAI is installed as a library
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
from io import BytesIO
import tempfile
import requests

GEMINI_API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
GEMINI_API_KEY = "AIzaSyBMJT1PyIXrZKH3e3bGuWPDlsStKz5gRPk"  # Replace with your actual API key

def convert_to_wav(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    wav_path = os.path.join(tempfile.gettempdir(), "converted_audio.wav")
    audio.export(wav_path, format="wav")
    return wav_path

def transcribe_audio(audio_file_path):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = r.record(source)
        try:
            text = r.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand the audio"
        except sr.RequestError as e:
            return "Error occurred; {0}".format(e)

def gemini_qa(prompt, question):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"{prompt}\n\nQuestion: {question}"
                    }
                ]
            }
        ]
    }
    params = {
        "key": GEMINI_API_KEY
    }
    response = requests.post(GEMINI_API_ENDPOINT, json=data, headers=headers, params=params)
    if response.status_code == 200:
        try:
            response_json = response.json()
            candidates = response_json.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    generated_answer = parts[0].get("text", "Failed to get response from Gemini AI")
                    return generated_answer
                else:
                    return "No generated answer found"
            else:
                return "No candidates found"
        except ValueError:
            return "Failed to parse response from Gemini AI (Malformed response)"
    else:
        return f"Failed to get response from Gemini AI (HTTP error: {response.status_code})"

def main():
    st.set_page_config(page_title="Audio to Text Converter & QA with Gemini AI", page_icon=":microphone:")
    st.markdown(
        """
        <style>
        .stApp {
            background-color: black;
        }
        .stTextInput>div>div>div>input {
            background-color: #e8eaf6;
            border-radius: 5px;
            padding: 8px;
            transition: all 0.3s ease;
        }
        .stTextInput>div>div>div>input:focus {
            outline: none;
            box-shadow: 0 0 5px #6c7b95;
        }
        .stButton>button {
            background-color: #4caf50;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Audio to Text Converter & Question Answering with Gemini AI")

    audio_file = st.file_uploader("Upload Audio File (.wav or .mp3)", type=["wav", "mp3"])

    if audio_file is not None:
        st.audio(audio_file, format='audio/wav')

        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(audio_file.read())
            audio_path = temp_file.name

        # Convert to WAV if not already in that format
        audio_path = convert_to_wav(audio_path)
        transcribed_text = transcribe_audio(audio_path)

        st.header("Results")
        st.subheader("Transcribed Text:")
        st.write(transcribed_text)

        question = st.text_input("Enter your question")

        if st.button("Ask Question") and question:
            st.header("Gemini AI Answer:")
            gemini_answer = gemini_qa(transcribed_text, question)
            st.write(gemini_answer)

if __name__ == "__main__":
    main()
