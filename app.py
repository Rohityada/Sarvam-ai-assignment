import streamlit as st
import fitz  # PyMuPDF
import requests
import base64
import json
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import os

# Sarvam API Configuration
SARVAM_API_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_API_KEY = "We  add openai key here"

# OpenAI API Configuration
os.environ["OPENAI_API_KEY"] = "We  add openai key here"

# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to send text to Sarvam API and get audio response
def text_to_speech_sarvam(text):
    # Split text into chunks of 500 characters
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    audio_chunks = []

    for chunk in chunks:
        payload = {
            "inputs": [chunk],
            "target_language_code": "hi-IN",
            "speaker": "meera",
            "pitch": 0,
            "pace": 1.00,
            "loudness": 1.5,
            "speech_sample_rate": 8000,
            "enable_preprocessing": True,
            "model": "bulbul:v1"
        }
        headers = {
            "Content-Type": "application/json",
            "api-subscription-key": SARVAM_API_KEY
        }
        try:
            response = requests.post(SARVAM_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            
            if 'audios' in response_data and response_data['audios']:
                audio_chunks.append(base64.b64decode(response_data['audios'][0]))
            else:
                st.error(f"No audio data in the response for chunk. Response: {response_data}")
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed for chunk: {str(e)}")

    return b''.join(audio_chunks) if audio_chunks else None

# Function to set up RAG model
def setup_rag_model(text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    
    chain = load_qa_chain(ChatOpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff")
    
    return docsearch, chain

# Streamlit Interface
st.title("PDF Question Answering with Speech Conversion")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.write("Extracting text from the uploaded PDF...")
    
    # Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf(uploaded_file)
    
    if pdf_text:
        st.write("Text extracted from PDF. Setting up the RAG model...")
        docsearch, chain = setup_rag_model(pdf_text)
        
        # Question input
        question = st.text_input("Ask a question about the PDF content:")
        
        if question:
            # Get relevant documents
            docs = docsearch.similarity_search(question)
            
            # Generate answer
            answer = chain.run(input_documents=docs, question=question)
            
            st.write("Answer:")
            st.write(answer)
            
            # Convert answer to speech
            if st.button("Convert Answer to Speech"):
                with st.spinner("Converting answer to speech..."):
                    audio_bytes = text_to_speech_sarvam(answer)
                
                if audio_bytes:
                    st.audio(audio_bytes, format='audio/wav')

                    # Provide download link for audio
                    b64 = base64.b64encode(audio_bytes).decode()
                    href = f'<a href="data:audio/wav;base64,{b64}" download="answer_audio.wav">Download the audio</a>'
                    st.markdown(href, unsafe_allow_html=True)
    else:
        st.error("No text found in the uploaded PDF.")

st.markdown("""
    <style>
    footer {
        visibility: hidden;
    }
    .made-with-love {
        position: fixed;
        bottom: 10px;
        left: 0;
        width: 100%;
        text-align: center;
        color: black;
    }
    </style>
    <div class="made-with-love">
        <p>Made with ❤️ by Rohit Yadav</p>
    </div>
    """, unsafe_allow_html=True)
        
