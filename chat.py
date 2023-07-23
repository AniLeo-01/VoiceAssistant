import whisper
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from streamlit_chat import message
from elevenlabs import generate
from dotenv import load_dotenv
from dataloader import load_deeplake

from langchain.embeddings import OpenAIEmbeddings


load_dotenv()
eleven_api_key = os.environ.get("ELEVEN_API_KEY")

TEMP_AUDIO_PATH = "temp_audio.wav"
AUDIO_FORMAT = "audio/wav"


def transcribe_audio(file_path):
    try:
        model = whisper.load_model("base.en")
        response = model.transcribe(file_path, verbose=None)
        return response["text"]
    except Exception as e:
        print(f"Error calling the Whisper API: {str(e)}")
        return None


def record_and_transcribe_audio():
    audio_bytes = audio_recorder()
    transcription = None
    if audio_bytes:
        st.audio(audio_bytes, format=AUDIO_FORMAT)

        with open(TEMP_AUDIO_PATH, "wb") as f:
            f.write(audio_bytes)

        if st.button("Transcribe"):
            transcription = transcribe_audio(TEMP_AUDIO_PATH)
            os.remove(TEMP_AUDIO_PATH)
            display_transcription(transcription)

    return transcription


def display_transcription(transcription):
    if transcription:
        st.write(f"Transcription: {transcription}")
        with open("audio_transcription.txt", "w+") as f:
            f.write(transcription)
    else:
        st.write("Error transcribing audio!")


def get_user_input(transcription):
    return st.text_input("", value=transcription if transcription else "", key="input")


def search_db(user_input, db):
    print(user_input)
    retriever = db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 100
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 4
    model = ChatOpenAI(model_name="gpt-3.5-turbo")
    qa = RetrievalQA.from_llm(model, retriever=retriever, return_source_documents=True)
    return qa({"query": user_input})

def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i], key=str(i))
        # Voice using Eleven API
        voice = "Bella"
        text = history["generated"][i]
        audio = generate(text=text, voice=voice, api_key=eleven_api_key)
        st.audio(audio, format="audio/mp3")

def run_streamlit():
    # Initialize Streamlit app with a title
    st.title(" Voice Assistant 🧙")
   
    # Load embeddings and the DeepLake database
    db = load_deeplake(embedding_function=OpenAIEmbeddings())

    # Record and transcribe audio
    transcription = record_and_transcribe_audio()

    # Get user input from text input or audio transcription
    user_input = get_user_input(transcription)

    # Initialize session state for generated responses and past messages
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]
        
    # Search the database for a response based on user input and update the session state
    if user_input:
        output = search_db(user_input, db)
        st.session_state.past.append(user_input)
        response = str(output["result"])
        st.session_state.generated.append(response)

    #Display conversation history using Streamlit messages
    if st.session_state["generated"]:
        display_conversation(st.session_state)

# Run the main function when the script is executed
if __name__ == "__main__":
    run_streamlit()
