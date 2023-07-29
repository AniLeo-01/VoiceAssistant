# Voice Assistant using Langchain and LLM

A documentation guide application that serves as a voice assistant for HuggingFace Hub documentations built using Langchain.

## Pre-requisites

Install the dependencies from the requirements.txt file

```
pip install -r requirements.txt
```

## Usage

#### Enter the environment variables in the .env file

```
ACTIVELOOP_TOKEN=<activeloop_token_id>
ELEVEN_API_KEY=<elevenlabs_api_id>
OPENAI_API_KEY=<openai_api_key>
DEEPLAKE_DATASET_PATH=<datalake_path>
```

#### Load documentation data

```
python dataloader.py
```

#### Start Streamlit server

```
streamlit run chat.py
```
