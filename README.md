# Earning-Calls-Processor
A tool for transforming earnings calls into structured datasets for predictive downstream analysis using advanced NLP techniques.

## How to run the project

This project uses [Gemma-2b](https://huggingface.co/google/gemma-2b-it), make sure to login to this page with your huggingface account and accept the terms and conditions to have access to the model. Then, generate an access token and save it in a `.env` file in the root of the project as follows:

```
ACCESS_TOKEN = your_token
```

You also need to install the following package, in addition to the library listed in the `requirements.txt` file:

```
python3 -m spacy download en_core_web_sm
```

The main script is `main.py`, you can run it with the following command from the `src` directory:

```
python main.py
```

This will process all the files (transcripts and audio). Make sure that you uploaded the audio files (too large to be pushed to GitHub) to an `Audio` directory. These audio files can be found in this [Google Drive folder](https://drive.google.com/drive/folders/1).

