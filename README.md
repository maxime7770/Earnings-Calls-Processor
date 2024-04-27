# Earning-Calls-Processor

This project was developed as part of the "Unleashing Novel Data st Scale" at Harvard. It is a collaborative effort by team members [Maxime Wolf](https://www.linkedin.com/in/maxime-wolf/), [Dilan SriDaran](https://www.linkedin.com/in/dilansridaran/) and [Nuobei Zhang](https://www.linkedin.com/in/nuobeizhang/).

You can read the complete report for this project [here](https://maximewolf.com/assets/pdf/Unleashing_Novel_Data.pdf)

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



## Problem

Traditional stock price prediction models largely depend on historical data and market indicators, but often overlook the rich, unstructured multimedia data available from earnings calls. This project seeks to bridge this gap by utilizing both textual and audio content from earnings calls, which offer deep insights into a company's performance and future outlook. The goal is to build a novel dataset to enhance prediction models that can better reflect the potential stock price movements influenced by these insights.


## Data

The original dataset includes:
- **80 quarterly earnings call transcripts** from major technology companies: Apple, Google, Microsoft, and Nvidia, spanning from January 2018 to April 2024.
- **Sources:** Yahoo Finance for daily stock price data; London Stock Exchange Group for transcripts and audio recordings.
- **Format:** Text transcripts and audio recordings (in `.txt` and `.mp3` formats respectively), segmented into unique utterances for detailed analysis.

## Methods

The project employs a multi-modal analytical pipeline that integrates text and audio data processing with machine learning models:

- **Text Analysis:**
  - **Sentiment Analysis:** Using FinBERT to assess sentiment scores within the transcripts.
  - **Topic Modeling:** Dynamic topic identification with BERTopic to uncover prevalent themes.
  - **Readability Scores:** Evaluation of text clarity and comprehension difficulty.

- **Audio Analysis:**
  - **Speech Characteristics:** Analysis of tempo, tone, and other vocal attributes using the librosa library.

- **Feature Engineering:**
  - Extraction of comprehensive features from both text and audio data, creating a rich dataset for modeling.

- **Predictive Modeling:**
  - Integration of extracted features into machine learning models to predict stock price movements post-earnings calls.



## Conclusion and Future Work

The preliminary results from this study show promising capabilities in utilizing advanced NLP techniques to extract meaningful insights from earnings call transcripts and audio data.

### Next Steps:
- **Data Expansion:** We aim to significantly increase the dataset by including more earnings calls from a diverse range of companies and extending the temporal coverage beyond the current dataset.
- **Model Enhancement:** Future iterations will explore the integration of additional predictive models and perhaps leveraging larger pre-trained models for deeper semantic analysis.
- **Validation and Refinement:** The models will undergo further validation and refinement to improve accuracy and reliability, especially in dynamically changing market conditions.

### Discussions:
- **Interpretability vs. Performance:** One of the key discussions revolves around balancing the interpretability of the models with their predictive performance. As models become more complex, ensuring that their outputs remain understandable to users becomes challenging.
- **Mitigating Bias:** The potential for look-ahead bias is a significant concern, especially as pre-trained models are used. Future work will focus on ensuring that the data used for training does not include future leakage that could artificially enhance performance.
