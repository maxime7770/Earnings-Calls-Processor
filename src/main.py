import pandas as pd
import helpers
import text_feature_extraction
import preprocessing
import sentiment_analysis
import audio_segmentation
import audio_processing


def preprocess(directory):
    return preprocessing.main(directory)


def segment_audios(directory):
    audio_segmentation.split_all(directory)


def attribute_and_similarity(file):

    data = pd.read_csv(file, index_col=False)

    test_ppg = data['text'][2]
    helpers.get_attribute(test_ppg)

    attribute_df = helpers.make_att_df(data)

    similarity_df = helpers.extract_similarity_score(data)

    merged_df = pd.merge(attribute_df, similarity_df, on=['company_name', 'date'], how='inner')
    merged_df = merged_df.drop('summary', axis=1)

    merged_df.to_csv('OutputData/attribute_and_similarity.csv')

    return merged_df


def readibility_and_topics(file):
    df = pd.read_csv(file)

    # Add unique ID
    df.reset_index(inplace = True)
    df.rename(columns = {'index': 'ID'}, inplace = True)
    df.head()

    # Calculate readability metrics
    df = text_feature_extraction.calculate_readability_metrics(df)

    # Topics
    sentence_df = text_feature_extraction.create_sentence_topics(df)
    
    df.to_csv('OutputData/readibility.csv')
    sentence_df.to_csv('OutputData/sentence_topics.csv')

    return df, sentence_df


def sentiment_features(file, sentence_df):
    df = pd.read_csv(file)

    df_sentiment = sentiment_analysis.get_sentiment_bert(df)
    df_sentiment = sentiment_analysis.get_sentiment_topic(df_sentiment)
    df_sentiment = sentiment_analysis.aggregate_sentiment(df_sentiment)

    sentence_df = sentiment_analysis.topic_modeling_sentiment(sentence_df, df)

    merged_df = sentiment_analysis.merge_sentiment_dataframes(df_sentiment, sentence_df)

    merged_df.to_csv('OutputData/sentiment_features.csv')

    return merged_df


def audio_features(directory):
    audio_features = audio_processing.get_features(directory, librosa_=True, wave2vec=False, embeddings=False)
    df = pd.DataFrame(audio_features)

    df.rename(columns={'company_1': 'company_name', 'date_1': 'date'}, inplace=True)

    for i in range(2, 16):
        df = df.drop(['company_' + str(i), 'date_' + str(i)], axis=1)

    df.to_csv('OutputData/audio_features.csv')

    return df




if __name__ == '__main__':
    output = preprocess('../EarningCallData/')
    print('Preprocessing done')
    path_output = '../OutputData/output.csv'
    output.to_csv(path_output)

    segment_audios('../Audio/')

    df1 = attribute_and_similarity(path_output)
    df2, sentence_df = readibility_and_topics(path_output)

    df3 = sentiment_features(path_output, sentence_df)

    df4 = audio_features('../AudioData/')

    # Merge all dataframes
    df = pd.merge(df1, df2, on=['company_name', 'date'], how='inner')
    df = pd.merge(df, df3, on=['company_name', 'date'], how='inner')
    df = pd.merge(df, df4, on=['company_name', 'date'], how='inner')

    df.drop(['Unnamed: 0.1'], axis=1, inplace=True)

    df.to_csv('OutputData/final_output.csv')

    print('Done')






