import src.sentiment_analysis as sentiment_analysis
import pandas as pd

# df = pd.read_csv('OutputData/sentence_level_output.csv')

# df_sentiment = sentiment_analysis.get_sentiment_bert(df)
# df_sentiment.to_csv('OutputData/sentiment_sentence_level.csv')
# df_sentiment = sentiment_analysis.get_sentiment_topic(df_sentiment)
# df_sentiment = sentiment_analysis.aggregate_sentiment(df_sentiment)

# print(df_sentiment.head())

# df_sentiment.to_csv('sentiment_features.csv')


# df1 = pd.read_csv('sentiment_features.csv')
# df2 = pd.read_csv('archives/sentence_with_topics_sentiment_aggregated.csv')

# df = pd.merge(df1, df2, on=['company_name', 'date'], how='inner')

# df.to_csv('test.csv')



# import pandas as pd


df1 = pd.read_csv('OutputData/sentiment_features.csv')
df4 = pd.read_csv('OutputData/features_audio.csv')
df2 = pd.read_csv('OutputData/readability.csv')
df3 = pd.read_csv('OutputData/attribute_and_similarity.csv')

df4.rename(columns={'company_1': 'company_name', 'date_1': 'date'}, inplace=True)
for i in range(2, 16):
    df4 = df4.drop(['company_' + str(i), 'date_' + str(i)], axis=1)


# merge all dataframes
df = pd.merge(df1, df2, on=['company_name', 'date'], how='inner')
df = pd.merge(df, df3, on=['company_name', 'date'], how='inner')
df = pd.merge(df, df4, on=['company_name', 'date'], how='inner')


df.to_csv('OutputData/final_output.csv')
