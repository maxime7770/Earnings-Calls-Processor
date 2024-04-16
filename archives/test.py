import pandas as pd


df1 = pd.read_csv('OutputData/sentiment_features.csv')
df4 = pd.read_csv('OutputData/features_audio.csv')
df2 = pd.read_csv('OutputData/readability.csv')
df3 = pd.read_csv('OutputData/attribute_and_similarity.csv')

df4.rename(columns={'company_1': 'company_name', 'date_1': 'date'}, inplace=True)
for i in range(2, 16):
    df4 = df4.drop(['company_' + str(i), 'date_' + str(i)], axis=1)

df1.drop(['Unnamed: 0'], axis=1, inplace=True)


# merge all dataframes
df = pd.merge(df1, df2, on=['company_name', 'date'], how='inner')
df = pd.merge(df, df3, on=['company_name', 'date'], how='inner')
df = pd.merge(df, df4, on=['company_name', 'date'], how='inner')

df.drop(['Unnamed: 0.1'], axis=1, inplace=True)

df.to_csv('OutputData/final_output.csv')
