import pandas as pd
import ast


def count_frequencies(full_list):

    val_counts = pd.Series(full_list).value_counts()
    # val_counts = val_counts.where(val_counts > 10)
    top_words_df = val_counts.to_frame().reset_index()
    top_words_df.columns = ['word', 'count']
    print(top_words_df.head(10))


dfs = pd.read_excel('datasets/tweets12.xlsx',  engine='openpyxl')
df = dfs[['Target', 'Date', 'Hashtags']]
# df = df[df['Target'] < 10]

hashtags_to_delete = ['covid19', 'κυβερνηση_μητσοτακη', 'κυβερνηση_τσιρκο', 'μητσοτακης', 'νδ_ξεφτιλες',
                      'coronavirus', 'covid19gr', 'covid_19', 'κορωνοιος', 'covid', 'κορονοιος', 'covid19greece',
                      'κορωνοϊός', 'νδ_ξεφτιλες', 'νδ_απατεωνες', 'covid2019', 'covid__19', 'covid--19',
                      'κυβερνηση_συμμορια', 'νδ', 'κυβέρνηση', 'νδ_χουντα', 'νδ_θελατε', 'κυβέρνηση_μητσοτάκη',
                      'covidー19', 'νδ_αλητες', 'κυβερνηση', 'κορονοϊός', 'μητσοτάκης', 'νδ_μαφια', 'συριζα_ξεφτιλες',
                      'naftemporiki', 'κοροναιος', 'syriza_xeftiles', 'lockdown', 'covidiots']

months = ['April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']


month_list = []
for i in range(0, len(df)):
    date = df.iloc[i]['Date']
    date = date.split()[0]
    month = date.split('-')[1]
    hashtags = df.iloc[i]['Hashtags']
    hashtags = ast.literal_eval(hashtags)
    hashtags = ','.join(hashtags)
    df.iat[i, 1] = date
    df.iat[i, 2] = hashtags
    month_list.append(month)


df['Month'] = month_list
# df.drop(['Target'], axis=1)
print(df['Hashtags'])

df_gb = df[['Month', 'Hashtags']].groupby(['Month'])['Hashtags'].transform(lambda x: ','.join(x))
df_gb = df_gb.drop_duplicates()

i = 0
for m in df_gb:
    print(months[i])
    li = list(m.split(","))
    li = [x for x in li if x not in hashtags_to_delete]
    count_frequencies(li)
    i+=1

# df_gb_label = df.groupby(['Month', 'Target']).count()
# df_gb_count_month = df.groupby(['Month']).count()['Target']
#
# count_list = df_gb_count_month.tolist()
# print(count_list)
#
# print(df_gb_label)
#
# df_agg = pd.DataFrame()
