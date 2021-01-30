import pandas as pd
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
# nltk.download()
from preprocessing import first_step, remove_intonation, import_additional_greek_stopwords
from numpy import random


def piechart(count_list):
    activities = ['Neutral', 'Optimistic', 'Pessimistic1', 'Pessimistic2']
    colors = ['cornflowerblue', 'limegreen', 'gold', 'red' ]
    plt.pie(count_list, labels=activities, colors=colors, startangle=90, autopct='%.1f%%')
    plt.show()

def count_frequencies(full_list):

    val_counts = pd.Series(full_list).value_counts()
    val_counts = val_counts.where(val_counts > 10)
    top_words_df = val_counts.to_frame().reset_index()
    top_words_df.columns = ['word', 'count']
    print(top_words_df.head(10))

def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(%d, 100%%, 50%%)" % random.randint(35, 120)

def create_word_cloud(full_list, title):

    words_string = ' '.join(word for word in full_list)
    words = words_string.replace("μητσοτακης", "μητσοτακη")

    wordcloud = WordCloud(width=800, height=800,
                          max_words=100,
                          min_word_length=4,
                          background_color='white',
                          # stopwords=stopwords,
                          min_font_size=10).generate(words)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)

    # default_colors = wordcloud.to_array()
    # plt.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=3),
    #            interpolation="bilinear")
    plt.title(title, fontsize=32)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


def basic_preprocessing(df):

    labels = df['label'].tolist()
    df = first_step(df)
    df['label'] = labels
    df.columns = ['text', 'label']

    return df


stopwords_greek = set(stopwords.words('greek'))
stopwords_greek = import_additional_greek_stopwords(stopwords_greek)


dfs = pd.read_excel('Datasets/finalTwitter_2.xlsx',  engine='openpyxl')

# get only text and sentiment
df = dfs.iloc[:, [1, 2]]
df.columns = ['text', 'label']
print("TOTAL: ", len(df))
# drop rows without labels
df = df[df['label'].notna()]

df_del = df[df['label'] == 10]
# print('# of deleted: ', len(df_del))

# drop irrelevant items
df = df[df['label'] < 10]
print('# of labeled: ', len(df))


print(df['label'].value_counts())
print('# of not NaN: ', len(df))

# DataFrames per class
neutral_df = df[df["label"] == 1]
optimistic_df = df[df["label"] == 2]
pessimistic1_df = df[df["label"] == 3]
pessimistic2_df = df[df["label"] == 4]

neutral_df = basic_preprocessing(neutral_df)
optimistic_df = basic_preprocessing(optimistic_df)
pessimistic1_df = basic_preprocessing(pessimistic1_df)
pessimistic2_df = basic_preprocessing(pessimistic2_df)

# neutral_df['text'].to_csv('neutral_corpus.txt', sep=' ')

print('Neutral: ', len(neutral_df))
print('Optimistic: ', len(optimistic_df))
print('Pessimistic1: ', len(pessimistic1_df))
print('Pessimistic2: ', len(pessimistic2_df))

count_list = [len(neutral_df), len(optimistic_df), len(pessimistic1_df), len(pessimistic2_df)]
piechart(count_list)


df_list = [neutral_df, optimistic_df, pessimistic1_df, pessimistic2_df]

for df in df_list:
    words_list = []
    for i in range(0, len(df)):
        init_tweet = str(df.iloc[i]['text'])
        label = str(df.iloc[i]['label'])

        # remove accents
        tweet = remove_intonation(init_tweet)

        # tokenization
        tokens = tweet.split()

        # remove stopwords
        words = [w for w in tokens if w not in stopwords_greek]
        tweet_clean = ' '.join([w for w in words])

        for w in words:
            words_list.append(w)

    count_frequencies(words_list)
    classes = {'1.0':'Neutral', '2.0':'Optimistic','3.0':'Pessimistic 1','4.0':'Pessimistic 2'}
    create_word_cloud(words_list, classes[label])







