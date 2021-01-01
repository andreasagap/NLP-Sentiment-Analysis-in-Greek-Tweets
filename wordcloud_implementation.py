import pandas as pd
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
# nltk.download()

stopwords_greek = set(stopwords.words('greek'))
stopwords_greek.add('της')
stopwords_greek.add('απο')
stopwords_greek.add('από')
stopwords_greek.add('είναι')
stopwords_greek.add('τους')
stopwords_greek.add('τη')
stopwords_greek.add('μας')
stopwords_greek.add('στα')
stopwords_greek.add('στις')
stopwords_greek.add('μου')
stopwords_greek.add('κυβερνηση')
# print(stopwords_greek)

dfs = pd.read_excel('twitter_db.xlsx',  engine='openpyxl', header=None)

df = dfs.iloc[:, [1, 2]]
df.columns = ['text', 'label']

# drop rows without labels
df = df[df['label'].notna()]

# DataFrames per class
neutral_df = df[df["label"] == 1]
optimistic_df = df[df["label"] == 2]
pessimistic1_df = df[df["label"] == 3]
pessimistic2_df = df[df["label"] == 4]

# neut['text'].to_csv('neutral_corpus.txt', sep=' ')

print('# of not NaN: ', len(df))

# print(len(neutral_df))
# print(len(optimistic_df))
# print(len(pessimistic1_df))
# print(len(pessimistic2_df))

full_list = []
for tweet in df['text']:

    rep = {"ά": "α", "έ": "ε", "ή": "η", "ί": "ι", "ό": "ο", "ύ": "υ", "ώ": "ω", "ϊ": "ι"}  # define desired replacements here

    rep = dict((nltk.re.escape(k), v) for k, v in rep.items())
    pattern = nltk.re.compile("|".join(rep.keys()))
    tweet = pattern.sub(lambda m: rep[nltk.re.escape(m.group(0))], tweet)

    word_set = tweet.split()
    # remove stopwords
    words = [w for w in word_set if w not in stopwords_greek]
    print(words)
    for w in words:
        full_list.append(w)


val_counts = pd.Series(full_list).value_counts()
print(val_counts)
val_counts = val_counts.where(val_counts > 50)
print(type(val_counts))
top_words_df = val_counts.to_frame().reset_index()
top_words_df.columns = ['word', 'count']
print(top_words_df.head(10))

# with open('optimistic_corpus.txt', 'w') as f:
#     for item in full_list:
#         if item not in search_words:
#             f.write("%s\n" % item)

# convert list to string
words_string = ' '.join(word for word in full_list)
words = words_string.replace("μητσοτακης", "μητσοτακη")

wordcloud = WordCloud(width=800, height=800,
                      max_words=100,
                      min_word_length=4,
                      background_color='white',
                      # stopwords=stopwords,
                      min_font_size=10).generate(words)

plt.figure(figsize=(8, 8), facecolor=None)
# plt.title('Neutral', fontsize=20)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()


