import pandas as pd
import numpy as np
import re
from tqdm import tqdm

import nltk

gr_stop = set(nltk.corpus.stopwords.words('greek'))

from gensim.models.fasttext import FastText

from nltk.stem import WordNetLemmatizer


# Text cleaning function for gensim fastText word embeddings in python
def process_text(document):
    # Remove extra white space from text
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Remove all the special characters from text
    document = re.sub(r'\W', ' ', str(document))

    # Remove all single characters from text
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Converting to Lowercase
    document = document.lower()

    # Word tokenization
    tokens = document.split()
    # Lemmatization using NLTK
    lemma_txt = [stemmer.lemmatize(word) for word in tokens]
    # Remove stop words
    lemma_no_stop_txt = [word for word in lemma_txt if word not in gr_stop]
    # Drop words
    tokens = [word for word in tokens if len(word) > 3]

    clean_txt = ' '.join(lemma_no_stop_txt)

    return clean_txt


stemmer = WordNetLemmatizer()


tweets_df = pd.read_excel('twitter_db.xlsx',  engine='openpyxl', header=None)


tweets_df = tweets_df.iloc[:, [1, 2]]
tweets_df.columns = ['text', 'label']
print(tweets_df)
all_sent = list(tweets_df['text'])
some_sent = all_sent[0:1000]


clean_corpus = [process_text(sentence) for sentence in tqdm(some_sent) if sentence.strip() != '']

word_tokenizer = nltk.WordPunctTokenizer()
word_tokens = [word_tokenizer.tokenize(sent) for sent in tqdm(clean_corpus)]
print(word_tokens)

# Defining values for parameters
embedding_size = 100
window_size = 5
min_word = 5
down_sampling = 1e-2

fast_Text_model = FastText(word_tokens,
                           size=embedding_size,
                           window=window_size,
                           min_count=min_word,
                           sample=down_sampling,
                           workers=2,
                           sg=1,
                           iter=100)

# Save fastText gensim model
fast_Text_model.save("ft_model_yelp")
