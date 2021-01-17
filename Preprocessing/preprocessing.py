import pandas as pd
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
# nltk.download()
from greek_stemmer import GreekStemmer
from cltk.stem.lemma import LemmaReplacer
from cltk.corpus.utils.formatter import cltk_normalize
from cltk.lemmatize.greek.backoff import BackoffGreekLemmatizer
from cltk.corpus.utils.importer import CorpusImporter
import spacy


def import_additional_greek_stopwords(stopwords_greek):
    stopwords_greek.add('της')
    # stopwords_greek.add('απο')
    stopwords_greek.add('απο')
    stopwords_greek.add('ειναι')
    stopwords_greek.add('τους')
    stopwords_greek.add('τη')
    stopwords_greek.add('μας')
    stopwords_greek.add('στα')
    stopwords_greek.add('στις')
    stopwords_greek.add('στους')
    stopwords_greek.add('μου')
    # stopwords_greek.add('κυβερνηση')

    return stopwords_greek


def remove_intonation(tweet):

    rep = {"ά": "α", "έ": "ε", "ή": "η", "ί": "ι", "ό": "ο", "ύ": "υ", "ώ": "ω", "ϊ": "ι"}

    rep = dict((nltk.re.escape(k), v) for k, v in rep.items())
    pattern = nltk.re.compile("|".join(rep.keys()))
    tweet = pattern.sub(lambda m: rep[nltk.re.escape(m.group(0))], tweet)

    return tweet


def tweet_preprocessing(df, stemming, lemmatization):

    # STOPWORDS
    stopwords_greek = set(stopwords.words('greek'))
    stopwords_greek = import_additional_greek_stopwords(stopwords_greek)

    # STEMMER
    stemmer = GreekStemmer()

    # LEMMATIZATION
    corpus_importer = CorpusImporter('greek')
    # corpus_importer.import_corpus('greek_models_cltk')
    # corpus_importer.import_corpus('greek_training_set_sentence_cltk')
    # print(corpus_importer.list_corpora)

    # lemmatizer = LemmaReplacer('greek')
    spacy_lemmatizer = spacy.load('el_core_news_lg')

    list_of_tweets = []
    for i in range(0, len(df)):

        tweet = str(df.iloc[i]['text'])

        # remove accents
        tweet = remove_intonation(tweet)

        # tokenization
        tokens = tweet.split()

        # if len(tokens) > 36 or len(tokens) < 3:
        #     df.loc[i, 'label'] = 10

        # remove stopwords
        if stemming is True:
            words = [stemmer.stem(w.upper()) for w in tokens if w not in stopwords_greek]
            tweet_clean = ' '.join([w for w in words])
            tweet_clean = tweet_clean.lower()
        else:
            words = [w for w in tokens if w not in stopwords_greek]
            tweet_clean = ' '.join([w for w in words])

            if lemmatization is True:
                # lemmatization
                tweet_lemmas = spacy_lemmatizer(tweet_clean)
                lemmas = [t.lemma_ for t in tweet_lemmas]
                tweet_clean = ' '.join([l for l in lemmas])

        print(tweet_clean)
        list_of_tweets.append(tweet_clean)

    df['tweet'] = list_of_tweets
    return df


if __name__ == '__main__':

    dfs = pd.read_excel('twitter_preproc.xlsx',  engine='openpyxl', header=None)

    # keep only Tweet + Sentiment
    df = dfs.iloc[:, [1, 2]]
    df.columns = ['text', 'label']

    # drop unlabeled rows
    df = df[df['label'].notna()]

    # drop tweets irrelavant to Covid19
    df = df[df['label'] < 10]

    df = tweet_preprocessing(df, True, False)

    df = df.drop(['text'], axis=1)

    # change column order
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    # change data type of Label
    df['label'] = df['label'].astype(int)

    print(df.head())

    # write to csv
    # df.to_csv('tweets_preprocessed_stemming.csv', index=False)



