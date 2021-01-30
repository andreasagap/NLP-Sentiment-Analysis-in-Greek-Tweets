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
import re
import wordsegment as ws
from nltk.tokenize import RegexpTokenizer
import numpy as np
# from handle_lexicon import *

ws.load()
# nltk.download('stopwords')

# def stopwords(words):
#     stopwords = set(nltk.corpus.stopwords.words('greek'))
#     filtered_text = [word for word in words if word not in stopwords]
#     return filtered_text


def first_step(csv):
    text_array = []
    for index, row in csv.iterrows():
        text = row[0].lower()
        text = nltk.re.sub(r"http\S+", "", text)
        # text = nltk.re.sub(r"#[α-ωΑ-Ω_]+", "", text)
        text = text.replace("#"," ")
        text = text.replace("_"," ")
        text = text.replace("ー", '')
        text = nltk.re.sub(r"@\S+", "", text)
        text = nltk.re.sub(r"[a-zA-Z0-9]", "", text)
        tokenizer = RegexpTokenizer(r'\w+')
        text = ' '.join(tokenizer.tokenize(text))
        # words = stopwords(nltk.word_tokenize(text))
        # text_array.append(' '.join(words))
        text_array.append(text)

    df = pd.DataFrame({"Text": text_array})
    # df.to_csv('preprocessingDataset.csv', mode='w', index=False, header=True, encoding="utf-8")
    return df


def import_additional_greek_stopwords(stopwords_greek):
    stopwords_greek.add('της')
    # stopwords_greek.add('απο')
    stopwords_greek.add('απο')
    stopwords_greek.add('ειναι')
    stopwords_greek.add('εχει')
    stopwords_greek.add('σας')
    stopwords_greek.add('τους')
    stopwords_greek.add('τη')
    stopwords_greek.add('μας')
    stopwords_greek.add('στα')
    stopwords_greek.add('στις')
    stopwords_greek.add('στους')
    stopwords_greek.add('μου')
    stopwords_greek.add('σου')
    # stopwords_greek.remove('μην')
    # stopwords_greek.remove('δεν')
    # stopwords_greek.remove('δε')
    # stopwords_greek.remove('μη')
    stopwords_greek.remove('μεθ')
    stopwords_greek.add('μια')
    stopwords_greek.add('κυβερνηση')
    stopwords_greek.add('ερνηση')
    stopwords_greek.add('ια')
    stopwords_greek.add('ー')
    stopwords_greek.add('νδ')
    stopwords_greek.add('μητσοτακη')
    stopwords_greek.add('μητσοτακης')
    stopwords_greek.add('κορονοιος')
    stopwords_greek.add('κορωνοιος')
    stopwords_greek.add('κορονοιο')
    stopwords_greek.add('κορωνοιο')
    stopwords_greek.add('κορωνοιου')
    stopwords_greek.add('nan')

    return stopwords_greek


def remove_intonation(tweet):

    rep = {"ά": "α", "έ": "ε", "ή": "η", "ί": "ι", "ό": "ο", "ύ": "υ", "ώ": "ω", "ϊ": "ι",
           "ἀ": "α", "ἐ": "ε", "ἤ": "η", "ἰ": "ι", "ἄ": "α", "ὐ": "υ", "ὡ": "ω", "ὦ": "ω",
           'ὖ': 'υ', 'ὅ': 'ο', 'ῆ': 'η', 'ῇ': 'η', 'ῦ': 'υ', 'ὁ': 'ο', 'ὑ': 'υ', 'ὲ': 'ε',
           'ὺ': 'υ', 'ἂ': 'α', 'ἵ': 'ι', 'ὴ': 'η', 'ὰ': 'α', 'ἅ': 'α', 'ὶ': 'ι', 'ἴ': 'ι',
           'ὸ': 'ο', 'ἥ': 'η', 'ἡ': 'η', 'ὕ': 'υ', 'ἔ': 'ε', 'ἳ': 'ι', 'ὗ': 'υ', 'ἃ': 'α',
           'ὃ': 'ο', 'ὥ': 'ω', 'ὔ': 'υ', 'ῖ': 'ι', 'ἣ': 'η', 'ἷ': 'ι', 'ἑ': 'ε', 'ᾧ': 'ω',
           'ἢ': 'η'}

    rep = dict((nltk.re.escape(k), v) for k, v in rep.items())
    pattern = nltk.re.compile("|".join(rep.keys()))
    tweet = pattern.sub(lambda m: rep[nltk.re.escape(m.group(0))], tweet)

    return tweet


def count_pos_neg_score(words, positives, negatives):
    pos = 0
    neg = 0
    for w in words:
        if w in positives:
            pos += 1
        if w in negatives:
            neg += 1

    return pos, neg


def drop_single_chars(words):
    for w in words:
        if len(w) == 1:
            words.remove(w)

    return words


import unidecode
def remove_accents(accented_string):
    unaccented_string = unidecode.unidecode(accented_string)
    return unaccented_string


def tweet_preprocessing(df, stemming, lemmatization):

    # STOPWORDS
    stopwords_greek = set(stopwords.words('greek'))

    print(stopwords_greek)
    sw_list = []
    for sw in stopwords_greek:
        sw_clear = remove_intonation(sw)
        sw_list.append(sw_clear)

    print(set(sw_list))



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

    positive_list, negative_list = get_pos_neg_lists(stemmer)
    print(positive_list)
    print(negative_list)
    pos_score_list = []
    neg_score_list = []
    for i in range(0, len(df)):

        tweet = str(df.iloc[i]['text'])
        # print(tweet)

        # remove accentuation
        tweet = remove_intonation(tweet)

        # tokenization
        tokens = tweet.split()
        tokens = drop_single_chars(tokens)

        # remove very small or very large tweets
        # if len(tokens) > 36 or len(tokens) < 3:
        #     df.loc[i, 'label'] = 10

        # p_score, n_score = count_pos_neg_score(tokens, positive_list, negative_list)

        # remove stopwords
        if stemming is True:
            words = [stemmer.stem(w.upper()) for w in tokens if w not in stopwords_greek]
            tweet_clean = ' '.join([w for w in words if len(w) > 1])
            tweet_clean = tweet_clean.lower()
            p_score, n_score = count_pos_neg_score(words, positive_list, negative_list)
            # print(p_score)
            # print(n_score)

        else:
            words = [w for w in tokens if w not in stopwords_greek]
            # if 'ξεφτιλες' in words or 'τσιρκο' in words:
            tweet_clean = ' '.join([w for w in words])
            # p_score, n_score = count_pos_neg_score(words, positive_list, negative_list)
            # print(p_score)
            # print(n_score)

            if lemmatization is True:
                # lemmatization
                tweet_lemmas = spacy_lemmatizer(tweet_clean)
                lemmas = [t.lemma_ for t in tweet_lemmas]
                tweet_clean = ' '.join([l for l in lemmas])

        # print(tweet_clean)
        if len(tweet_clean)>=1:
            list_of_tweets.append(tweet_clean)
        else:
            list_of_tweets.append(np.nan)

        pos_score_list.append(p_score)
        neg_score_list.append(n_score)

        # df.loc[i]['text'] = tweet_clean

    df['tweet'] = list_of_tweets
    df['#pos'] = pos_score_list
    df['#neg'] = neg_score_list
    return df


if __name__ == '__main__':

    dfs = pd.read_excel('finalTwitter_v2.xlsx',  engine='openpyxl')

    # keep only Tweet + Sentiment columns
    df = dfs.iloc[:, [1, 2]]
    df.columns = ['text', 'label']

    # drop unlabeled rows
    df = df[df['label'].notna()]

    print(len(df))

    # drop tweets irrelevant to Covid19
    df = df[df['label'] < 10]

    print(len(df))

    labels = df['label'].tolist()

    df = first_step(df)
    df['label'] = labels

    print(df)
    df.columns = ['text', 'label']

    df = tweet_preprocessing(df, True, False)

    df = df.drop(['text'], axis=1)
    #
    # # change column order
    cols = df.columns.tolist()
    cols = ['tweet', 'label', '#pos', '#neg']
    df = df[cols]

    # # change data type of Label
    df['label'] = df['label'].astype(int)

    print(df.head())


    # df['tweet'].replace('', np.nan, inplace=True)

    df.dropna(inplace=True)

    # write to csv
    # df.to_csv('tweets_stemming_drop_hashtag_content.csv', index=False)



