import pandas as pd
from preprocessing import remove_intonation
from greek_stemmer import GreekStemmer


def stemming(word, stemmer):
    word = stemmer.stem(word.upper())
    return word.lower()


def get_pos_neg_lists(stemmer):

    lexicon = pd.read_csv("lexicon/posneg_lexicon.csv")
    # stemmer = GreekStemmer()

    print(lexicon)
    for i in range(0, len(lexicon)):
        word = lexicon.iloc[i]['word']
        word = remove_intonation(word)
        word = stemmer.stem(word.upper()).lower()
        lexicon.iat[i, 0] = word


    # lexicon = lexicon.drop_duplicates()
    positiveDf = lexicon[lexicon['sentiment'] == 'positive']
    negativeDf = lexicon[lexicon['sentiment'] == 'negative']

    positive_words = positiveDf['word'].tolist()
    negative_words = negativeDf['word'].tolist()

    return set(positive_words), set(negative_words)


