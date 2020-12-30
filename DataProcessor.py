import pandas as pd
import nltk
import re
import wordsegment as ws
from nltk.tokenize import RegexpTokenizer
ws.load()
nltk.download('stopwords')
def stopwords(words):
    stopwords = set(nltk.corpus.stopwords.words('greek'))
    filtered_text = [word for word in words if word not in stopwords]
    return filtered_text


csv = pd.read_csv("dataset_v2.csv",engine='python')
text_array = []
for index, row in csv.iterrows():
    text = row[10].lower()
    text = nltk.re.sub(r"http\S+", "", text)
    text = text.replace("#"," ")
    text = text.replace("_"," ")
    text = nltk.re.sub(r"@\S+", "", text)
    text = nltk.re.sub(r"[a-zA-Z0-9]", "", text)
    tokenizer = RegexpTokenizer(r'\w+')
    text = ' '.join(tokenizer.tokenize(text))
    words = stopwords(nltk.word_tokenize(text))
    text_array.append(' '.join(words))
df = pd.DataFrame({"Text":text_array})
df.to_csv('preprocessingDataset.csv', mode='w', index=False, header=True,encoding="utf-8")