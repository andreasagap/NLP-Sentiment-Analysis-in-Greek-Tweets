import pandas as pd
import nltk
import re
import wordsegment as ws
ws.load()
def removeAndSegmentHTAGS(text,to_leave):
    text=text.split()
    for i,part in enumerate(text):
        if part in to_leave:
            text[i]=""
            continue
        if part.startswith('#'):
            part=(re.sub("(#)","",part)) # removing Hashtags but keeping the words
            part =" ".join(ws.segment(part))
            text[i]=part
    text= (" ".join(text))
    return text


csv = pd.read_csv("dataset_v2.csv",engine='python')
text_array = []
for index, row in csv.iterrows():
    text = row[10].lower()
    text = nltk.re.sub(r"http\S+", "", text)
    text = text.replace("@","")
    text = removeAndSegmentHTAGS(text, [])
    #text = nltk.re.sub(r"covid\S+", "", text)
    text = nltk.re.sub(r"[a-zA-Z0-9]", "", text)
    text_array.append(text)
df = pd.DataFrame({"Text":text_array})
df.to_csv('preprocessingDataset.csv', mode='w', index=False, header=True,encoding="utf-8")