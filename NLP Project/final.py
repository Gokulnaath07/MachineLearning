import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv('Tweets.csv')
data.head(7)

shape=data.shape
print(shape)

cols=data.columns.to_list()
print(cols)

data.isnull().sum()

"""Dropping columns(text and airlinesentiment) row"""

df=data.dropna(subset=['text', 'airline_sentiment'])

df.head()

drop_the_cols=["tweet_location", "user_timezone",
                "negativereason", "negativereason_confidence"]

df = df.drop(columns=drop_the_cols, errors="ignore")

df.head()

df.head()

df.isnull().sum()

col=["airline_sentiment_gold","negativereason_gold","tweet_coord"]
df=df.drop(columns=col, errors='ignore')

df.head(50)

df.isnull().sum()

"""Tweet Length Distributions + Label Proportions"""

df["char_length"]=df["text"].astype(str).apply(len)

plt.figure(figsize=(10,8))
sea.histplot(df["char_length"], bins=50, kde=True)
plt.xlabel("Character Length")
plt.ylabel("Frequency")
plt.title("Tweet Character Length Distribution")
plt.show()

df["Word_character"]=df["text"].astype(str).apply(lambda x: len(x.split()))

plt.figure(figsize=(10,8))
sea.histplot(df["Word_character"], bins=40, kde=True)
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.title("Tweet Word Count Distribution")
plt.show()

plt.figure(figsize=(9,6))
sea.boxplot(x="airline_sentiment", y="Word_character", data=df)
plt.title("Tweet Word Length by Sentiment Class")
plt.xlabel("Sentiment")
plt.ylabel("Word Count")
plt.show()

plt.figure(figsize=(10,8))
df["airline_sentiment"].value_counts().plot(kind="bar")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.title("Label proportion")
plt.xticks(rotation=45)
plt.show()

from wordcloud import WordCloud, STOPWORDS

sentiments=df["airline_sentiment"].unique()

for i, sentiment in enumerate(sentiments, 1):
  text=" ".join(df[df["airline_sentiment"]==sentiment]['text'].astype(str))

  wordcloud=WordCloud(width=800,
                      height=600,
                      stopwords=STOPWORDS,
                      background_color="white").generate(text)
  plt.subplot(2,2,i)
  plt.imshow(wordcloud, interpolation="bilinear")
  plt.axis("off")
  plt.title("Wordcloud for every sentiment class")
  plt.show()

from collections import Counter
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words=set(stopwords.words('english'))

def preprocessing_wrd(text):
  text=text.lower()
  text=re.sub(r"http\S+|www\S+|https\S+", "", text)
  text=re.sub(r"@\w+|#\w+", "", text)
  text=re.sub(r"[^a-zA-Z\s]", "", text)

  tokens=text.split()
  tokens=[t for t in tokens if t not in stop_words and len(t)>2]
  return tokens

def words_apply(df, sentiment, n=15):
  all_words=df[df["airline_sentiment"]==sentiment]['text'].astype(str).apply(preprocessing_wrd)
  flatwords=[word for sublist in all_words for word in sublist]

  word_count=Counter(flatwords).most_common(n)
  word, count=zip(*word_count)

  plt.figure(figsize=(10,8))
  sea.barplot(x=list(count), y=list(word),orient="h")
  plt.title(f"Top {n} frequent tweet words")
  plt.xlabel("Count")
  plt.ylabel("Words")
  plt.show()

for sentiment in sentiments:
  words_apply(df, sentiment)

from itertools import chain

df['cleaned_text'] = df['text'].apply(preprocessing_wrd)

# all_words = list(chain.from_iterable(df['cleaned_text']))
all_words=[word for wordlist in df["cleaned_text"] for word in wordlist]
word_counts = Counter(all_words).most_common(20)

words, counts = zip(*word_counts)

plt.figure(figsize=(10, 5))
sea.barplot(x=list(counts), y=list(words), orient='h')
plt.title("Overall Top 20 Most Common Words")
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.show()

from nltk.stem import PorterStemmer

stop_words=set(stopwords.words('english'))
stemmer=PorterStemmer()

def preprocee_words(text):
  text=text.lower()
  text=re.sub(r"[^a-zA-Z\s]", "", text)
  tokens=text.split()
  tokens=[stemmer.stem(word) for word in tokens if word not in stop_words]
  return " ".join(tokens)

df['clean_text']=df['text'].astype(str).apply(preprocee_words)
print(df['clean_text'])

from sklearn.feature_extraction.text import CountVectorizer

Bag_of_words=CountVectorizer(max_features=5000)
x_BoW=Bag_of_words.fit_transform(df['clean_text'])
print(x_BoW)

from sklearn.feature_extraction.text import TfidfVectorizer

tdidf=TfidfVectorizer(max_features=5000)
x_tdidf=tdidf.fit_transform(df['clean_text'])
print(x_tdidf)

