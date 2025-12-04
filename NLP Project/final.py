import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
import warnings
warnings.filterwarnings('ignore')

# !pip install tensorflow

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

Bag_of_words=CountVectorizer(max_features=20000)
x_BoW=Bag_of_words.fit_transform(df['clean_text'])
print(x_BoW)

from sklearn.feature_extraction.text import TfidfVectorizer

tdidf=TfidfVectorizer(max_features=20000)
x_tdidf=tdidf.fit_transform(df['clean_text'])
print(x_tdidf)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

y=df['airline_sentiment']

x_train, x_test, y_train, y_test=train_test_split(x_tdidf, y, test_size=0.2, random_state=27, stratify=y)

"""Logistic regression"""

logistic_reg=LogisticRegression(max_iter=500, class_weight='balanced')
logistic_reg.fit(x_train, y_train)

y_pred_lr=logistic_reg.predict(x_test)
print("\tLOGISTIC REGRESSION")
clasa_rep=classification_report(y_test, y_pred_lr)
acc_lr=accuracy_score(y_test, y_pred_lr)
print("Confusion Matrix:\n", clasa_rep)
print("Accuracy Score:", acc_lr)

conf_mat_lr=confusion_matrix(y_test, y_pred_lr)
sea.heatmap(conf_mat_lr, cmap='Blues', annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# from sklearn.model_selection import GridSearchCV

# parameters={
#     'criterion':['gini', 'entropy'],
#     'max_depth':[20,40,60,80, 100],
#     'min_samples_split':[2,5,10,15,20],
#     'min_samples_leaf':[1,2,5,10],
#     'max_features':['auto', 'sqrt', 'log2']
# }

# grid=GridSearchCV(
#     estimator=DecisionTreeClassifier(),
#     param_grid=parameters,
#     cv=5,
#     n_jobs=-1,
#     scoring='f1_weighted'
# )

# grid.fit(x_train, y_train)

# print(grid.best_params_)
# print(grid.best_score_)

des_tree=DecisionTreeClassifier(criterion='entropy',
                                max_depth=100,
                                max_features='sqrt',
                                min_samples_leaf=2,
                                min_samples_split=15
                                )
des_tree.fit(x_train, y_train)

y_pred_dt=des_tree.predict(x_test)
print("\tDECISION TREE")
acc_dt=accuracy_score(y_test, y_pred_dt)
clas_rep_dt=classification_report(y_test, y_pred_dt)
print("Confusion Matrix:\n", clas_rep_dt)
print("Accuracy Score:", acc_dt)

conf_mat_dt=confusion_matrix(y_test, y_pred_dt)
sea.heatmap(conf_mat_dt, cmap='Blues',annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Decision Tree Confusion Matrix")
plt.show()

"""LSTM"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print(tf.__version__)

tokenizer=Tokenizer(num_words=20000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['text'])

X_Seq = tokenizer.texts_to_sequences(df['text'])
X_pad=pad_sequences(X_Seq, maxlen=50, padding='post')

y=pd.factorize(df['airline_sentiment'])[0]

print(y)

x_train, x_test, y_train, y_test=train_test_split(X_pad, y, test_size=0.2, random_state=27, stratify=y)

model=Sequential()
model.add(Embedding(input_dim=20000, output_dim=128, input_length=50))
model.add(LSTM(128))
model.add(Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history=model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=8,
    batch_size=32,

)

y_pred_Lstm=np.argmax(model.predict(x_test), axis=1)

print("\tLSTM")
acc_lstm=accuracy_score(y_test, y_pred_Lstm)
clas_rep_lstm=classification_report(y_test, y_pred_Lstm)
print("Confusion Matrix:\n", clas_rep_lstm)
print("Accuracy Score:", acc_lstm)

con_lstm=confusion_matrix(y_test, y_pred_Lstm)
print("\tConfusion matrix LSTM")
sea.heatmap(con_lstm, cmap='Blues', annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("LSTM Confusion Matrix")
plt.show()

# !pip install transformers torch

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

x=df['text'].astype(str)
y=df['airline_sentiment']

label2id={label: i for i, label in enumerate(y.unique())}
id2label={value: key for key, value in label2id.items()}

print(label2id)

print(id2label)

y=y.map(label2id)

print(y)

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=27, stratify=y)

tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'index': idx
        }
x_test_list = list(x_test)
y_test_list = list(y_test)

train_dataset = TweetDataset(x_train, y_train, tokenizer)
test_dataset  = TweetDataset(x_test, y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3
).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss = {round(total_loss/len(train_loader),4)}")

model.eval()
y_pred, y_true, indexes = [], [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(batch['labels'].numpy())
        indexes.extend(batch['index'].numpy())

acc_bert=accuracy_score(y_true, y_pred)
print("Accuracy:", acc_bert)

print(classification_report(y_true, y_pred, target_names=label2id.keys()))

cm_bert = confusion_matrix(y_true, y_pred)
sea.heatmap(cm_bert, annot=True, cmap='Blues')
plt.title("BERT (PyTorch) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

acc=acc_bert
match acc:
  case acc if acc_lstm>acc:
    print("The best accuracy achieved is from LSTM: ", acc_lstm)
  case acc if acc_dt> acc_bert:
    print("The best accuracy achieved is from Decision Tree: ", acc_dt)
  case acc if acc_lr>acc:
      print("The best accuracy achieved is from Logistic Regression: ", acc_lr)
  case _:
    print("The best accuracy achieved is from BERT: ", acc_bert)

"""According to all these weighted f1 values we can say the BERT has the best F1 which is 0.84 also the accuracy is better with the BERT is the best performing model for this sentiment classification task. While the simpler ML models like Decision tree and Logistic regression are good and faster, these black box models LSTM and BERT performed better with their explainability.

Per class insights:
1. Negative tweets: The bert model and lstm can caputre the negative tweets better than the simpler models which might be wrong in some cases.
2. Neutral: All models performed better but the BERT still has some edge over others here
3. Positive tweets: With high f1 and recall the BERT scored here as well.

This explains that the trade off between interpretablity and performance. Black box models can be slower but their explainability can be better at performance.
"""

indexes=np.array(indexes)
y_true=np.array(y_true)
y_pred=np.array(y_pred)

order = np.argsort(indexes)

y_true_ordered=y_true[order]
y_pred_ordered=y_pred[order]

error_df=pd.DataFrame({
    'x_test_texts': [x_test_list[i] for i in order],
    'y_test': y_true_ordered,
    'y_pred_dt': y_pred_ordered,
})

misplacced_tweets=error_df[error_df['y_test']!=error_df['y_pred_dt']]
misplaced_tweets=misplacced_tweets.sample(15, random_state=27)

misplaced_tweets