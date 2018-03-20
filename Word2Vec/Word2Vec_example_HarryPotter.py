import pandas as pd
import re
from nltk import ngrams
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
from gensim.models import Word2Vec
import logging

def transform_row(row):
    # Xoa so o dau dong
    row = re.sub(r"^[0-9\.]+", "", row)
    
    # Xoa dau cham cau o cuoi cau
    row = re.sub(r"[\.,\?]+$", "", row)
    
    # Xoa dau cham cau trong cau
    row = row.replace(",", "").replace(".", "") \
        .replace(";", "").replace("“", "") \
        .replace(":", "").replace("”", "") \
        .replace('"', "").replace("'", "") \
        .replace("!", "").replace("?", "")
    
    row = row.strip()
    return row 

def kieu_ngram(string, n=1):
    gram_str = list(ngrams(string.split(), n))
    return [ " ".join(gram).lower() for gram in gram_str ]

def Tokenizer(data):
	tknzr = Tokenizer(lower=True, split=" ")
	return tknzr.fit_on_texts(data)

def Visualization(model):
	words_np = []
	words_label = []
	for word in model.wv.vocab.keys():
	    words_np.append(model.wv[word])
	    words_label.append(word)

	pca = PCA(n_components=2)
	pca.fit(words_np)
	reduced = pca.transform(words_np)

	plt.rcParams["figure.figsize"] = (20,20)
	for index,vec in enumerate(reduced):
	    if index <200:
	        x,y=vec[0],vec[1]
	        plt.scatter(x,y)
	        plt.annotate(words_label[index],xy=(x,y))
	plt.show()

# Load data 
df = pd.read_csv("./HarryPotter.txt",sep="/", names=["row"]).dropna()

# Remove punctuation
df["row"] = df.row.apply(transform_row)

# Split word using ngram
df["1gram"] = df.row.apply(lambda t: kieu_ngram(t, 1))
df["2gram"] = df.row.apply(lambda t: kieu_ngram(t, 2))

# Combine data and word2vec with Gensim
df["context"] = df["1gram"] + df["2gram"]
#df["context"] = df.row.apply(lambda t: Tokenizer(t))
train_data = df.context.tolist()

# Training gensim model
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)    
model = Word2Vec(train_data, size=100, window=10, min_count=3, workers=4, sg=1)

# Truy van
result = model.wv.similar_by_word("ron")
print(result)

# Visualize
#Visualization(model)