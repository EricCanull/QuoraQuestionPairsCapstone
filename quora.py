# %% [markdown]
# # Quora Question Pairs
# - Capstone Project: Eric Canull

# %%
# Import libraries
import datetime
import logging
import os
import warnings
from time import time

import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fuzzywuzzy import fuzz
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
from scipy.spatial.distance import (
    braycurtis, canberra, chebyshev, cityblock, cosine, euclidean, jaccard,
    minkowski, sqeuclidean)
from scipy.stats import kurtosis, skew
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (
    KFold, cross_val_predict, cross_val_score, train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from tqdm import tqdm_notebook

import xgboost as xgb
from graphs import heatmap_chart, pie_chart, histogram
from textcleaner import clean_text, drop_null, print_questions

warnings.filterwarnings("ignore")
start_nb = time()

# %%
# Initialize logging.
logging.basicConfig(filename='logs/quora.log',
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

# %%
# Global Variables
TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"
GOOGLE_EMBEDDING_FILE = "data/googlenews/GoogleNews-vectors-negative300.bin.gz"
GLOVE_EMBEDDING_FILE = 'data/glove.6B/glove.6B.300d.txt'
GLOVE_WORD2VEC_FILE = 'data/glove.6B/word2vec/glove.6B.300d.word2vec.txt'
GLOVE_WORD2VEC_MODEL = "models/glove_300d_word2vec.model"
GLOVE_WORD2VEC_NORM_MODEL = "models/glove_300d_norm_word2vec.model"
Q1_WMD_TRAINING_DATA_FILE = "data/npy/q1_wmd_train.npy"
Q2_WMD_TRAINING_DATA_FILE = "data/npy/q2_wmd_train.npy"
EMBEDDING_DIM = 300

stops = set(stopwords.words("english"))

# %%
# Import files
pd.options.mode.chained_assignment = None
train_df = pd.read_csv(filepath_or_buffer=TRAIN_FILE)
test_df = pd.read_csv(filepath_or_buffer=TEST_FILE)

# %%
# Loading into a Gensim Word2Vec model class may take several minutes.
if not os.path.exists(GLOVE_EMBEDDING_FILE):
    raise FileNotFoundError(
        "Warning! You need to download the Glove embeddings")

if not os.path.exists(GLOVE_WORD2VEC_MODEL):
    glove2word2vec(GLOVE_EMBEDDING_FILE, GLOVE_WORD2VEC_FILE)
    word2vec = KeyedVectors.load_word2vec_format(GLOVE_WORD2VEC_FILE, binary=False)
else:
    word2vec = gensim.models.KeyedVectors.load(GLOVE_WORD2VEC_MODEL)

#%% Save word embedding models   
# word2vec.save('models/glove_300d_word2vec.model')
# norm_word2vec.save('models/glove_300d_norm_word2vec.model')

# %% [markdown]
# ## L<sup>2</sup> unit-normalized WMD
# ### Necessary for computing cosine similiarity

# %%
# Loading into a Gensim Word2Vec model class may take several minutes.
if not os.path.exists(GLOVE_WORD2VEC_NORM_MODEL):
    norm_word2vec = word2vec
    norm_word2vec.init_sims(replace=True)
else:
    norm_word2vec = gensim.models.KeyedVectors.load(GLOVE_WORD2VEC_NORM_MODEL)

# %%
# Null counts in training set
# train_df.isnull().sum()
train_df[train_df.isnull().any(axis=1)]

# %%
# Null counts in test set
# test_df.isnull().sum()
test_df[test_df.isnull().any(axis=1)]

# %%
# Drop rows with null values
train_df = drop_null(train_df)
test_df = drop_null(test_df)

# %%
# Training dataset summary statistics
train_df.describe()

# %%
# Perform text cleaning
train_df["question1"] = train_df['question1'].apply(clean_text)
train_df['question2'] = train_df['question2'].apply(clean_text)
print_questions(train_df)

# %% [markdown]
# ## Transform questions into vectors

# %% Functions to compute WMD and normalized WMD
def wmd(q1, q2):
    q1 = str(q1).lower().split()
    q2 = str(q2).lower().split()
    q1 = [w for w in q1 if w not in stops and w.isalpha()]
    q2 = [w for w in q2 if w not in stops and w.isalpha()]
    return word2vec.wmdistance(q1, q2)


def norm_wmd(q1, q2):
    q1 = str(q1).lower().split()
    q2 = str(q2).lower().split()
    q1 = [w for w in q1 if w not in stops and w.isalpha()]
    q2 = [w for w in q2 if w not in stops and w.isalpha()]
    return norm_word2vec.wmdistance(q1, q2)


def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if w not in stops]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(norm_word2vec[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())

# %%
# transform questions into vectors
question1_vectors = np.zeros((train_df.shape[0], EMBEDDING_DIM))
for i, q in enumerate(tqdm_notebook(train_df.question1.values)):
    question1_vectors[i, :] = sent2vec(q)

# %%
# print q1 embedded matrix
question1_vectors

# %%
# transform questions into vectors
question2_vectors = np.zeros((train_df.shape[0], EMBEDDING_DIM))
for i, q in enumerate(tqdm_notebook(train_df.question2.values)):
    question2_vectors[i, :] = sent2vec(q)

# %%
# print q2 embedded matrix
question2_vectors

# %%
# Persist data sets
np.save(open(Q1_WMD_TRAINING_DATA_FILE, 'wb'), question1_vectors)
np.save(open(Q2_WMD_TRAINING_DATA_FILE, 'wb'), question2_vectors)

# %%
print("WMD question1 data shape: ", question1_vectors.shape)
print("WMD question2 data shape: ", question2_vectors.shape)


#%%
#Function to calculate normalized word share between two questions
def word_share_norm(x):
    w1 = set(map(lambda word: word.lower().strip(), str(x['question1']).split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), str(x['question2']).split(" ")))    
    return 1.0 * len(w1 & w2)/(len(w1) + len(w2))

#%%
# Combine all questions into corpus for analysis similar to Term-frequency in TFIDF
train_questions = pd.Series(train_df['question1'].tolist() + train_df['question2'].tolist()).astype(str)
test_questions = pd.Series(test_df['question1'].tolist() + test_df['question2'].tolist()).astype(str)


# Implement TFIDF function
def get_weight(count, eps=5000, min_count=2):
    if count < min_count:
        return 0 #remove words only appearing once 
    else:
        R = 1.0 / (count + eps)
        return R

eps = 5000 
words = (" ".join(train_questions)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}
print("Most common words: ", (sorted(weights.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10]))


#Calculate TFIDF word match share as our new feature
def tfidf_word_share_norm(x):
    w1 = set(map(lambda word: word.lower().strip(), str(x['question1']).split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), str(x['question2']).split(" "))) 
    if len(w1)==0 or len(w2)==0:
        return 0
    common = w1 & w2
    share_weight = [weights.get(word, 0) for word in common]
    total_weight = [weights.get(word, 0) for word in w1]+[weights.get(word, 0) for word in w2]
    return np.sum(share_weight)/np.sum(total_weight)

# %% [markdown]
# ## Feature Engineering
# ### - 29 Total Features 
# * The length of word.
# * The length of character.
# * The length of common word between question1 and question2.
# * The length difference between question1 and question2.
# * Word share and TDIDF Word share norm
# * Fuzz ratios
# * The word movers distance
# * The normalized word movers distanc
# * Cosine distance
# * City block (Manhattan) distance
# * Jaccard distance
# * Canberra distance
# * Chebyshev_distance
# * Euclidean distance
# * Minkowski distance
# * Bray-Curtis distance
# * Skewness and kurtosis

# ### Distances using scipy.spatial.distance libraries

# %%
# Engineering Features
train_df['len_q1'] = train_df.question1.apply(lambda x: len(str(x)))
train_df['len_q2'] = train_df.question2.apply(lambda x: len(str(x)))
train_df['diff_len'] = train_df.len_q1 - train_df.len_q2
train_df['len_char_q1'] = train_df.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
train_df['len_char_q2'] = train_df.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
train_df['len_word_q1'] = train_df.question1.apply(lambda x: len(str(x).split()))
train_df['len_word_q2'] = train_df.question2.apply(lambda x: len(str(x).split()))
train_df['word_share'] = train_df.apply(word_share_norm, axis=1)
train_df['TFIDF_share'] = train_df.apply(tfidf_word_share_norm, axis=1, raw=True)
train_df['common_words'] = train_df.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
train_df['fuzz_ratio'] = train_df.apply(lambda x: fuzz.ratio(str(x['question1']), str(x['question2'])), axis=1)
train_df['fuzz_partial_ratio'] = train_df.apply( lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
train_df['fuzz_partial_token_set_ratio'] = train_df.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
train_df['fuzz_partial_token_sort_ratio'] = train_df.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
train_df['fuzz_token_set_ratio'] = train_df.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
train_df['fuzz_token_sort_ratio'] = train_df.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
train_df['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
train_df['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
train_df['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
train_df['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
train_df['chebyshev_distance'] = [chebyshev(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
train_df['sqeuclidean_distance'] = [sqeuclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
train_df['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
train_df['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
train_df['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
train_df['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
train_df['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
train_df['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
train_df['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]


#%%
histogram("Training Question Lengths", train_df['len_q1'], train_df['len_q2'])
histogram("Training Questions Word Lenghts", train_df['len_char_q1'], train_df['len_char_q2'])
histogram("Training Questions Char Lenghts", train_df['len_word_q1'], train_df['len_word_q2'])
histogram("Training Questions Skew", train_df['skew_q1vec'], train_df['skew_q2vec'])

# %% [markdown]
# ## Word2vect Modeling
# ### WMD is a measure of distance. The similarities in WmdSimilarity are simply the negative distance. Don't confuse distances and similarities.
# - Two similar sentences will have a high similarity score and a small distance.
# - Two very different sentences will have low similarity score, and a large distance.

# %%
# Drop first three columns since they won't be used
train_df.drop(['id', 'qid1', 'qid2'], axis=1, inplace=True)

# %%
# Create a Word2Vec wmd distance column in train dataframe
train_df['wmd'] = train_df.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)

# %%
train_df.head(2)

# %% [markdown]
# ## Normalized Word2vec Modeling

# %%
# Create a normalized Word2Vec wmd distance column in train dataframe
train_df['norm_wmd'] = train_df.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)

# %%
# Replaced inf values with NaN then drop
# train_df = train_df.replace(
#     [np.inf, -np.inf], np.nan).dropna(subset=["wmd", "norm_wmd"], how="all")
train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["norm_wmd"], how="all")

# %%
# Count NaN values in dataframe
train_df.isna().sum()

# %%
# Drop non-feature columns for spliting data
train_df.drop(['question1', 'question2'], axis=1, inplace=True)
# train_df.drop(['Unnamed: 0'], axis=1, inplace=True)
train_df = train_df[pd.notnull(train_df['cosine_distance'])]
train_df = train_df[pd.notnull(train_df['braycurtis_distance'])]
# train_df = train_df[pd.notnull(train_df['jaccard_distance'])]

# %%
train_df.isnull().sum()

# %%
# Preview data
train_df.head(5)

# %%
# Distance Features
# 'len_q1'                          'len_q2'
# 'diff_len'                        'len_char_q1'
# 'len_char_q2'                     'len_word_q1'
# 'len_word_q2'                     'common_words'
# 'word_share"                      'TFIDF_share'
# 'fuzz_ratio'                      'fuzz_partial_ratio'
# 'fuzz_partial_token_set_ratio'    'fuzz_partial_token_sort_ratio'
# 'fuzz_token_set_ratio'            'fuzz_token_sort_ratio'
# 'cosine_distance'                 'cityblock_distance'
# 'jaccard_distance'                'canberra_distance'
# 'chebyshev_distance'              'sqeuclidean_distance'
# 'euclidean_distance'              'minkowski_distance'
# 'braycurtis_distance'             'skew_q1vec'
# 'skew_q2vec'                      'kur_q1vec'
# 'kur_q2vec'                       'wmd'         
# 'norm_wmd

# Create list with all feature all feature name headers except "is_duplicate"
features_list = train_df.loc[:, train_df.columns !=
                             'is_duplicate'].columns.tolist()

# Print list of features in train_df
for a, b in enumerate(features_list):
    print("[{}] {}".format(a, b))

# %%
# Helper to find index specific columns to drop features from train_df
# print(features_list[15:27])
train_df[features_list[0:9]].head(2)

# %%
# Show heatmap features charts
heatmap_chart(train_df, features_list[1:30])  # All features
heatmap_chart(train_df, features_list[0:9])  # String Features
heatmap_chart(train_df, features_list[10:15])  # Fuzzy Ratio Features
heatmap_chart(train_df, features_list[16:30])  # Distance Features

# %%
# Split data
X = train_df.loc[:, train_df.columns != 'is_duplicate']
y = train_df.loc[:, train_df.columns == 'is_duplicate']
train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.3, random_state=123)

# %%
# Check types
train_x.dtypes

#%% [markdown]
# ## Random Forest

# %%
# Random Forest
# k=10, split the data into 10 equal parts
print("Starting training at", datetime.datetime.now())
t0 = time()

kfold = KFold(n_splits=10, random_state=1)
model = RandomForestClassifier(criterion='gini', n_estimators=700,
                               min_samples_split=10, min_samples_leaf=1,
                               max_features='auto', oob_score=True,
                               random_state=1, n_jobs=-1)
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
print('Accuracy', accuracy_score(test_y, pred_y))
print(classification_report(test_y, pred_y))

t1 = time()
# evaluate the prediction results
print("Training ended at", datetime.datetime.now())
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

#%%
result_rm = cross_val_score(
    model, train_x, train_y, cv=10, scoring='accuracy')
print('The cross validated score for Random Forest Classifier is:',
      round(result_rm.mean()*100, 2))
y_pred = cross_val_predict(model, train_x, train_y, cv=10)
sns.heatmap(confusion_matrix(
    pred_y, y_pred), annot=True, fmt='3.0f', cmap="summer")
plt.title('Confusion_matrix', y=1.05, size=15)

#%% [markdown]
# ## Decision Tree

# %%
print("Starting training at", datetime.datetime.now())
t0 = time()

clf = tree.DecisionTreeClassifier(criterion="entropy")
# train model
clf = clf.fit(train_x, train_y.values.ravel())
# make prediction
pred_y = clf.predict(test_x)

t1 = time()
# evaluate the prediction results
print("Training ended at", datetime.datetime.now())
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))
print('Accuracy', accuracy_score(test_y, pred_y))
for line in classification_report(test_y, pred_y).split("\n"):
    print(line)

#%% [markdown]
# ## LinearSVC

# %%
print("Starting training at", datetime.datetime.now())
t0 = time()

clf = LinearSVC(random_state=123456)
# train model
clf = clf.fit(train_x, train_y.values.ravel())
# make prediction
pred_y = clf.predict(test_x)

t1 = time()
# evaluate the prediction results
print("Training ended at", datetime.datetime.now())
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))
print('Accuracy', accuracy_score(test_y, pred_y))
for line in classification_report(test_y, pred_y).split("\n"):
    print(line)

#%% [markdown]
# ## MultinomialNB

# %%
print("Starting training at", datetime.datetime.now())
t0 = time()

clf = MultinomialNB()
# train model
clf = clf.fit(train_x, train_y.values.ravel())
# make prediction
pred_y = clf.predict(test_x)

t1 = time()
# evaluate the prediction results
print("Training ended at", datetime.datetime.now())
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))
print('Accuracy', accuracy_score(test_y, pred_y))
for line in classification_report(test_y, pred_y).split("\n"):
    print(line)

# %% [markdown]
# ## Xgboost

# %%
# Create XGBModel
print("Starting training at", datetime.datetime.now())
t0 = time()

xgbmodel = xgb.XGBClassifier(max_depth=50,
                             n_estimators=80,
                             learning_rate=0.1,
                             colsample_bytree=0.7,
                             gamma=0,
                             reg_alpha=4,
                             objective='binary:logistic',
                             eta=0.3, silent=1,
                             subsample=0.8).fit(train_x,
                                                      train_y.values.ravel())

pred_y = xgbmodel.predict(test_x)
cm = confusion_matrix(test_y, pred_y)

t1 = time()
print("Training ended at", datetime.datetime.now())
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

print(cm)
print('Accuracy', accuracy_score(test_y, pred_y))
print(classification_report(test_y, pred_y))

#%%
#Normalized feature values
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_df)
test_data_scaled = scaler.fit_transform(test_df)
#%%
#Split training data to train and validation data sets
label = train_data_raw['is_duplicate']
X_train, X_valid, y_train, y_valid = train_test_split(train_data, label, test_size=0.2, random_state=10)
X_train_scaled, X_valid_scaled, y_train, y_valid = train_test_split(train_data_scaled, label, test_size=0.2, random_state=10)

#%%
dtrain = xgb.DMatrix(train_x)
dtest = xgb.DMatrix(test_x)
watchlist = [(dtest, 'eval'), (dtrain, 'train')]

# advanced: start from a initial base prediction

print ('start running example to start from a initial prediction')
# specify parameters via map, definition are same as c++ version
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}

# train xgboost for 1 round
bst = xgb.train(param, dtrain, 1, watchlist)
# Note: we need the margin value instead of transformed prediction in set_base_margin
# do predict with output_margin=True, will always give you margin values before logistic transformation
ptrain = bst.predict(dtrain, output_margin=True)
ptest = bst.predict(dtest, output_margin=True)
dtrain.set_base_margin(ptrain)
dtest.set_base_margin(ptest)


print('this is result of running from initial prediction')
bst = xgb.train(param, dtrain, 1, watchlist)

#%%
# Graph decision tree
#Visualize the prediction
from matplotlib.pylab import rcParams
from xgboost import plot_tree
rcParams['figure.figsize'] = 120,100
plot_tree(bst)


# %%
# Show feature importance levels
fig, ax = plt.subplots()
ts = pd.Series(xgbmodel.get_booster().get_fscore())
ts.sort_values(axis=0, ascending=True)
ax = ts.sort_values().plot(kind="barh", title="Features importance",
                           figsize=(12, 10), use_index=True)
ax.yaxis.label.set_color('#a2a2f7')
plt.show()

# %%
# Must install graphviz before using
# Opens plot tree in graphviz
dot = xgb.to_graphviz(xgbmodel, num_trees=4, rankdir="LR")
dot.render("images/xgplot_tree")

# %% [markdown]
# # Utility Functions
# ### CAUTION: The commands below could remove existing dataframe

# %%
# EXPORT: saved csv data
# train_df.to_csv('data/train_simple_feat.csv')
train_df.to_csv('data/train_all_feat.csv')
# train_df.to_csv('data/train_vect_feat_drop_feat.csv')
# train_df.to_csv('data/train_wmd_dropinf.csv')

# %%
# IMPORT: saved csv data
# train_df = pd.read_csv('data/train_vect_feat.csv')
# train_df = pd.read_csv('data/train_simple_feat.csv')
train_df = pd.read_csv('data/csv/train_all_feat.csv')
# train_df = train_df[pd.notna(train_df['wmd'])]

# %%
train_df.head()

# %%
#train_df.drop(['norm_wmd'], axis=1, inplace=True)

# %% [markdown]
# # Testing Functions
# %%
q1 = "What is Amazon\'s organisational structure?	What was it like to work for Amazon in the 90\'s?"
q2 = "What is postpositives.com?	What is Devbitrack.com?"
q3 = 'What would a Trump presidency mean for current international master’s students on an F1 visa?'
q4 = 'How will a Trump presidency affect the students presently in US or planning to study in US?'

# %%
# test = norm_wmd(q1, q2)
test = wmd(q3, q4)
print(test)

# %%
test = norm_wmd(q3, q4)

print(test)
