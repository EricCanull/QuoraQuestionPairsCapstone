from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords 


def clean_text(text):
    # Pre process and convert texts to a list of words
    text = str(text).lower().strip()
    text = text.lower()

    # Clean the text
    text = re.sub("[^A-Za-z0-9^,!./'+-=]", " ", text)
    text = re.sub("√", "square root", text)
    text = re.sub("≈", "almost equal to", text)
    text = re.sub("≠", "not equal to", text)
    text = re.sub("و", "and", text)
    text = re.sub("∫","integral", text)
    text = re.sub("β", "beta", text)
    text = re.sub("σ", "standard deviation", text)
    text = re.sub("≤", "less than or equals to", text)
    text = re.sub("し", "in hiragana", text)
    text = re.sub("シ", "in katakana", text)
    text = re.sub("π", "pi", text)
    text = re.sub("ω", "omega", text)
    text = re.sub(r" (the[\s]+|The[\s]+)?US(A)? ", " America ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r" india ", " India ", text)
    text = re.sub(r" switzerland ", " Switzerland ", text)
    text = re.sub(r" china ", " China ", text)
    text = re.sub(r" chinese ", " Chinese ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r" programing ", " programming ", text)
    text = re.sub(r" bestfriend ", " best friend ", text)
    
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r" III ", " 3 ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"-", " - ", text)
    text = re.sub(r"=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # text = text.split()

    return text

def preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
        .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
        .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
        .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
        .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    
    
    porter = PorterStemmer()
    pattern = re.compile(r'\W')
    
    if type(x) == type(''):
        x = re.sub(pattern, ' ', x)
    
    
    if type(x) == type(''):
        x = porter.stem(x)
        example1 = BeautifulSoup(x)
        x = example1.get_text()
               
    return x

# Show how many question columns have null values
def print_null_counts(df1, df2):
    df1 = ['Train dataframe:', str(df1.isnull().sum())]
    df2 = ['Test dataframe:', str(df2.isnull().sum())]
    print("{:s}\n{:s}\n\n{:s}\n{:s}".format(
        df1[0], df1[1], df2[0], df2[1]))

# Prints 10 questions at positions (0, 10, 20, etc...)
def print_questions(df):
    a = 0
    for i in range(a, a+10):
        print(df.question1[i])
        print(df.question2[i])
        print()



# Drop rows in dataframe where a cell contains a null value
def drop_null(df):
   return df.dropna(how='any').reset_index(drop=True)
