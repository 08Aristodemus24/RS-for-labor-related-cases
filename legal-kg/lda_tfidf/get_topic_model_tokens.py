import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')
import os
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
stemmer = PorterStemmer() #or SnowballStemmer(language='english')
import pandas as pd
import numpy as np
from gensim import corpora, models
import re

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

# the DataFrame read here is created by the TOPIC_MODEL_LDA.ipynb file which 
# has the structure
# text,index,tid
#  Competition Commission of India Shri Satyendra Singh vs Ghaziabad Development Authority   on 28 February 2018 ,0,6406437
# COMPETITION COMMISSION OF INDIA ,1,6406437
# Case No ,2,6406437
# .
# .
# .
# Would it be reasonable to balance OBC reservation with societal interests by instituting OBC cut-off marks that are slightly lower than that of the general category? It is reasonable to balance reservation with other societal interests ,5775,63489929
# To maintain standards of excellence cut off marks for OBCs should be set not more than 10 marks out of 100 below that of the general category See paras 274-278 These Writ Petitions and Contempt Petition are accordingly disposed of ,5776,63489929
# In the facts and circumstances the parties are to bear their own costs J (Dalveer Bhandari) New Delhi; April 10 2008 ,5777,63489929
df=pd.read_csv("FOR_TF_IDF.csv")
df = df.reset_index(drop=True)
print(df.head())
print("====== PREPROCESS START....")
processed_docs = df['text'].map(preprocess)
print("====== PREPROCESS ENDS....")

print("===== Dictionary stuff....")
dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=3, no_above=0.65, keep_n=1000000)
print(dictionary)
print("===== Stuff ends....")

print("==== BOW work....")
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

print("===== Loading the LDA model ...")
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)

print("====== STARTING ITERATIONS!!!!!!")
master = pd.read_csv("FOR_TF_IDF.csv")
master=master.reset_index(drop=True)
for i in range(len(master)):
    print("------ LINE ",i+1,' OF ',len(master)," ------")
    unseen_document = master.loc[i,'text']
    bow_vector = dictionary.doc2bow(preprocess(unseen_document))
    x=""
    for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
        print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
        x = lda_model.print_topic(index, 10)
        x = re.sub(r'\*',' ',x)
        x = re.sub(r'[0-9]','',x)
        x = re.sub(r'\+',' ',x)
        x = re.sub(r'\.',' ',x)
        x = re.sub(r'"',' ',x)
        x = re.sub(r' +',' ',x)
        x = re.sub(r' ',',',x)
        break
    master.loc[i,'tokens'] = x
    if i%5000 == 0:
        master.to_csv("TM_TOKENS.csv",index=False)
    #print(x)

# this get_topic_model_tokens.py file then outputs the TM_TOKENS.csv file
# which will be used by work_on_topic_models.py file
master.to_csv("TM_TOKENS.csv",index=False)
