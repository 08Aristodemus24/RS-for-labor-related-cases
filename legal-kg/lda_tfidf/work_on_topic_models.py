from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import re

# this .csv file has been created by get_topic_model_tokens.py
# which has the structure:
# text,index,tid,tokens
# Competition Commission of India Shri Satyendra Singh vs Ghaziabad Development Authority   on 28 February 2018 ,0,6406437,",state,india,countri,backward,union,gener,minist,hous,electr,chief,"
# COMPETITION COMMISSION OF INDIA ,1,6406437,",state,india,countri,backward,union,gener,minist,hous,electr,chief,"
# Case No ,2,6406437,",case,fact,examin,record,evid,accus,polic,offic,offenc,statement,"
# .
# .
# .
# Would it be reasonable to balance OBC reservation with societal interests by instituting OBC cut-off marks that are slightly lower than that of the general category? It is reasonable to balance reservation with other societal interests ,5775,63489929,",right,power,exercis,court,public,person,principl,judici,reason,mean,"
# To maintain standards of excellence cut off marks for OBCs should be set not more than 10 marks out of 100 below that of the general category See paras 274-278 These Writ Petitions and Contempt Petition are accordingly disposed of ,5776,63489929,",court,order,high,appeal,case,learn,respond,judgment,decis,appel,"
# In the facts and circumstances the parties are to bear their own costs J (Dalveer Bhandari) New Delhi; April 10 2008 ,5777,63489929,",right,power,exercis,court,public,person,principl,judici,reason,mean,"
# - Michael
data = pd.read_csv("TM_TOKENS.csv")

print(" ========== modifying output")
'''for i in range(len(data)):
    print("sentence ",i+1,'of ',len(data))
    data.loc[i,'tokens'] = re.sub(r',',' ',data.loc[i,'tokens'])
    data.loc[i,'tokens'] = re.sub(r' +',' ',data.loc[i,'tokens'])
data.to_csv("MODIFIED_TM_TOKENS.csv",index=False)'''

print(data.head())

print("producing TF-IDF results .......")
v = TfidfVectorizer()

# the tokens column of the dataframe are then passed to a term frequency-ivnerse document frequency
# algorithm to create and train a model out of this data - Michaels
x = v.fit_transform(data['tokens'])
print("======== RESULTS GENERATED !")
feature_names = v.get_feature_names()

result = pd.DataFrame(columns=['tid','sentence','word','score'])
SIZE=0
for i in range(len(data)):
    print("======== SENTENCE ",i+1," =========")
    feature_index = x[i,:].nonzero()[1]
    tfidf_scores = zip(feature_index, [x[i, y] for y in feature_index])
    for w, s in [(feature_names[j], s) for (j, s) in tfidf_scores]:
        result.loc[SIZE,'tid'] = data.loc[i,'tid']
        result.loc[SIZE,'sentence'] = data.loc[i,'text']
        result.loc[SIZE,'word'] = w
        result.loc[SIZE,'score'] = s
        SIZE+=1
    if i%1000 == 0:
        result.to_csv("TFIDF_SCORES/"+str(i)+"_TF_IDF_SCORES.csv",index=False)
        SIZE = 0
        # resets the dataframe to blank
        # - Michael
        result = pd.DataFrame(columns=['tid','sentence','word','score'])
        print("FILE SAVED!!")

# this whole loop populates the result DataFrame and every iteration divisible by 1000
# the file is named <i>_TF_IDF_SCORES.csv e.g. 0_TF_IDF_SCORES.csv, 1000_TF_IDF_SCORES.csv
# and once loop is done this last line basically creates the last .csv file - Michael
result.to_csv("TFIDF_SCORES/last_TF_IDF_SCORES.csv",index=False)

# all in all work_on_topic_models.py creates the <num>_TF_IDF_SCORES.csv files to be processed by 
# MERGE_TFIDF_RESULTS.py - Michael