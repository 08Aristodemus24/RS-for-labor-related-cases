import re
import os
import pandas as pd
import stanza

#================================================================================================
#================================================================================================

# tokens is a list, and labels is a list - Michael
def update_labels(tokens,labels,model_name):
  # declare a DataFrame with 0 rows with columns 'PIPELINE/MODEL', 'TOKEN', 'LABEL' - Michael
  R = pd.DataFrame(columns=['PIPELINE/MODEL','TOKEN','LABEL'])
  print("=====================================================")
  print("=====================================================")
  print("=====================================================")
  print("              FOR THE MODEL ",model_name,"           ")
  print("")
  print(" Number of tokens = ",len(tokens))
  print(" Number of labels = ",len(labels))
  print("=====================================================")
  print("=====================================================")
  print("=====================================================")


  for i in range(len(labels)):
    # do not add a row to the DataFrame if labels[i] in the labels list
    # contains an 'O' character - Michael
    if labels[i]=='O':
      continue
    else:
      print(tokens[i]," : ",labels[i])

      # initially length of DataFrame R will be 0 since it will be an empty one
      # but as loop goes through labels and tokens this DataFrame will be populated
      # - Michael
      X=len(R)
      #print("Current Length: ",X)

      # e.g. R.loc[0, 'PIPELINE/MODEL'] = model_name, will mean select the 0th row and the
      # 'PIPELINE/MODEL' column and set it to the model_name which is 'STANZA-ONTONOTES'
      # this goes the same for R.loc[0, 'TOKEN'] and R.loc[0, 'LABEL'] - Michael
      R.loc[X,'PIPELINE/MODEL']=model_name
      R.loc[X,'TOKEN']=tokens[i]
      R.loc[X,'LABEL']=labels[i]
      #print("New Lengrh: ",len(R))
  #print("TABLE after model ",model_name,": ")
  #print(R)
  return R

  '''--------------------------------------------------------------------------------------'''

def stanza_nlp(text):
  print("======================   STANZA   =========================")

  # stanza is like other NLP libraries like nltk, and spacy - Michael
  stanza.download('en')
  nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
  doc = nlp(text)

  # t stands for token and l stands for label - Michael
  t=[]
  l=[]
  #print(*[f'entity: {ent.text}\ttype: {ent.type}' for sent in doc.sentences for ent in sent.ents], sep='\n')
  for sent in doc.sentences:
    for ent in sent.ents:
      t.append(ent.text)
      l.append(ent.type)
  #print(t)
  #print(l)

  # returns a DataFrame with structure:
  #   PIPELINE/MODEL  TOKEN  LABEL
  # 0 'STANZA-ONTONOTES'        val    val
  # .
  # .       ...        ...    ...
  # .
  # n - 1   'STANZA-ONTONOTES'        val    val
  # 
  # - Michael
  return update_labels(t,l,'STANZA-ONTONOTES')

#================================================================================================
#================================================================================================

def create_triples_format(tid,df):
  f = pd.DataFrame(columns = ['tid','label','token'])
  for i in range(len(df)):
    # length of DataFrame will initially be 0 since df is empty
    # and because it starts from 0 and then populates the rows at this index
    # and then the next row at an index of a DataFrame that has now been populated
    # at least once this somehow mimics just an appending function where in each
    # iteration a row is added to the DataFrame object - Michael
    s = len(f)
    f.loc[s,'tid'] = tid
    f.loc[s,'label'] = df.loc[i,'LABEL']
    f.loc[s,'token'] = df.loc[i,'TOKEN']

  # returns a DataFrame with structure
  #   tid    label    token
  # 0  <some text id>    <val>    <val>
  # .
  # .  ...    ...    ...
  # .
  # n - 1  <some text id>    <val>    <val>
  return f

def generate_entities(text):
  r = stanza_nlp(text)

  # returns a DataFrame with structure:
  #   PIPELINE/MODEL  TOKEN  LABEL
  # 0 'STANZA-ONTONOTES'        val    val
  # .
  # .       ...        ...    ...
  # .
  # n - 1   'STANZA-ONTONOTES'        val    val
  # 
  # - Michael
  return r

# go up in directory and access all the files of the directory data/LEGAL_TEXT which I checked may be either the jurisprudence or the corpus juris (body of law) - Michael
files = os.listdir("../data/LEGAL_TEXT/")
#os.mkdir("STANZA_TRIPLES/")

print("TOTAL FILES = ",len(files))

# loop goes through all the legal text files - Michael
for i in range(len(files)):
  print("========== FILE #",i+1," ===========")

  # get only the string from -len(string) to -4 since array indeces we know in 
  # python are arranged 0, 1, 2, ..., n - 1 or -n, -n + 1, ..., -2, -1 
  # why this is is because every string which represents the file is in written like 
  # <text id>.txt and so removing the last 4 characters which is .txt is what this line
  # ['100074926.txt', '100123.txt', ..., '999236.txt']
  # does overall the variable essentially means title id or maybe text id 
  # e.g. -10 -9 -8 -7 -6 -5 -4 -3 -2 -1 -----> -10 -9 -8 -7 -6 -5
  # -5 -4 -3 -2 -1 -----> -5
  # -4 -3 -2 -1 -----> -4
  # ['100074926', '100123', ..., '999236']
  # - Michael
  tid = (files[i])[:-4]

  # because we only get the files with names more than 4 characters we open only
  # the files with 5 or more characters slice until the -4 index and try to open it
  # - Michael
  f = open("../data/LEGAL_TEXT/"+tid+".txt")
  text = f.read()
  name = tid+".csv"
  r1 = generate_entities(text)
  r2 = create_triples_format(tid,r1)

  # creates a DataFrame with structure
  #   tid    label    token
  # 0  <some text id>    <val>    <val>
  # .
  # .  ...    ...    ...
  # .
  # n - 1  <some text id>    <val>    <val>
  r2.to_csv("STANZA_TRIPLES/"+tid+".csv",index=False)