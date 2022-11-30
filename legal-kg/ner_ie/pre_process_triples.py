import numpy as np 
import pandas as pd 
import os


# these preprocess functions do similar things but have different implementations depending on what the user wants - Michael

#============================================================================================

#                   PREPROCESSING WITH CARDINAL AND BLANK TOKENS PRESERVED

#============================================================================================

# it seeems like this is the next part after stanza_ner.py after the STANZA_TRIPLES directory 
# and its output .csv files are create. This pre_process_triples.py file calls these 3 functions
# by the order preprocess_basic() which loops through each .csv file in the created STANZA_TRIPLES folder
# and then in each .csv file its token columns are all essentially transformed into lower case strings
# and then saved to another to be created directory called PRE_PROCESSED_TRIPLES/WITH_CARDINAL/
# the path for each file would now then be PRE_PROCESSED_TRIPLES/WITH_CARDINAL/<some name>.csv
# - Michael
def preprocess_basic():
    # each file is a DataFrame with structure:
    #   tid    label    token
    # 0  <some text id>    <val>    <val>
    # .
    # .  ...    ...    ...
    # .
    # n - 1  <some text id>    <val>    <val>
    # - Michael
    files = os.listdir("STANZA_TRIPLES/")
    print("Number of files are: ",len(files))
    master_list = []
    count = []
    for i in range(len(files)):
        print("================  DOCUMENT ",(i+1),"  ================")
        print("----- DOCUMENT ID:: ",files[i])
        data = pd.read_csv("STANZA_TRIPLES/"+files[i])
        data['token'] = data['token'].str.lower()
        print("Number of rows before: ",len(data))
        data = data.drop_duplicates(inplace=False)
        print("Number of rows after: ",len(data))
        count.append(len(data))
        master_list.append(data)
        # each DataFrame has its token column reduced to lowercase strings
        # and duplicate rows in DataFrame is removed each DataFrame is 
        # then exported to another DataFrame with structure:
        #   tid    label    token
        # 0  <some text id>    <val>    <val>
        # .
        # .  ...    ...    ...
        # .
        # n - 1  <some text id>    <val>    <val>
        # - Michael
        data.to_csv("PRE_PROCESSED_TRIPLES/WITH_CARDINAL/"+files[i],index=False)


    print("Number of documents : ",len(master_list))
    merged = pd.concat(master_list)
    print("The master dataframe has number of rows = ",len(merged))
    print("Average number of entities = ",sum(count)/len(count))

    print("-------------------------------------------------------------------------------------")

#============================================================================================

#                   PREPROCESSING TO REMOVE CARDINAL AND BLANK TOKENS

#============================================================================================

# this function accesses the files of newly created directory PRE_PROCESSED_TRIPLES/WITH_CARDINAL/ by 
# prepocess_basic() in the end it creates another directory with new files called 
# PRE_PROCESSED_TRIPLES/WITHOUT_CARDINAL/ and the path of the files would be 
# PRE_PROCESSED_TRIPLES/WITHOUT_CARDINAL/<some name>.csv - Michael
def preprocess_remove_cardinal():
    # each file is a DataFrame with structure:
    #   tid    label    token
    # 0  <some text id>    <val>    <val>
    # .
    # .  ...    ...    ...
    # .
    # n - 1  <some text id>    <val>    <val>
    # - Michael
    files = os.listdir("PRE_PROCESSED_TRIPLES/WITH_CARDINAL/")
    print("Number of files are: ",len(files))
    master_list = []
    count = []
    for i in range(len(files)):
        if files[i]==".DS_Store":
            # once a file with name .DS_Store is found do not continue in the next lines
            # but skip through this iteration onto the next one - Michael
            continue
        print("================  DOCUMENT ",(i+1),"  ================")
        print("----- DOCUMENT ID:: ",files[i])
        data = pd.read_csv("PRE_PROCESSED_TRIPLES/WITH_CARDINAL/"+files[i])
        print("Number of rows before: ",len(data))

        # check the label column of a DataFrame if it has CARDINAL. data.label != 'CARDINAL'
        # returns a list with all indeces either equating to True if its not a 'CARDINAL'
        # and false if it is indeed 'CARDINAL'. E.g.
        #   label
        # 0  CARDINAL
        # .
        # .
        # .
        # n - 1 <string otherwise CARDINAL> != 'CARDINAL'
        # 
        # results in [False, True, True, ..., True] which can be used to index the DataFrame again
        # and return all rows with its indeces the same as the indeces that hold a true value in the
        # previous resulting boolean list. E.g. df[[False, True, True, False, ..., True]] indeces of boolean list
        # which are True are 1, 2, ..., n - 1 therefore get only the rows of the DataFrame with these indeces
        # - Michael
        data = data[data.label != 'CARDINAL']

        print("Number of rows after removing cardinal labels: ",len(data))
        count.append(len(data))
        master_list.append(data)
        data.to_csv("PRE_PROCESSED_TRIPLES/WITHOUT_CARDINAL/"+files[i],index=False)
    print("Number of documents : ",len(master_list))
    merged = pd.concat(master_list)
    print("The master dataframe has number of rows = ",len(merged))
    print("Average number of entities = ",sum(count)/len(count))

#============================================================================================

#============================================================================================

# this function access the files of newly created directory PRE_PROCESSED_TRIPLES/WITHOUT_CARDINAL/ by
# preprocess_remove_cardinal() in the end it creates another directory with new files called
# PRE_PROCESSED_TRIPLES/ and the path of the files would have a path of PRE_PROCESSED_TRIPLES/labels.csv
# - Michael
def preprocess_identify_labels():
    
    df = pd.DataFrame(columns=['tid','PERSON','NOP','FAC','ORG','GPE','LOC','PRODUCT','EVENT','WORK_OF_ART','LAW','LANGUAGE','DATE','TIME','PERCENT','MONEY','QUANTITY','ORDINAL','CARDINAL'])
    
    # # each file is a DataFrame with structure:
    #   tid    label    token
    # 0  <some text id>    <string otherwise CARDINAL>    <val>
    # .
    # .  ...    ...    ...
    # .
    # n - 1  <some text id>    <string otherwise CARDINAL>    <val>
    # - Michael
    files = os.listdir("PRE_PROCESSED_TRIPLES/WITHOUT_CARDINAL/")
    print("Number of files are: ",len(files))
    print(df.columns)
    for i in range(len(files)):
        print("================  DOCUMENT ",(i+1),"  ================")
        print("----- DOCUMENT ID:: ",files[i])
        data = pd.read_csv("PRE_PROCESSED_TRIPLES/WITHOUT_CARDINAL/"+files[i])
        labels = []

        # data['label'].unique() or data.label.unique() accesses the label column
        # and uses the self.unique() method which returns all unique values in the
        # DataFrame's column as a numpy array then this its method self.tolist()
        # just returns this numpy array as a list 
        # ['<uniquelabel1>', '<unqiuelabel2>', '<uniquelabel3>', ..., '<uniquelabeln>']
        # - Michael
        labels = data.label.unique().tolist()

        print(labels)
        print(len(labels))
        print(type(labels))
        s = len(df)
        df.loc[s,'tid'] = (files[i])[:-4]

        # the labels list is actually the columns of the initially created DataFrame above with
        # columns 'PERSON','NOP','FAC','ORG','GPE','LOC','PRODUCT','EVENT','WORK_OF_ART','LAW',
        # 'LANGUAGE','DATE','TIME','PERCENT','MONEY','QUANTITY','ORDINAL','CARDINAL'. - Michael
        for j in range(len(labels)):
            df.loc[s,labels[j]]=1
    
    # because the labels column doesn't have anymore the CARDINAL string which has been
    # removed previously by the preprocess_remove_cardinal and then extracted out of the
    # DataFrame using self.unique and looped through and used as access labels[j] 
    # in df.loc[s, labels[j]] and set to 1 the CARDINAL column of the DataFrame is left out
    # and is therefore not populated which becomes Nan when the rest of its columns are populated with
    # 1 we then identify all the Nan values in the DataFrame and replace it with 0 instead
    # - Michael
    df = df.replace(np.nan, 0)

    for i in df.columns:
        # access each column in the DataFrame and assign a new column with all its values 
        # converted to int types - Michael
        df[i]=df[i].astype(int)
    #df = df.replace(1.0,1)
    print(df)

    # output csv has structure:
    #   tid    PERSON    NOP    FAC    ORG    GPE    LOC    PRODUCT    EVENT    WORK_OF_ART    LAW    LAGNUAGE    DATE    TIME    PERCENT    MONEY    QUANTITY    ORDINAL    CARDINAL
    # 0  1       1        1      1      1      1      1        1         1           1          1         1        1        1        1         1         1           1          0
    # .
    # .
    # .
    # n - 1  1    1       1      1      1      1      1        1         1           1          1         1        1        1        1         1         1           1          0
    df.to_csv("PRE_PROCESSED_TRIPLES/labels.csv",index=False)

# this preprocess function is used over the two latter preprocess functions - Michael
preprocess_identify_labels()


