import pandas as pd
import numpy as np
import os

# list all the files of this directory which contains a number of
# .csv files with name <number>_TF_IDF_SCORES.csv
# this is the structure of the dataframe
# tid,sentence,word,score
# 6406437, Competition Commission of India Shri Satyendra Singh vs Ghaziabad Development Authority   on 28 February 2018 ,state,0.2286191876936881
# 6406437, Competition Commission of India Shri Satyendra Singh vs Ghaziabad Development Authority   on 28 February 2018 ,india,0.3245053101805335
# 6406437, Competition Commission of India Shri Satyendra Singh vs Ghaziabad Development Authority   on 28 February 2018 ,countri,0.3245053101805335
# .
# .
# .
# 6406437, Competition Commission of India Shri Satyendra Singh vs Ghaziabad Development Authority   on 28 February 2018 ,hous,0.3245053101805335
# 6406437, Competition Commission of India Shri Satyendra Singh vs Ghaziabad Development Authority   on 28 February 2018 ,electr,0.3245053101805335
# 6406437, Competition Commission of India Shri Satyendra Singh vs Ghaziabad Development Authority   on 28 February 2018 ,chief,0.3245053101805335
files = os.listdir("TFIDF_SCORES/")
merged = pd.DataFrame(columns=['tid','sentence','word','score'])
for i in range(len(files)):
    print("======== DATAFRAME ",i+1," OF ",len(files)," ==========")
    df = pd.read_csv("TFIDF_SCORES/"+files[i])
    df = df.drop_duplicates(subset=['tid','word'])
    merged = merged.append(df)

# once duplicate rows based on columns tid and word are dropped
# merge into a single result.csv file with structure
# tid,sentence,word,score
# 112850760,The questions are of great constitutional significance affecting the principle of independence of the judiciary which is a basic feature of the Constitution and we would therefore prefer to being the discussion by making a few prefatory remarks highlighting what the true function of the judiciary should be in a country like India which is marching along the road to social justice with the banner of democracy and the rule of law for the principle of independence of the judiciary is not an abstract conception but it is a living faith which must derive its inspiration from the constitutional charter and its nourishment and sustenance from the constitutional values ,public,0.2927085674952852
# 112850760,The questions are of great constitutional significance affecting the principle of independence of the judiciary which is a basic feature of the Constitution and we would therefore prefer to being the discussion by making a few prefatory remarks highlighting what the true function of the judiciary should be in a country like India which is marching along the road to social justice with the banner of democracy and the rule of law for the principle of independence of the judiciary is not an abstract conception but it is a living faith which must derive its inspiration from the constitutional charter and its nourishment and sustenance from the constitutional values ,court,0.246276644375576
# 112850760,The questions are of great constitutional significance affecting the principle of independence of the judiciary which is a basic feature of the Constitution and we would therefore prefer to being the discussion by making a few prefatory remarks highlighting what the true function of the judiciary should be in a country like India which is marching along the road to social justice with the banner of democracy and the rule of law for the principle of independence of the judiciary is not an abstract conception but it is a living faith which must derive its inspiration from the constitutional charter and its nourishment and sustenance from the constitutional values ,power,0.2979380404974127
# .
# .
# .
# 8141
# 27391014,The DG also reported that ACI used its dominance in the upstream relevant market to enhance its presence in the downstream relevant market amounting to a violation of section 4(2)(e) of the Act 6 5 Based on the analysis of the provisions of License Agreement and facts gathered during the investigation the DG found that the restrictions imposed by ACI do not satisfy the 'reasonable restrictions test' required to balance the conflicting interest in a business relationship ,judici,0.3373404332488141
# 27391014,The DG also reported that ACI used its dominance in the upstream relevant market to enhance its presence in the downstream relevant market amounting to a violation of section 4(2)(e) of the Act 6 5 Based on the analysis of the provisions of License Agreement and facts gathered during the investigation the DG found that the restrictions imposed by ACI do not satisfy the 'reasonable restrictions test' required to balance the conflicting interest in a business relationship ,reason,0.3373404332488141
# 27391014,The DG also reported that ACI used its dominance in the upstream relevant market to enhance its presence in the downstream relevant market amounting to a violation of section 4(2)(e) of the Act 6 5 Based on the analysis of the provisions of License Agreement and facts gathered during the investigation the DG found that the restrictions imposed by ACI do not satisfy the 'reasonable restrictions test' required to balance the conflicting interest in a business relationship ,mean,0.3373404332488141
# 
# this is the file to be processed by 
merged.to_csv("MERGED_TFIDF/result.csv",index=False)

