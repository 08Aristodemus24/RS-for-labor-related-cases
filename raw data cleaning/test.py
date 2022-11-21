import pandas as pd
import spacy
import neuralcoref
import nltk

df = pd.read_csv('./Major Philippine Labor Law Resources.csv', index_col=0)

# this is for jurisprudence
# who was doing who, date where it was committed, name of the person, 
ENTITY_TYPES = ['article', 'section', 'chapter', 'person', 'company', 'enterprise', 'business', 'geographic region', 'geographic entity', 'organization']



