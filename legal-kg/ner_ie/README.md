# INFORMATION EXTRACTION
## The following directory implements NER annotators for Information Extraction.<br>
1. Install the required dependencies using requirements.txt<br>
2. In order to record the STANZA NER results, create a directory "STANZA_TRIPLES/" in the (print working directory which is the path of the working directory or the directory the user is currently in that he is working in to maybe somehow do some things-Michael). <br>
3. The file 'stanza_ner.py' stores the annotation results of the text files in the "STANZA_TRIPLES/" directory.<br>
4. The results are then preprocessed using the 'pre_process_triples.py' file. <br>
5. In order to record the results, create 2 sub-directories: 'WITH_CARDINAL/' and 'WITHOUT_CARDINAL/' in the 'PRE_PROCESSED_TRIPLES/' directory.

<!-- Michael -->
this directory contains the text_from_html_1152.csv which I speculate may have been:
- created by a file living in the whole legal-kg directory
- a dataset from the beginning to be used in the creation of the knowledge graph

pre_process_triples.py
- preprocess_identify_labels() is the only function ran

compare_annotators.ipynb
- 

stanza_ner.py (stanza named entity recognition)
- this file used the stanza python library