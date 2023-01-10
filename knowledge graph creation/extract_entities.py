import os

from lexnlp.extract.en.acts import get_act_list
from lexnlp.extract.en.amounts import get_amounts
from lexnlp.extract.en.citations import get_citations
from lexnlp.extract.en.entities.nltk_re import get_companies
from lexnlp.extract.en.constraints import get_constraints
from lexnlp.extract.en.copyright import get_copyright
from lexnlp.extract.en.courts import get_courts
from lexnlp.extract.en.dates import get_dates
from lexnlp.extract.en.definitions import get_definitions
from lexnlp.extract.en.distances import get_distances
from lexnlp.extract.en.durations import get_durations
from lexnlp.extract.en.geoentities import get_geoentities
# from lexnlp.extract.en.money import get_money
# this only include $, GBP, rupee, yen, yuan,
# and not pesos so not that useful
from lexnlp.extract.en.percents import get_percents
from lexnlp.extract.en.regulations import get_regulations
from lexnlp.extract.en.trademarks import get_trademarks

import nltk
import pandas as pd

# lexnlp version is still in 1.8.0 so upgrade to latest
# this is why it gives an attirbute error when accessing the
# self.load_entities_from_files() method of Dictionary Entry

# from lexnlp.extract.common.base_path import lexnlp_test_path
# from lexnlp.extract.en.dict_entities import prepare_alias_banlist_dict, AliasBanRecord, DictionaryEntry, DictionaryEntryAlias

# def load_entities_dict():
#     base_path = os.path.join(lexnlp_test_path, 'lexnlp/extract/en/tests/test_geoentities')
#     entities_fn = os.path.join(base_path, 'geoentities.csv')
#     aliases_fn = os.path.join(base_path, 'geoaliases.csv')
#     return DictionaryEntry.load_entities_from_files(entities_fn, aliases_fn)


# _CONFIG = list(load_entities_dict())

nltk.download('averaged_perceptron_tagger')

pint_labor_conventions = pd.read_csv('../raw data cleaning/Proposed International Labor Conventions.csv', index_col=0)
corpus = pint_labor_conventions['page_text_content'][0]

callbacks = [
    get_act_list,
    get_amounts,
    get_citations,
    get_companies,
    get_constraints,
    get_copyright,

    # get_courts,

    get_dates,
    get_definitions,
    get_distances,
    get_durations,

    get_geoentities,

    get_percents,
    get_regulations,
    get_trademarks
]

for callback in callbacks:
    # prints the function name
    print(callback.__name__)

    # use each function by passing corpus or body of text
    entities = callback(corpus, geo_config_list=_CONFIG) if callback.__name__ == "get_geoentities" else callback(corpus)

    # unzip the generator object
    print(*entities, sep='\n', end='\n')
    
    

