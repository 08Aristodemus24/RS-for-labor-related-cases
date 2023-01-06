import pandas as pd
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


# from lexnlp.extract.common.base_path import lexnlp_test_path
# from lexnlp.extract.en.dict_entities import prepare_alias_banlist_dict, AliasBanRecord, DictionaryEntry, DictionaryEntryAlias

# def load_entities_dict():
#     base_path = os.path.join(lexnlp_test_path, 'lexnlp/extract/en/tests/test_geoentities')
#     entities_fn = os.path.join(base_path, 'geoentities.csv')
#     aliases_fn = os.path.join(base_path, 'geoaliases.csv')
#     return DictionaryEntry.load_entities_from_files(entities_fn, aliases_fn)


# _CONFIG = list(load_entities_dict())





pint_labor_conventions = pd.read_csv('../raw data cleaning/Proposed International Labor Conventions.csv', index_col=0)
# print(pint_labor_conventions['page_text_content'][0])

callbacks = [
    get_act_list,
    get_amounts,
    get_citations,
    get_companies,
    get_constraints,
    get_copyright,

    # get_courts,

    # get_dates,
    # get_definitions,
    # get_distances,
    # get_durations,

    # get_geoentities,

    # get_percents,
    # get_regulations,
    # get_trademarks
]

for callback in callbacks:
    entities = callback(pint_labor_conventions['page_text_content'][0])
    print(*entities, sep='\n')
    

