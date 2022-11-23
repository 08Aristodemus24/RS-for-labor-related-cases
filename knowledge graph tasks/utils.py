import pandas as pd


def peek(data_1: pd.DataFrame) -> None:
    for content in data_1:
        print("||{}||".format(content), '\n')