import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_excel(path)
    #encoding = 'sys.getfilesystemcoding()'
    return df

train_path = "../data/Train.xlsx"
test_path = "../data/Test.xlsx"

if __name__ == "__main__":
    # test your module here
    # train_path = "data/Train.xlsx"
    # test_path = "data/Test.xlsx"
    df = load_data(train_path)
    print(df.info())