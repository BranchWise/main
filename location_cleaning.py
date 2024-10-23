import pandas as pd
import numpy as np

def load_data():
    # Load the data
    edmonton_neighbourhoods = pd.read_csv('data/edmonton_neighborhoods.csv')
    edmonton_vaccant_land = pd.read_csv('data/edmonton_vaccant_land.csv')
    edmonton_pests = pd.read_csv('data/edmonton_pests.csv')
    # unzip and read the tree data
    # edmonton_trees = pd.read_csv('data/edmonton_trees.csv.zip')
    # return edmonton_neighbourhoods, edmonton_vaccant_land, edmonton_pests, edmonton_trees


def main():
    load_data()
    print("Data loaded successfully")

if __name__ == '__main__':
    main()
