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
    return edmonton_neighbourhoods, edmonton_vaccant_land, edmonton_pests

def clean_and_merge_data():
    # create dataframes
    neighbourhoods_df, vaccant_land_df, pests_df = load_data()
    
    # make sure the neighbourhoods col is consistent
    neighbourhoods_df.rename(columns={'Neighbourhood Name':'neighbourhood'}, inplace=True)
    vaccant_land_df.rename(columns={'NEIGHBOURHOOD_NAME':'neighbourhood'}, inplace=True)
    pests_df.rename(columns={'Neighbourhood':'neighbourhood'}, inplace=True)

    # convert the neighbourhoods to lowercase
    neighbourhoods_df['neighbourhood'] = neighbourhoods_df['neighbourhood'].str.lower().str.strip()
    vaccant_land_df['neighbourhood'] = vaccant_land_df['neighbourhood'].str.lower().str.strip()
    pests_df['neighbourhood'] = pests_df['neighbourhood'].str.lower().str.strip()

    # merge the dataframes
    merged_df = pd.merge(neighbourhoods_df, vaccant_land_df, on='neighbourhood', how='outer')
    merged_df = pd.merge(merged_df, pests_df, on='neighbourhood',how='outer')

    # handle missing values
    merged_df.fillna(0, inplace=True)

    # keep only the columns we need (neighbourhood, Geometry Multipolygon, OWNERSHIP_TYPE, ZONING, TYPOLOGY, Pest)
    keep_cols = ['neighbourhood', 'Geometry Multipolygon', 'OWNERSHIP_TYPE', 'ZONING', 'TYPOLOGY', 'Pest']
    merged_df = merged_df[keep_cols]

    # drop duplicate rows
    merged_df.drop_duplicates(inplace=True)

    return merged_df


def main():
    load_data()
    print("Data loaded successfully")
    final_df = clean_and_merge_data()
    print("Data cleaned and merged successfully")
    print(final_df)

    with open('location_final_data.txt', 'w') as f:
        f.write(final_df.to_string())

if __name__ == '__main__':
    main()
