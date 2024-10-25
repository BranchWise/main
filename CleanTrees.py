import pandas as pd

dataset_1 = pd.read_csv('SpeciesList_2024-10-24-7-57-5-PM.csv')
dataset_2 = pd.read_csv('TreeGOER_2023.csv', delimiter='|')

dataset_1['species'] = dataset_1['Genus Name'].str.strip() + " " + dataset_1['Species Name'].str.strip()

columns_to_drop_from_dataset_1 = ['Synonyms', 'Family', 'Order', 'Class', 'Species Code', 
                                  'Growth Form', 'Percent Leaf Type', 'Leaf Type', 
                                  'Growth Rate', 'Longevity', 'Height at Maturity (feet)']

dataset_1_cleaned = dataset_1.drop(columns=columns_to_drop_from_dataset_1)

dataset_2_pivot = dataset_2.pivot(index='species', columns='var', values='MEAN')

dataset_2_pivot.reset_index(inplace=True)

columns_to_drop_from_dataset_2 = ['LAT', 'LON', 'MCWD', 'PETColdestQuarter', 'PETDriestQuarter', 
                                  'PETWarmestQuarter', 'PETWettestQuarter', 'PETseasonality', 
                                  'annualPET', 'aridityIndexThornthwaite', 'bdod', 'bio07', 'bio08', 
                                  'bio09', 'bio10', 'bio11', 'bio13', 'bio14', 'bio15', 'bio18', 
                                  'bio19', 'cec', 'clay', 'climaticMoistureIndex', 'continentality', 
                                  'elev', 'growingDegDays0', 'growingDegDays5', 'maxTempColdest', 
                                  'meanTempColdest', 'meanTempWarmest', 'minTempWarmest', 
                                  'monthCountByTemp10', 'nitrogen', 'phh2o', 'sand', 'silt', 'soc', 
                                  'thermicityIndex', 'topoWet', 'tri']

dataset_2_pivot_cleaned = dataset_2_pivot.drop(columns=columns_to_drop_from_dataset_2, errors='ignore')

final_merged_data = pd.merge(dataset_1_cleaned, dataset_2_pivot_cleaned, on='species', how='left')

final_merged_data = final_merged_data.fillna("NULL")

print(final_merged_data.head())

final_merged_data.to_csv('cleaned_tree_data.csv', index=False)