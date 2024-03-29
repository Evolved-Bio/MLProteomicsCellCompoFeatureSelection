#Step 1: Uploading Files

from google.colab import files
import io
import pandas as pd
import numpy as np

#Load individual proteomics datasets as csv files
uploaded = files.upload()
df1 = pd.read_csv(io.BytesIO(uploaded['PR_1.csv']))
df2 = pd.read_csv(io.BytesIO(uploaded['PR_2.csv']))
df3 = pd.read_csv(io.BytesIO(uploaded['PR_3.csv']))
df4 = pd.read_csv(io.BytesIO(uploaded['PR_4.csv']))
df5 = pd.read_csv(io.BytesIO(uploaded['PR_5.csv']))



#Step 2: Fetching Ensembl Protein IDs

import pandas as pd
import requests
import json
import time

# Function to get Ensembl ID
def get_ensembl_id(names):
    server = "http://rest.ensembl.org"
    for name in names:
        if name == 'nan':
            continue
        print(f"Fetching Ensembl ID for: {name}")
        ext = f"/xrefs/symbol/homo_sapiens/{name}?content-type=application/json"
        r = requests.get(server + ext, headers={"Content-Type": "application/json"})

        if r.ok:
            decoded = r.json()
            if decoded:
                ensembl_id = decoded[0]['id']
                print(f"Found Ensembl ID: {ensembl_id}")
                return ensembl_id

    print(f"No Ensembl ID found for: {names}")
    return None

# Function to apply get_ensembl_id
def apply_get_ensembl_id(row):
    gene_names = [name for name in str(row['Gene_Name']).split(';') if name != 'nan']
    protein_ids = [id for id in str(row['Protein_ID']).split(';') if id != 'nan']
    protein_names = [name for name in str(row['Protein_name']).split(';') if name != 'nan']

    for names in [gene_names, protein_ids, protein_names]:
        if names:  # Check if the list is not empty
            ensembl_id = get_ensembl_id(names)
            if ensembl_id is not None:
                return ensembl_id

    print(f"Ensembl ID not found for row {row.name}")
    return None

# Loop through all CSV files
for i, df in enumerate([df1, df2, df3, df4, df5]):
    print(f"Processing CSV file {i+1}")

    # Apply the function only to the first three columns in each row
    df['Ensembl_ID'] = df[['Gene_Name', 'Protein_ID', 'Protein_name']].apply(apply_get_ensembl_id, axis=1)

    # Drop rows where Ensembl_ID is NaN
    df.dropna(subset=['Ensembl_ID'], inplace=True)

    # Drop the 'Gene_Name', 'Protein_ID', 'Protein_name' columns
    df.drop(['Gene_Name', 'Protein_ID', 'Protein_name'], axis=1, inplace=True)

    # Save the updated dataframe to a new CSV file
    df.to_csv(f'PR_{i+1}_with_Ensembl.csv', index=False)

from google.colab import files

# Save the dataframe to a CSV file
df1.to_csv('PR_1_with_Ensembl.csv', index=False)
df2.to_csv('PR_2_with_Ensembl.csv', index=False)
df3.to_csv('PR_3_with_Ensembl.csv', index=False)
df4.to_csv('PR_4_with_Ensembl.csv', index=False)
df5.to_csv('PR_5_with_Ensembl.csv', index=False)

# Download the CSV file
files.download('PR_1_with_Ensembl.csv')
files.download('PR_2_with_Ensembl.csv')
files.download('PR_3_with_Ensembl.csv')
files.download('PR_4_with_Ensembl.csv')
files.download('PR_5_with_Ensembl.csv')



#Step 3: Turning Ensembl IDs to Uniprot ones

!pip install mygene
import pandas as pd
import mygene

def get_uniprot_id(ensembl_id):
    print(f"Fetching Uniprot ID for: {ensembl_id}")
    mg = mygene.MyGeneInfo()
    result = mg.query('ensembl.gene:' + ensembl_id, scopes='ensembl.gene', fields='uniprot', species='human')
    uniprot_id = None
    if result and 'hits' in result and len(result['hits']) > 0:
        hit = result['hits'][0]
        if 'uniprot' in hit:
            uniprot_id = hit['uniprot']['Swiss-Prot'] if 'Swiss-Prot' in hit['uniprot'] else None
            if uniprot_id:
                print(f"Found Uniprot ID: {uniprot_id}")
            else:
                print(f"No Uniprot ID found for: {ensembl_id}")
    else:
        print(f"No Uniprot ID found for: {ensembl_id}")
    return uniprot_id

# Loop through all CSV files
for i, df in enumerate([df1, df2, df3, df4, df5]):
    print(f"Processing CSV file {i+1}")

    # Fetch the Uniprot ID for each Ensembl ID
    df['Uniprot_ID'] = df['Ensembl_ID'].apply(get_uniprot_id)

    # Drop rows where Uniprot_ID is NaN
    df.dropna(subset=['Uniprot_ID'], inplace=True)

    # Save the updated dataframe to a new CSV file
    df.to_csv(f'PR_{i+1}_with_Uniprot.csv', index=False)

# Save the dataframe to a CSV file
df1.to_csv('PR_1_with_Uniprot.csv', index=False)
df2.to_csv('PR_2_with_Uniprot.csv', index=False)
df3.to_csv('PR_3_with_Uniprot.csv', index=False)
df4.to_csv('PR_4_with_Uniprot.csv', index=False)
df5.to_csv('PR_5_with_Uniprot.csv', index=False)


# Download the CSV file
files.download('PR_1_with_Uniprot.csv')
files.download('PR_2_with_Uniprot.csv')
files.download('PR_3_with_Uniprot.csv')
files.download('PR_4_with_Uniprot.csv')
files.download('PR_5_with_Uniprot.csv')



#Step 4: Merging proteomics datasets using Uniprot_IDs

import pandas as pd
import ast

# Function to get the first Uniprot ID from the list
def get_first_uniprot_id(id_list):
    if isinstance(id_list, str) and id_list.startswith('['):
        return ast.literal_eval(id_list)[0]
    else:
        return id_list

# List of file names
file_names = ['PR_1_with_Uniprot.csv', 'PR_2_with_Uniprot.csv', 'PR_3_with_Uniprot.csv', 'PR_4_with_Uniprot.csv', 'PR_5_with_Uniprot.csv']

# List to store dataframes
dataframes = []

# Read each dataframe and preprocess it
for file_name in file_names:
    df = pd.read_csv(file_name)

    # Apply the function to the 'Uniprot_ID' column
    df['Uniprot_ID'] = df['Uniprot_ID'].apply(get_first_uniprot_id)

    # Remove any duplicates based on 'Ensembl_ID' and 'Uniprot_ID' within each dataframe
    df = df.drop_duplicates(subset=['Ensembl_ID', 'Uniprot_ID'], keep='first')

    dataframes.append(df)

# Merge the dataframes using outer join on 'Ensembl_ID' and 'Uniprot_ID'
final_df = dataframes[0]
for df in dataframes[1:]:
    final_df = pd.merge(final_df, df, on=['Ensembl_ID', 'Uniprot_ID'], how='outer')

# Sort the final dataframe by 'Ensembl_ID' and 'Uniprot_ID'
final_df = final_df.sort_values(['Ensembl_ID', 'Uniprot_ID'])

# Fill missing values with 0
final_df = final_df.fillna(0)

# Move 'Uniprot_ID' column to the second position
cols = final_df.columns.tolist()
cols.insert(1, cols.pop(cols.index('Uniprot_ID')))
final_df = final_df[cols]

# Filter rows where all conditions are 0
condition_columns = final_df.columns[2:]  # Excludes 'Ensembl_ID' and 'Uniprot_ID'
final_df = final_df[final_df[condition_columns].sum(axis=1) != 0]

# Save the final dataframe to a new CSV file named 'merged_df.csv'
final_df.to_csv('merged_df.csv', index=False)

# Download the CSV file
files.download('merged_df.csv')



#Step 5: Fetching cellular components from Uniprot

!pip install pandas
!pip install requests
!pip install tqdm

import pandas as pd
import requests
import tqdm
import re

# Load the merged dataframe
df = pd.read_csv('merged_df.csv')

# Define a function to fetch protein information from UniProt
def get_protein_category(uniprot_id):
    # Query the UniProt API
    try:
        response = requests.get(f'https://www.uniprot.org/uniprot/{uniprot_id}.txt')
        response.raise_for_status()
    except requests.exceptions.RequestException as err:
        print(f"Error occurred with UniProt API for UniProt ID '{uniprot_id}': {err}")
        return 'Others'

    # Get the protein's subcellular location
    lines = response.text.split('\n')
    location_lines = [line for line in lines if line.startswith('CC   -!- SUBCELLULAR LOCATION:')]
    location_text = ' '.join([line.split(': ')[1] for line in location_lines])

    # Remove any text starting from { or [
    location_text = re.sub("[\{\[].*", "", location_text)

    # Split by any punctuation (.,;)
    location_text = re.split("[.,;]", location_text)[0]

    # Strip the white space
    location_text = location_text.strip()

    # Classify "Secreted" as "Extracellular Space"
    if location_text.lower() == 'secreted':
        location_text = 'Extracellular Space'

    # Check for cellular components and group them
    if 'extracellular matrix' in location_text.lower():
        location_text = 'Extracellular Matrix'
    elif 'extracellular space' in location_text.lower():
        location_text = 'Extracellular Space'
    elif 'nucleus' in location_text.lower():
        location_text = 'Nucleus'
    elif any(x in location_text.lower() for x in ['cytosol', 'cytoskeleton']):
        location_text = 'Cytoplasm'
    elif any(x in location_text.lower() for x in ['membrane', 'endoplasmic reticulum', 'golgi apparatus']):
        location_text = 'Membrane'
    elif 'mitochondrion' in location_text.lower():
        location_text = 'Mitochondrion'
    elif any(x in location_text.lower() for x in ['lysosome', 'endosome', 'peroxisome']):
        location_text = 'Vesicles'

    if location_text == '':
        print(f"UniProt ID '{uniprot_id}' does not have an identified subcellular location.")
        return 'Others'
    else:
        print(f"UniProt ID '{uniprot_id}' has location '{location_text}'.")
        return location_text

# Apply the function to the 'Uniprot_ID' column
for uniprot_id in tqdm.tqdm(df['Uniprot_ID'], desc='Fetching categories'):
    df.loc[df['Uniprot_ID'] == uniprot_id, 'Category'] = get_protein_category(uniprot_id)

# Save the updated dataframe
df.to_csv('merged_df_with_categories.csv', index=False)

# Count and print the number of proteins in each category
category_counts = df['Category'].value_counts()
print("\nNumber of proteins in each category:")
for category, count in category_counts.items():
    print(f"{category}: {count}")

# Download the CSV file
files.download('merged_df_with_categories.csv')
