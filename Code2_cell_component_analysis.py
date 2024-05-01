#Step 1: Uploading Files with Both Ensembl and Uniprot IDs

from google.colab import files
import io
import pandas as pd
import numpy as np

#Load data
uploaded = files.upload()
df = pd.read_csv(io.BytesIO(uploaded['merged_df_with_categories.csv']))



#Step 2: Breaking down the database based on cellular component composition

import pandas as pd
import zipfile
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the merged dataframe with categories
df = pd.read_csv('merged_df_with_categories.csv')

# Count the number of proteins in each category
category_counts = df['Category'].value_counts()

# Get the top 5 categories (including 'Others')
top_categories = category_counts.index[:5].tolist()

# If 'Others' not in the top 5, append it to the list
if 'Others' not in top_categories:
    top_categories.append('Others')

# Assign 'Others' to the proteins that are not in the top 5 categories
df.loc[~df['Category'].isin(top_categories), 'Category'] = 'Others'

# Get unique categories
categories = df['Category'].unique()

# Initialize a list to store csv file names
csv_files = []

# Write separate CSV file for each category
for category in categories:
    filename = f'merged_df_{category.replace(" ", "_")}.csv'
    df_category = df[df['Category'] == category]
    df_category.to_csv(filename, index=False)
    csv_files.append(filename)
    print(f"Category '{category}' has {len(df_category)} proteins.")

# Create a ZipFile object
zip_filename = 'Grouped_Proteomics_Repository.zip'
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    # Add each csv file to the zip file
    for file in csv_files:
        zipf.write(file)

print(f'Separate CSV files have been written for the following categories: {", ".join(categories)}')
print(f'All CSV files are also available in the zip file: {zip_filename}')

from google.colab import files
files.download('Grouped_Proteomics_Repository.zip')



#Step 3: Preanalysis using PCA and LDA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
import zipfile

from google.colab import files

markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
svg_files = []  # List to store SVG filenames

def process_and_plot(filename, data_frame):
    # Drop the specified columns
    data_frame = data_frame.drop(columns=['Uniprot_ID', 'Category'])

    df_T = data_frame.set_index('Ensembl_ID').T

    # Data standardization
    scaler = StandardScaler()
    df_T_scaled = scaler.fit_transform(df_T)

    conditions = [idx.split('-')[0] for idx in df_T.index]
    unique_conditions = np.unique(conditions)

    # Adjust font size
    font = {'size': 15, 'weight': 'bold'}
    plt.rc('font', **font)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df_T_scaled)
    plt.figure(figsize=(8, 6))
    for i, condition in enumerate(unique_conditions):
        ix = [i for i, cond in enumerate(conditions) if cond == condition]
        plt.scatter(X_pca[ix, 0], X_pca[ix, 1], marker=markers[i % len(markers)], label=condition, s=72)
    plt.xlabel('First Principal Component', weight='bold')
    plt.ylabel('Second Principal Component', weight='bold')
    plt.title(f'PCA Scatter Plot for {filename}', weight='bold')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    file_name = f'PCA_{filename}.svg'
    plt.savefig(file_name, format='svg', bbox_inches='tight')
    plt.show()
    svg_files.append(file_name)

    # LDA
    lda = LDA(n_components=2)
    X_lda = lda.fit_transform(df_T_scaled, conditions)
    plt.figure(figsize=(8, 6))
    for i, condition in enumerate(unique_conditions):
        ix = [j for j, cond in enumerate(conditions) if cond == condition]
        plt.scatter(X_lda[ix, 0], X_lda[ix, 1], marker=markers[i % len(markers)], label=condition, s=72)
    plt.xlabel('LDA Dimension 1', weight='bold')
    plt.ylabel('LDA Dimension 2', weight='bold')

    plt.title(f'LDA Scatter Plot for {filename}', weight='bold')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    file_name = f'LDA_{filename}.svg'
    plt.savefig(file_name, format='svg', bbox_inches='tight')
    plt.show()
    svg_files.append(file_name)


# Starting with combined csv file
df = pd.read_csv('merged_df_with_categories.csv')
process_and_plot("merged_df_with_categories", df)

# Process the category csv files inside the zip
zip_filename = 'Grouped_Proteomics_Repository.zip'
archive = zipfile.ZipFile(zip_filename, 'r')
csv_filenames = archive.namelist()

for csv_filename in csv_filenames:
    with archive.open(csv_filename) as file:
        df = pd.read_csv(file)
    process_and_plot(csv_filename[:-4], df)  # remove .csv from filename for title and saved SVG name

# Zip the SVG files
zip_file_name = 'Preanalysis_full.zip'
with zipfile.ZipFile(zip_file_name, 'w') as zipf:
    for svg_file in svg_files:
        zipf.write(svg_file, arcname=svg_file)

# Download the ZIP
files.download('/content/' + zip_file_name)



#Step 4: Feature Importance

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import zipfile
from google.colab import files

# Define a color palette using the new method
colors = matplotlib.colormaps['tab20']  # This provides 20 distinct colors

def process_and_plot(file_name, df):
    # Drop unwanted columns
    df_conditions = df.drop(['Ensembl_ID', 'Uniprot_ID', 'Category'], axis=1)

    # Extract base condition names by removing repeat numbers
    condition_names = df_conditions.columns.str.rsplit('-', n=1).str[0].unique()

    # Calculate average for each base condition over its repeats
    df_avg = pd.DataFrame(index=df_conditions.index)
    for condition in condition_names:
        condition_columns = [col for col in df_conditions.columns if col.startswith(condition)]
        df_avg[condition] = df_conditions[condition_columns].mean(axis=1)

    # Compute the feature importance for each condition using RandomForest
    importance_data = []
    for condition in df_avg.columns:
        X = df_avg.drop(condition, axis=1)
        y = df_avg[condition]

        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)

        importance = model.feature_importances_
        importance_data.append(pd.DataFrame({'Protein': X.columns, 'Importance': importance, 'Condition': condition}))

    # Combine importance data into a single DataFrame
    importance_df = pd.concat(importance_data)

    # Sort by Condition for alphabetical order
    importance_df = importance_df.sort_values('Condition')

    # Create the boxplot with enhanced visuals
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=importance_df, x='Condition', y='Importance', hue='Condition', showfliers=False, linewidth=2.5, palette=[colors(i) for i in range(len(condition_names))], dodge=False)
    plt.title('Feature Importances by Condition', fontsize=18)
    plt.xlabel('Condition', fontsize=16)
    plt.ylabel('Importance', fontsize=16)
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    plt.legend(title='Condition', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # Adjust y-axis limit if necessary (you can change these numbers based on your data)
    plt.ylim(0, 0.35)

    file_name_svg = f'Condition_Importances_{file_name}.svg'
    plt.savefig(file_name_svg, format='svg', bbox_inches='tight')
    plt.show()
    return file_name_svg

# Starting with the file outside of the zip
df = pd.read_csv('merged_df_with_categories.csv')
svg_files = [process_and_plot("merged_df_with_categories", df)]

# Now process the files inside the zip
zip_filename = 'Grouped_Proteomics_Repository.zip'
archive = zipfile.ZipFile(zip_filename, 'r')
csv_filenames = archive.namelist()

for csv_filename in csv_filenames:
    with archive.open(csv_filename) as file:
        df = pd.read_csv(file)
    svg_files.append(process_and_plot(csv_filename[:-4], df))  # remove .csv from filename for title and saved SVG name

# Zip the SVG files
zip_file_name = 'Condition_Importances_full.zip'
with zipfile.ZipFile(zip_file_name, 'w') as zipf:
    for svg_file in svg_files:
        zipf.write(svg_file, arcname=svg_file)

# Download the ZIP
files.download('/content/' + zip_file_name)


#Step 4: Expression Distribution

!pip install rpy2
%load_ext rpy2.ipython

%%R
install.packages("BiocManager")
BiocManager::install("limma")

import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import glob
from sklearn.preprocessing import StandardScaler
from google.colab import files
import seaborn as sns

# Set Seaborn context for better font sizes
sns.set_context("talk")

# Create a list to store names of generated SVG files
generated_svg_files = []

def process_category(file_name, df):
    # Drop non-numeric columns
    df_numeric = df.drop(['Ensembl_ID', 'Uniprot_ID', 'Category'], axis=1)

    # Standard scaling for numerical columns
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

    # Extract base conditions from the scaled columns
    conditions = set()
    for col_name in df_scaled.columns:
        condition = col_name.rsplit('-', 1)[0]
        conditions.add(condition)

    # Sort the conditions alphabetically
    sorted_conditions = sorted(conditions)

    # Create a DataFrame to store mean and std for each condition
    summary_df = pd.DataFrame(columns=['Condition', 'Mean', 'Std'])

    # Compute mean and std for each condition
    for condition in conditions:
        # Columns for the current condition
        condition_columns = [col for col in df_scaled.columns if col.startswith(condition)]

        # Compute mean and standard deviation across the columns
        mean_expression = df_scaled[condition_columns].mean(axis=0)
        std_expression = mean_expression.std()

        # Append to the summary DataFrame
        summary_df = pd.concat([summary_df, pd.DataFrame({
            'Condition': [condition],
            'Mean': [mean_expression.mean()],
            'Std': [std_expression]
        })], ignore_index=True)

    # Creating a long-form dataframe and adjust the Condition column
    long_df = pd.melt(df_scaled, value_vars=df_scaled.columns, var_name='Condition', value_name='Expression')
    long_df['Condition'] = long_df['Condition'].apply(lambda x: x.rsplit('-', 1)[0])

    # Plotting the results using Violin Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=long_df, x='Condition', y='Expression', inner='quartile', order=sorted_conditions)
    plt.ylim(-1, 1)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Expression', fontsize=14)
    plt.title(f'Expression (Violin Plot) for {file_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{file_name[:-4]}_violin.svg", format='svg')
    generated_svg_files.append(f"{file_name[:-4]}_violin.svg")
    plt.show()

# Load the zip file
zip_filename = 'Grouped_Proteomics_Repository.zip'
archive = zipfile.ZipFile(zip_filename, 'r')

# List of CSV files including the original CSV file
csv_files = [
    'merged_df_with_categories.csv',
] + archive.namelist()

# Process each CSV file
for file_name in csv_files:
    if file_name in archive.namelist():
        with archive.open(file_name) as file:
            df = pd.read_csv(file)
    else:
        df = pd.read_csv(file_name)
    process_category(file_name, df)

# Create a ZIP file with all SVG plots
with zipfile.ZipFile('Differential_Expression_Analysis_plots.zip', 'w') as zipf:
    for file_name in generated_svg_files:
        zipf.write(file_name, arcname=file_name)

files.download('/content/Differential_Expression_Analysis_plots.zip')



#Step 6: Correlation Network Analysis

!pip install networkx matplotlib scipy pandas
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import pandas as pd
import zipfile
import glob
from google.colab import files
from sklearn.preprocessing import StandardScaler

def preprocess_and_scale(df):
    df_numeric = df.drop(['Ensembl_ID', 'Uniprot_ID', 'Category'], axis=1)
    df_numeric = df_numeric.groupby(df_numeric.columns.str.split('-').str[0], axis=1).mean()

    # Scale the data
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

    return df_scaled

def create_corr_network(df, file_name, corr_threshold=0.6):
    df = preprocess_and_scale(df)

    columns = df.columns
    corr_matrix = df.corr()

    # Create a networkx graph
    G = nx.Graph()

    # Add nodes to the graph
    for col in columns:
        G.add_node(col)

    # Add edges to the graph (only for correlations above the threshold)
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if j > i and np.abs(corr_matrix.loc[col1, col2]) > corr_threshold:
                G.add_edge(col1, col2, weight=corr_matrix.loc[col1, col2])

    # Assign colors to conditions
    color_map = dict(zip(df.columns, mcolors.CSS4_COLORS.keys()))
    node_colors = [color_map[node] for node in G.nodes]

    # Draw the graph
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, node_color=node_colors, node_size=100, font_size=12, font_weight='bold', edge_color='grey', width=2.0)

    # Draw legend
    for label, color in color_map.items():
        plt.plot([], [], 'o', color=color, label=label)
    plt.legend(fontsize=10)  # Adjusted fontsize for legend

    plt.title(f"Correlation Network for {file_name}", fontsize=18)
    plt.savefig(f"{file_name[:-4]}.svg", format='svg')
    plt.show()

# Create a list to store names of generated SVG files
generated_svg_files = []

# Load the zip file
zip_filename = 'Grouped_Proteomics_Repository.zip'
archive = zipfile.ZipFile(zip_filename, 'r')

# Get the CSV filenames in the zip file
csv_filenames = archive.namelist()

# Process each CSV file in the zip file
for csv_filename in csv_filenames:
    with archive.open(csv_filename) as file:
        df = pd.read_csv(file)
    print(f"Creating correlation network for {csv_filename}...")
    create_corr_network(df, csv_filename)
    generated_svg_files.append(f"{csv_filename[:-4]}.svg")  # append the generated SVG file name to the list

# Process the original merged file with all categories
original_csv_filename = 'merged_df_with_categories.csv'
original_df = pd.read_csv(original_csv_filename)
print(f"Creating correlation network for {original_csv_filename}...")
create_corr_network(original_df, original_csv_filename)
generated_svg_files.append(f"{original_csv_filename[:-4]}.svg")  # append the generated SVG file name to the list

# Create a ZIP file with all SVG plots
with zipfile.ZipFile('Correlation_Network_plots.zip', 'w') as zipf:
    for file_name in generated_svg_files:
        zipf.write(file_name, arcname=file_name)

files.download('/content/Correlation_Network_plots.zip')



#Step 7: Volcano plot

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
import zipfile

def calculate_stats(cat_samples, rest_samples):
    t_stat, p_val = ttest_ind(cat_samples, rest_samples, equal_var=False, nan_policy='omit')
    mean_cat = np.mean(cat_samples)
    mean_rest = np.mean(rest_samples)
    log2_fold_change = np.log2((mean_cat + 1e-10) / (mean_rest + 1e-10))
    return p_val, log2_fold_change

# Load the original dataframe
original_df = pd.read_csv('merged_df_with_categories.csv')

# Top categories file names
categories_files = [
    'merged_df_Cytoplasm.csv',
    'merged_df_Extracellular_Space.csv',
    'merged_df_Membrane.csv',
    'merged_df_Nucleus.csv',
    'merged_df_Others.csv'
]

# Create a folder to store the plots
plot_folder = 'plots'
os.makedirs(plot_folder, exist_ok=True)

# Scaler for standardizing data
scaler = StandardScaler()

# List of conditions without repeats
conditions = list(set([col.split('-')[0] for col in original_df.columns[2:-1]]))

# Create a ZIP file to store the plots
with zipfile.ZipFile('Volcano_Plots.zip', 'w') as zipf:
    for category_file in categories_files:
        category_df = pd.read_csv(category_file)
        category_proteins = category_df['Ensembl_ID']
        rest_df = original_df.loc[~original_df['Ensembl_ID'].isin(category_proteins)].copy()

        for condition in conditions:
            replicate_cols = [col for col in original_df.columns if col.startswith(condition)]
            if all(col in category_df.columns for col in replicate_cols):
                category_df[condition] = category_df[replicate_cols].mean(axis=1)
            if all(col in rest_df.columns for col in replicate_cols):
                rest_df[condition] = rest_df[replicate_cols].mean(axis=1)

        category_name = os.path.basename(category_file).replace('merged_df_', '').replace('.csv', '')

        p_values = []
        log2_fold_changes = []

        for condition in conditions:
            category_samples = category_df[condition].dropna().values
            rest_samples = rest_df[condition].dropna().values

            p_val, log2_fold_change = calculate_stats(category_samples, rest_samples)

            if not np.isnan(p_val) and not np.isnan(log2_fold_change):
                p_values.append(p_val)
                log2_fold_changes.append(log2_fold_change)
                plt.scatter(log2_fold_change, -np.log10(p_val), s=30)
                plt.annotate(condition, (log2_fold_change, -np.log10(p_val)), fontsize=8, alpha=0.7)

        plt.title(f'Volcano plot for {category_name}', fontsize=16)
        plt.xlabel('Log2 Fold Change', fontsize=14)
        plt.ylabel('-Log10 P-value', fontsize=14)
        plt.axhline(y=-np.log10(0.05), color='grey', linestyle='--')
        plt.axvline(x=0, color='grey', linestyle='--')
        plt.xlim([min(log2_fold_changes) - 1, max(log2_fold_changes) + 1])
        plt.ylim([0, max(-np.log10(p_values)) + 1])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plot_filename = os.path.join(plot_folder, f'{category_name}.svg')
        plt.savefig(plot_filename)
        plt.show()
        plt.close()

        zipf.write(plot_filename, arcname=f'{category_name}.svg')
        print(f"Plot saved to {plot_filename}")

files.download('/content/Volcano_Plots.zip')



#Step 8: correlation network for proteins in each condition

import pandas as pd
import zipfile
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import files
from sklearn.preprocessing import StandardScaler

def preprocess_and_avg_conditions(df):
    """
    Preprocess the dataframe by averaging repeats for each condition, scaling the data, and returning the processed dataframe.
    """
    # Drop non-numeric columns
    df_numeric = df.drop(['Ensembl_ID', 'Uniprot_ID', 'Category'], axis=1)

    # Group by condition name (before '-') and calculate mean for each condition
    df_avg = df_numeric.groupby(df_numeric.columns.str.split('-').str[0], axis=1).mean()
    
    # Initialize StandardScaler
    scaler = StandardScaler()
    
    # Scale the averaged data
    scaled_data = scaler.fit_transform(df_avg)
    
    # Convert scaled data back to DataFrame with original column names
    df_scaled = pd.DataFrame(scaled_data, columns=df_avg.columns)
    
    return df_scaled

def compute_correlation(df):
    """
    Compute the correlation matrix for the given dataframe.
    """
    return df.corr()

def visualize_corr_matrix(corr_matrix, title, filename):
    """
    Visualize the correlation matrix using a heatmap, display it, and save as SVG.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(title, fontsize=18)
    plt.savefig(filename + '.svg', format='svg')
    plt.show()  # To display the heatmap

# Load the combined dataframe
df_combined = pd.read_csv('merged_df_with_categories.csv')

# Initialize the list to keep track of generated SVGs
svg_files = []

# Compute the correlation matrix for the combined dataset
combined_corr = compute_correlation(preprocess_and_avg_conditions(df_combined))

# Visualize the correlation matrix for the combined dataset and append to SVG list
visualize_corr_matrix(combined_corr, "Combined Dataset Correlation Matrix", "combined_dataset")
svg_files.append("combined_dataset.svg")

# Dictionary to store correlation matrices for each category
category_correlations = {}

# Load the zip file containing individual category CSVs
zip_filename = 'Grouped_Proteomics_Repository.zip'
with zipfile.ZipFile(zip_filename, 'r') as archive:
    for csv_filename in archive.namelist():
        with archive.open(csv_filename) as file:
            df_category = pd.read_csv(file)

        # Compute the correlation matrix for this category and store it
        category_corr = compute_correlation(preprocess_and_avg_conditions(df_category))
        category_correlations[csv_filename] = category_corr

        # Visualize the correlation matrix for this category, save as SVG, and append to SVG list
        svg_name = csv_filename[:-4]  # remove .csv extension
        visualize_corr_matrix(category_corr, f"Correlation Matrix for {svg_name}", svg_name)
        svg_files.append(svg_name + ".svg")

# Zip the SVG files
zip_file_name = 'Correlation_Matrix.zip'
with zipfile.ZipFile(zip_file_name, 'w') as zipf:
    for svg_file in svg_files:
        zipf.write(svg_file, arcname=svg_file)

# Download the ZIP file
files.download('/content/' + zip_file_name)
