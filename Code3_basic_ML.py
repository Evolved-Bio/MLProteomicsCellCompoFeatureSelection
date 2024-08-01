#Step 1: Uploading Files with Both Ensembl and Uniprot IDs

from google.colab import files
import io
import pandas as pd
import numpy as np

#Load data
uploaded = files.upload()
df = pd.read_csv(io.BytesIO(uploaded['merged_df_with_categories.csv']))




#Step 2: Breaking down the database

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




#Step 3: Random forest for the combined dataset with hyper parameter tunning

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile

# Load the dataframe
df = pd.read_csv('merged_df_with_categories.csv')

# Transpose the dataframe and drop unwanted columns
df = df.transpose()
df = df.drop(['Ensembl_ID', 'Uniprot_ID', 'Category'])

# Initialize label encoder
le = LabelEncoder()

# Group conditions by removing the dash and numbers after it
conditions = df.index.str.split('-').str[0].to_list()

# Apply label encoding to the conditions
y = le.fit_transform(conditions)

# Extract features
X = df.values

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize a base Random Forest classifier
clf = RandomForestClassifier(random_state=42)

# K-Fold Cross Validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("Starting Grid Search for hyperparameter tuning...")

# Grid Search for hyperparameter tuning with verbose set to 3 for more detailed updates
grid_search = GridSearchCV(clf, param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=3)
grid_search.fit(X_train, y_train)

print("Grid Search completed!")
print(f"Best hyperparameters found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy score: {grid_search.best_score_:.4f}")

# Evaluate the best model on the test set
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# Print model performance
print("\nModel Performance on Test Set:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='micro'))
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# All unique classes in the dataset
all_classes = np.unique(y)
confusion = confusion_matrix(y_test, y_pred, labels=all_classes)

# Plotting the confusion matrix
all_class_names = le.inverse_transform(np.unique(y))
plt.figure(figsize=(10,7))
sns.heatmap(confusion, annot=True, fmt="d", cmap='Blues', xticklabels=all_class_names, yticklabels=all_class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Random Forest Confusion Matrix Heatmap with Hyper Tuning')

# Save the heatmap as an SVG file
file_name = 'Merged_RF_confusion_matrix_HyperTunned.svg'
plt.savefig(file_name, format='svg')
plt.show()  # This will display the heatmap

# Create a ZIP file with the generated SVG plot
svg_files = [file_name]
zip_file_name = 'Merged_RF_confusion_matrix_HyperTunned.zip'

with zipfile.ZipFile(zip_file_name, 'w') as zipf:
    for file in svg_files:
        zipf.write(file, arcname=file)

files.download('/content/Merged_RF_confusion_matrix_HyperTunned.zip')




#Step 4: RandomForest Feature selection

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# Load the scaled and merged file
df = pd.read_csv('merged_df_with_categories.csv', index_col='Ensembl_ID')

# Drop duplicate proteins based on Ensembl_ID and Uniprot_ID
df = df.drop_duplicates(subset=['Uniprot_ID'], keep='first')

# Drop Uniprot_ID and Category columns for the feature selection process
X = df.drop(columns=['Uniprot_ID', 'Category'])

# Initialize label encoder
le = LabelEncoder()

# Function to extract conditions from the columns
def extract_condition(column):
    return column.rsplit('-', 1)[0]  # Split based on last dash to remove repeat numbers

# Create labels from column
conditions = [extract_condition(col) for col in X.columns]
y = le.fit_transform(conditions)

# Separate features
X = X.T.values  # Transpose the dataframe to match conditions with features

# Initialize the selector
selector = SelectFromModel(estimator=RandomForestClassifier(n_estimators=100, random_state=42))

# Fit the selector to the data
selector.fit(X, y)

# Get a mask, or boolean array, of the features selected
mask = selector.get_support()

# Apply the mask to get the selected features
selected_features = df.index[mask]

# Create a new DataFrame with only the selected features
feature_selected_df = df.loc[selected_features]

# Print some information
print("Number of features selected:", len(selected_features))
print("Accuracy of model with selected features: ", cross_val_score(selector.estimator_, X[:, mask], y, cv=5).mean())
print("Selected features:\n", selected_features)

# Save DataFrame to CSV
feature_selected_df.to_csv('feature_selected_df.csv')




#Step 5: Random Forest ML model for Feature selected version with hyper parameter tunning

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataframe
df = pd.read_csv('feature_selected_df.csv')

# Transpose the dataframe and drop unwanted columns
df = df.transpose()
df = df.drop(['Ensembl_ID', 'Uniprot_ID', 'Category'])

# Initialize label encoder
le = LabelEncoder()

# Group conditions by removing the dash and numbers after it
conditions = df.index.str.split('-').str[0].to_list()

# Apply label encoding to the conditions
y = le.fit_transform(conditions)

# Extract features
X = df.values

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize a base Random Forest classifier
clf = RandomForestClassifier(random_state=42)

# K-Fold Cross Validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("Starting Grid Search for hyperparameter tuning...")

# Grid Search for hyperparameter tuning with verbose set to 3 for more detailed updates
grid_search = GridSearchCV(clf, param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=3)
grid_search.fit(X_train, y_train)

print("Grid Search completed!")
print(f"Best hyperparameters found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy score: {grid_search.best_score_:.4f}")

# Evaluate the best model on the test set
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# Print model performance
print("\nModel Performance on Test Set:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='micro'))
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# All unique classes in the dataset
all_classes = np.unique(y)
confusion = confusion_matrix(y_test, y_pred, labels=all_classes)

# Plotting the confusion matrix
all_class_names = le.inverse_transform(np.unique(y))
plt.figure(figsize=(10,7))
sns.heatmap(confusion, annot=True, fmt="d", cmap='Blues', xticklabels=all_class_names, yticklabels=all_class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Random Forest Confusion Matrix Heatmap with Hyper Tuning for FS')

# Save the heatmap as an SVG file
file_name = 'FS_Merged_RF_confusion_matrix_HyperTunned.svg'
plt.savefig(file_name, format='svg')
plt.show()  # This will display the heatmap

# Create a ZIP file with the generated SVG plot
svg_files = [file_name]
zip_file_name = 'FS_Merged_RF_confusion_matrix_HyperTunned.zip'

with zipfile.ZipFile(zip_file_name, 'w') as zipf:
    for file in svg_files:
        zipf.write(file, arcname=file)  # arcname parameter sets the name for the file inside the ZIP

files.download('/content/FS_Merged_RF_confusion_matrix_HyperTunned.zip')




#Step 6: Feature selection with 25, 50, and 75% retention of proteins for ablation study
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the scaled and merged file
df = pd.read_csv('merged_df_with_categories.csv', index_col='Ensembl_ID')

# Drop duplicate proteins based on Ensembl_ID and Uniprot_ID
df = df.drop_duplicates(subset=['Uniprot_ID'], keep='first')

# Drop Uniprot_ID and Category columns for the feature selection process
X = df.drop(columns=['Uniprot_ID', 'Category'])

# Initialize label encoder
le = LabelEncoder()

# Function to extract conditions from the columns
def extract_condition(column):
    return column.rsplit('-', 1)[0]  # Split based on last dash to remove repeat numbers

# Create labels from columns
conditions = [extract_condition(col) for col in X.columns]
y = le.fit_transform(conditions)

# Separate features
X = X.T.values  # Transpose the dataframe to match conditions with features

# Initialize the selector
selector = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the selector to the data
selector.fit(X, y)

# Get feature importances
importances = selector.feature_importances_

# Sort features by importance
indices = np.argsort(importances)[::-1]

# Calculate the number of features to select for 25%, 50%, and 75% retention
n_features = len(importances)
n_25 = int(n_features * 0.25)
n_50 = int(n_features * 0.50)
n_75 = int(n_features * 0.75)

# Select top features for each retention level
selected_features_25 = df.index[indices[:n_25]]
selected_features_50 = df.index[indices[:n_50]]
selected_features_75 = df.index[indices[:n_75]]

# Create new DataFrames for each retention level
feature_selected_df_25 = df.loc[selected_features_25]
feature_selected_df_50 = df.loc[selected_features_50]
feature_selected_df_75 = df.loc[selected_features_75]

# Save DataFrames to CSV
feature_selected_df_25.to_csv('feature_selected_df_25.csv')
feature_selected_df_50.to_csv('feature_selected_df_50.csv')
feature_selected_df_75.to_csv('feature_selected_df_75.csv')

# Print some information
print("Number of features selected (25%):", len(selected_features_25))
print("Number of features selected (50%):", len(selected_features_50))
print("Number of features selected (75%):", len(selected_features_75))

print("Accuracy of model with 25% selected features: ", cross_val_score(selector, X[:, indices[:n_25]], y, cv=5).mean())
print("Accuracy of model with 50% selected features: ", cross_val_score(selector, X[:, indices[:n_50]], y, cv=5).mean())
print("Accuracy of model with 75% selected features: ", cross_val_score(selector, X[:, indices[:n_75]], y, cv=5).mean())





#Step 7: Random Forest ML model for ablation study

!pip install memory_profiler

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
from google.colab import files
import time
from memory_profiler import memory_usage

# Function to measure the memory usage and execution time
def profile_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        mem_usage_before = memory_usage()[0]
        result = func(*args, **kwargs)
        mem_usage_after = memory_usage()[0]
        end_time = time.time()
        print(f"Memory usage before: {mem_usage_before} MB")
        print(f"Memory usage after: {mem_usage_after} MB")
        print(f"Memory usage increased by: {mem_usage_after - mem_usage_before} MB")
        print(f"Execution time: {end_time - start_time} seconds")
        return result
    return wrapper

# List of feature-selected files
feature_selected_files = ['feature_selected_df_25.csv', 'feature_selected_df_50.csv', 'feature_selected_df_75.csv']

@profile_function
def process_file(file):
    print(f"\nProcessing {file}...")
    
    # Load the dataframe
    df = pd.read_csv(file)

    # Transpose the dataframe and drop unwanted columns
    df = df.transpose()
    df = df.drop(['Ensembl_ID', 'Uniprot_ID', 'Category'])

    # Initialize label encoder
    le = LabelEncoder()

    # Group conditions by removing the dash and numbers after it
    conditions = df.index.str.split('-').str[0].to_list()

    # Apply label encoding to the conditions
    y = le.fit_transform(conditions)

    # Extract features
    X = df.values

    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize a base Random Forest classifier
    clf = RandomForestClassifier(random_state=42)

    # K-Fold Cross Validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    # Define hyperparameters to tune
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    print("Starting Grid Search for hyperparameter tuning...")

    # Grid Search for hyperparameter tuning with verbose set to 3 for more detailed updates
    grid_search = GridSearchCV(clf, param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=3)
    grid_search.fit(X_train, y_train)

    print("Grid Search completed!")
    print(f"Best hyperparameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy score: {grid_search.best_score_:.4f}")

    # Evaluate the best model on the test set
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)

    # Print model performance
    print("\nModel Performance on Test Set:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, average='micro'))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # All unique classes in the dataset
    all_classes = np.unique(y)
    confusion = confusion_matrix(y_test, y_pred, labels=all_classes)

    # Plotting the confusion matrix
    all_class_names = le.inverse_transform(np.unique(y))
    plt.figure(figsize=(10,7))
    sns.heatmap(confusion, annot=True, fmt="d", cmap='Blues', xticklabels=all_class_names, yticklabels=all_class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Random Forest Confusion Matrix Heatmap with Hyper Tuning for {file}')

    # Save the heatmap as an SVG file
    file_name = f'FS_{file}_RF_confusion_matrix_HyperTunned.svg'
    plt.savefig(file_name, format='svg')
    plt.show()  # This will display the heatmap

    # Create a ZIP file with the generated SVG plot
    svg_files = [file_name]
    zip_file_name = f'FS_{file}_RF_confusion_matrix_HyperTunned.zip'

    with zipfile.ZipFile(zip_file_name, 'w') as zipf:
        for svg_file in svg_files:
            zipf.write(svg_file, arcname=svg_file)

    files.download(f'/content/{zip_file_name}')

# Process each feature-selected file and profile memory and time
for file in feature_selected_files:
    process_file(file)




#Step 8: Random Forest ML model for cellular component categories with hyper parameter tunning

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import glob

from google.colab import files

csv_files = [
    'merged_df_Cytoplasm.csv',
    'merged_df_Extracellular_Space.csv',
    'merged_df_Membrane.csv',
    'merged_df_Nucleus.csv',
    'merged_df_Others.csv'
]

svg_files = []

def process_and_train(file):
    df = pd.read_csv(file)
    df = df.drop(['Ensembl_ID', 'Uniprot_ID', 'Category'], axis=1)
    df = df.transpose()

    le = LabelEncoder()
    conditions = df.index.str.split('-').str[0].to_list()
    y = le.fit_transform(conditions)
    X = df.values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(clf, param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_

    # Predict using the best model on the test set
    y_pred = best_rf.predict(X_test)

    all_class_names = le.classes_  # Get all class names from the label encoder
    cm = confusion_matrix(y_test, y_pred, labels=le.transform(all_class_names))

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel Performance for {file}:")
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Accuracy on Test Set: {accuracy*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=all_class_names))

    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=all_class_names, yticklabels=all_class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix Heatmap for {file}')
    svg_file_name = file.split('.')[0] + '_heatmap.svg'
    plt.savefig(svg_file_name, format='svg')
    plt.show()

    svg_files.append(svg_file_name)

# Process each file and train
for file in csv_files:
    process_and_train(file)

# ZIP the SVG plots
zip_file_name = 'Component_plots_HyperTunned.zip'
with zipfile.ZipFile(zip_file_name, 'w') as zipf:
    for svg_file in svg_files:
        zipf.write(svg_file, arcname=svg_file)

files.download('/content/Component_plots_HyperTunned.zip')



#Step 9: Feature selection using PCA

!pip install kneed

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from google.colab import files
import matplotlib.pyplot as plt

# Load the scaled and merged file
df = pd.read_csv('merged_df_with_categories.csv', index_col='Ensembl_ID')

# Drop duplicate proteins based on Ensembl_ID and Uniprot_ID
df = df.drop_duplicates(subset=['Uniprot_ID'], keep='first')

# Drop Uniprot_ID and Category columns for the feature selection process
X = df.drop(columns=['Uniprot_ID', 'Category'])

# Separate features
X = X.T.values  # Transpose the dataframe to match conditions with features

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
pca.fit(X_scaled)

# Calculate explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Function to find the cutoff point
def find_cutoff(explained_variance_ratio, threshold=0.001):
    differences = np.diff(explained_variance_ratio)
    cutoff = np.where(differences < threshold)[0][0] + 1
    return min(cutoff, len(explained_variance_ratio) // 2)  # Cap at 50% of components

# Find the optimal number of components
n_components_optimal = find_cutoff(explained_variance_ratio)

# Get the feature importances based on the absolute sum of loadings for the optimal components
feature_importance = np.sum(np.abs(pca.components_[:n_components_optimal]), axis=0)
feature_importance = feature_importance / np.sum(feature_importance)

# Select features based on importance
importance_threshold = np.mean(feature_importance)  # Use mean importance as threshold
selected_features_mask = feature_importance > importance_threshold
selected_features = df.index[selected_features_mask]

# Create a new DataFrame with only the selected features
feature_selected_df = df.loc[selected_features]

# Print some information
print(f"Total number of features: {X.shape[1]}")
print(f"Number of components selected: {n_components_optimal}")
print("Number of features selected:", len(selected_features))
print("Selected features:\n", selected_features)

# Plot the explained variance ratio
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(explained_variance_ratio)+1), explained_variance_ratio, 'b-')
plt.plot(range(1, len(explained_variance_ratio)+1), np.cumsum(explained_variance_ratio), 'r-')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio and Cumulative Explained Variance vs Number of Components')
plt.axvline(x=n_components_optimal, color='g', linestyle='--', label=f'Selected components: {n_components_optimal}')
plt.legend(['Individual', 'Cumulative', 'Cutoff'])
plt.grid(True)
plt.tight_layout()

# Save the plot as SVG
svg_filename = 'pca_explained_variance_plot.svg'
plt.savefig(svg_filename, format='svg')
print(f"Explained variance plot saved as '{svg_filename}'")

# Save DataFrame to CSV
csv_filename = 'pca_feature_selected_df.csv'
feature_selected_df.to_csv(csv_filename)

print(f"CSV file '{csv_filename}' has been created with the selected features.")

# Download the new CSV file and the SVG plot
files.download(csv_filename)
files.download(svg_filename)




#Step 10: Random Forest ML model for PCA-Selected Features

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
from google.colab import files

# Load the dataframe
df = pd.read_csv('pca_feature_selected_df.csv')

# Transpose the dataframe and drop unwanted columns
df = df.transpose()
df = df.drop(['Ensembl_ID', 'Uniprot_ID', 'Category'])

# Initialize label encoder
le = LabelEncoder()

# Group conditions by removing the dash and numbers after it
conditions = df.index.str.split('-').str[0].to_list()

# Apply label encoding to the conditions
y = le.fit_transform(conditions)

# Extract features
X = df.values

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize a base Random Forest classifier
clf = RandomForestClassifier(random_state=42)

# K-Fold Cross Validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("Starting Grid Search for hyperparameter tuning...")

# Grid Search for hyperparameter tuning with verbose set to 3 for more detailed updates
grid_search = GridSearchCV(clf, param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=3)
grid_search.fit(X_train, y_train)

print("Grid Search completed!")
print(f"Best hyperparameters found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy score: {grid_search.best_score_:.4f}")

# Evaluate the best model on the test set
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# Print model performance
print("\nModel Performance on Test Set:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='micro'))
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# All unique classes in the dataset
all_classes = np.unique(y)
confusion = confusion_matrix(y_test, y_pred, labels=all_classes)

# Plotting the confusion matrix
all_class_names = le.inverse_transform(np.unique(y))
plt.figure(figsize=(10,7))
sns.heatmap(confusion, annot=True, fmt="d", cmap='Blues', xticklabels=all_class_names, yticklabels=all_class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Random Forest Confusion Matrix Heatmap with Hyper Tuning for PCA-FS')

# Save the plot as SVG file
svg_file_name = 'PCA_FS_Merged_RF_confusion_matrix_HyperTunned.svg'
plt.savefig(svg_file_name, format='svg')

# Create a ZIP file with the generated SVG plot
zip_file_name = 'PCA_FS_Merged_RF_confusion_matrix_HyperTunned.zip'

with zipfile.ZipFile(zip_file_name, 'w') as zipf:
    zipf.write(svg_file_name, arcname=svg_file_name)  # arcname parameter sets the name for the file inside the ZIP

files.download(zip_file_name)




#Step 11: Feature selection using LDA

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from google.colab import files
import matplotlib.pyplot as plt

# Load the scaled and merged file
df = pd.read_csv('merged_df_with_categories.csv', index_col='Ensembl_ID')

# Drop duplicate proteins based on Ensembl_ID and Uniprot_ID
df = df.drop_duplicates(subset=['Uniprot_ID'], keep='first')

# Drop Uniprot_ID and Category columns for the feature selection process
X = df.drop(columns=['Uniprot_ID', 'Category'])

# Initialize label encoder
le = LabelEncoder()

# Function to extract conditions from the columns
def extract_condition(column):
    return column.rsplit('-', 1)[0]  # Split based on last dash to remove repeat numbers

# Create labels from column
conditions = [extract_condition(col) for col in X.columns]
y = le.fit_transform(conditions)

# Separate features
X = X.T.values  # Transpose the dataframe to match conditions with features

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply LDA with SVD solver
n_components = min(len(np.unique(y)) - 1, X_scaled.shape[1])
lda = LinearDiscriminantAnalysis(n_components=n_components, solver='svd')
X_lda = lda.fit_transform(X_scaled, y)

# Get the feature importances based on LDA coefficients
feature_importance = np.sum(np.abs(lda.coef_), axis=0)
feature_importance = feature_importance / np.sum(feature_importance)

# Sort features by importance
sorted_idx = np.argsort(feature_importance)[::-1]
sorted_importance = feature_importance[sorted_idx]

# Calculate cumulative explained variance ratio
cumulative_importance = np.cumsum(sorted_importance)

# Find the optimal number of features (95% explained variance)
threshold = 0.95
n_features_optimal = np.where(cumulative_importance >= threshold)[0][0] + 1

print(f"Optimal number of features: {n_features_optimal}")

# Select the optimal number of features
selected_features_mask = sorted_idx[:n_features_optimal]

# Get the selected features
selected_features = df.index[selected_features_mask]

# Create a new DataFrame with only the selected features
feature_selected_df = df.loc[selected_features]

# Print some information
print("Number of features selected:", len(selected_features))
print("Cumulative explained variance ratio:", cumulative_importance[n_features_optimal-1])
print("Selected features:\n", selected_features)

# Plot the cumulative explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_importance)+1), cumulative_importance)
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio vs Number of Features')
plt.axvline(x=n_features_optimal, color='r', linestyle='--', label=f'Optimal number of features: {n_features_optimal}')
plt.axhline(y=threshold, color='g', linestyle='--', label=f'Threshold: {threshold}')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot as SVG
svg_filename = 'LDA_cumulative_variance_plot.svg'
plt.savefig(svg_filename, format='svg')
print(f"Cumulative variance plot saved as '{svg_filename}'")

# Save DataFrame to CSV
csv_filename = 'lda_feature_selected_df.csv'
feature_selected_df.to_csv(csv_filename)

print(f"CSV file '{csv_filename}' has been created with the selected features.")

# Download the new CSV file and the SVG plot
files.download(csv_filename)
files.download(svg_filename)




#Step 12: Random Forest ML model for RDA-Selected Features

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
from google.colab import files

# Load the dataframe
df = pd.read_csv('lda_feature_selected_df.csv')

# Transpose the dataframe and drop unwanted columns
df = df.transpose()
df = df.drop(['Ensembl_ID', 'Uniprot_ID', 'Category'])

# Initialize label encoder
le = LabelEncoder()

# Group conditions by removing the dash and numbers after it
conditions = df.index.str.split('-').str[0].to_list()

# Apply label encoding to the conditions
y = le.fit_transform(conditions)

# Extract features
X = df.values

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize a base Random Forest classifier
clf = RandomForestClassifier(random_state=42)

# K-Fold Cross Validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("Starting Grid Search for hyperparameter tuning...")

# Grid Search for hyperparameter tuning with verbose set to 3 for more detailed updates
grid_search = GridSearchCV(clf, param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=3)
grid_search.fit(X_train, y_train)

print("Grid Search completed!")
print(f"Best hyperparameters found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy score: {grid_search.best_score_:.4f}")

# Evaluate the best model on the test set
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# Print model performance
print("\nModel Performance on Test Set:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='micro'))
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# All unique classes in the dataset
all_classes = np.unique(y)
confusion = confusion_matrix(y_test, y_pred, labels=all_classes)

# Plotting the confusion matrix
all_class_names = le.inverse_transform(np.unique(y))
plt.figure(figsize=(10,7))
sns.heatmap(confusion, annot=True, fmt="d", cmap='Blues', xticklabels=all_class_names, yticklabels=all_class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Random Forest Confusion Matrix Heatmap with Hyper Tuning for LDA-FS')

# Save the heatmap as an SVG file
file_name = 'LDA_FS_Merged_RF_confusion_matrix_HyperTunned.svg'
plt.savefig(file_name, format='svg')
plt.show()  # This will display the heatmap

# Create a ZIP file with the generated SVG plot
svg_files = [file_name]
zip_file_name = 'LDA_FS_Merged_RF_confusion_matrix_HyperTunned.zip'

with zipfile.ZipFile(zip_file_name, 'w') as zipf:
    for file in svg_files:
        zipf.write(file, arcname=file)  # arcname parameter sets the name for the file inside the ZIP

files.download('/content/LDA_FS_Merged_RF_confusion_matrix_HyperTunned.zip')
