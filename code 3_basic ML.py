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




#Step 4: Feature selection

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




#Step 5: Random Forest for Feature selected version with hyper parameter tunning

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




#Step 6: Random Forest for cellular componentcategories with hyper parameter tunning

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