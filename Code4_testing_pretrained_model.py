#Step 1: Loading files
from google.colab import files
import io
import pandas as pd
import numpy as np

#Load data
uploaded = files.upload()
df1 = pd.read_csv(io.BytesIO(uploaded['merged_df_with_categories_training.csv']))
df2 = pd.read_csv(io.BytesIO(uploaded['merged_df_with_categories_testing.csv']))



#Step 2: Random forest for the combined dataset
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import joblib

# Load the dataframe
df = pd.read_csv('merged_df_with_categories_training.csv')

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
file_name = 'Merged_RF_confusion_matrix_HyperTunned_training.svg'
plt.savefig(file_name, format='svg')
plt.show()

# Create a ZIP file with the generated SVG plot
svg_files = [file_name]
zip_file_name = 'Merged_RF_confusion_matrix_HyperTunned_training.zip'

with zipfile.ZipFile(zip_file_name, 'w') as zipf:
    for file in svg_files:
        zipf.write(file, arcname=file)

files.download('/content/Merged_RF_confusion_matrix_HyperTunned_training.zip')

# Save the Model and its LabelEncoder
model_filename = 'my_trained_rf_model.sav'
joblib.dump(best_rf, model_filename)

encoder_filename = 'my_label_encoder.sav'
joblib.dump(le, encoder_filename)



#Step 3: Testing the Random Forest model with testing data

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from google.colab import files
import zipfile

# Load Second CSV and Preprocess
df_test = pd.read_csv('merged_df_with_categories_testing.csv')
df_test = df_test.drop(['Ensembl_ID', 'Uniprot_ID', 'Category'], axis=1)
df_test = df_test.transpose()
conditions_test = df_test.index.str.split('-').str[0].to_list()

# Group by condition and calculate mean for replicates
conditions_series = pd.Series(conditions_test, index=df_test.index)
df_aggregated = df_test.groupby(conditions_series).mean()

# Load LabelEncoder and model
le = joblib.load('my_label_encoder.sav')
loaded_model = joblib.load('my_trained_rf_model.sav')

# Predict probabilities for the aggregated test set
X_test_aggregated = df_aggregated.values
y_pred_prob_aggregated = loaded_model.predict_proba(X_test_aggregated)

# Identify the top two probabilities for each test sample and the corresponding conditions
top_two_indices = np.argsort(y_pred_prob_aggregated, axis=1)[:, -2:]
top_two_probs = np.take_along_axis(y_pred_prob_aggregated, top_two_indices, axis=1)

# Display two closest training conditions and similarity scores for each test condition
for i, condition in enumerate(df_aggregated.index.to_list()):
    candidates = []
    for idx in top_two_indices[i]:
        candidates.append(le.inverse_transform([idx])[0]) # Modify this line

    similarity_scores = top_two_probs[i] * 100

    print(f"Test Condition: {condition}")
    for j in range(2):
        print(f"  Candidate {j+1}: {candidates[j]}, Similarity Score: {similarity_scores[j]:.2f}%")



#Step 4: Breaking down the database
import pandas as pd
import zipfile
import numpy as np
from sklearn.preprocessing import StandardScaler
from google.colab import files

# Function to process dataset
def process_and_zip(df, dataset_name):
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

    # Write separate CSV file for each category, including dataset_name in the filename
    for category in categories:
        filename = f'{dataset_name}_{category.replace(" ", "_")}_merged_df.csv'
        df_category = df[df['Category'] == category]
        df_category.to_csv(filename, index=False)
        csv_files.append(filename)
        print(f"Category '{category}' in {dataset_name} has {len(df_category)} proteins.")

    # Create a ZipFile object
    zip_filename = f'{dataset_name}_Grouped_Proteomics_Repository.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        # Add each csv file to the zip file
        for file in csv_files:
            zipf.write(file)

    print(f'Separate CSV files have been written for the following categories in {dataset_name}: {", ".join(categories)}')
    print(f'All CSV files are also available in the zip file: {zip_filename}')

    files.download(zip_filename)

# df1 and df2 are are training and test datasets
# Process the training dataset
process_and_zip(df1, 'training')

# Process the testing dataset
process_and_zip(df2, 'testing')



#Step 5: Random Forest for cellular component categories with hyper parameter tunning
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from google.colab import files

csv_files = [
    'training_Cytoplasm_merged_df.csv',
    'training_Extracellular_Space_merged_df.csv',
    'training_Membrane_merged_df.csv',
    'training_Nucleus_merged_df.csv',
    'training_Others_merged_df.csv'
]

svg_files = []  # List to store SVG filenames
model_encoder_files = []  # List to store model and encoder filenames

def process_and_train(file):
    df = pd.read_csv(file)
    df = df.drop(['Ensembl_ID', 'Uniprot_ID', 'Category'], axis=1)
    df = df.transpose()

    le = LabelEncoder()
    conditions = df.index.str.split('-').str[0].to_list()
    y = le.fit_transform(conditions)
    X = df.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

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

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    all_class_names = le.inverse_transform(np.unique(y))

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

    model_filename = file.split('.')[0] + '_rf_model.joblib'
    encoder_filename = file.split('.')[0] + '_label_encoder.joblib'
    joblib.dump(best_rf, model_filename)
    joblib.dump(le, encoder_filename)
    return model_filename, encoder_filename

for file in csv_files:
    model_filename, encoder_filename = process_and_train(file)
    model_encoder_files.extend([model_filename, encoder_filename])

zip_file_name = 'ML_Component_plots_with_hyperparameter_tuning.zip'
with zipfile.ZipFile(zip_file_name, 'w') as zipf:
    for file in svg_files + model_encoder_files:
        zipf.write(file, arcname=file)

files.download(zip_file_name)



#Step 6: Testing the Random Forest models with testing data in each category
import pandas as pd
import numpy as np
import joblib
import zipfile
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

categories = ['Cytoplasm', 'Extracellular_Space', 'Membrane', 'Nucleus', 'Others']

for category in categories:
    # Load model, encoder, and data specific to each category
    model_filename = f'training_{category}_merged_df_rf_model.joblib'
    encoder_filename = f'training_{category}_merged_df_label_encoder.joblib'
    loaded_model = joblib.load(model_filename)
    le = joblib.load(encoder_filename)

    df_test = pd.read_csv(f'testing_{category}_merged_df.csv')
    df_test = df_test.drop(['Ensembl_ID', 'Uniprot_ID', 'Category'], axis=1)
    df_test = df_test.transpose()
    conditions_test = df_test.index.str.split('-').str[0].to_list()
    conditions_series = pd.Series(conditions_test, index=df_test.index)
    df_aggregated = df_test.groupby(conditions_series).mean()
    X_test_aggregated = df_aggregated.values

    # Predict probabilities for the aggregated test set
    y_pred_prob_aggregated = loaded_model.predict_proba(X_test_aggregated)

    # Identify the top two probabilities and corresponding conditions for each test sample
    top_two_indices = np.argsort(y_pred_prob_aggregated, axis=1)[:, -2:]
    top_two_probs = np.take_along_axis(y_pred_prob_aggregated, top_two_indices, axis=1)

    print(f"--- {category} Category Analysis ---")
    for i, condition in enumerate(df_aggregated.index.to_list()):
        candidates = le.inverse_transform(top_two_indices[i])
        similarity_scores = top_two_probs[i] * 100  # Convert to percentage

        print(f"Test Condition: {condition}")
        for j in range(2):
            print(f"  Candidate {j+1}: {candidates[j]}, Similarity Score: {similarity_scores[j]:.2f}%")
