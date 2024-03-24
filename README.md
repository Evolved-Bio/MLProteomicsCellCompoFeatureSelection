****Machine Learning and Proteomics: Leveraging Domain Knowledge for Feature Selection in a Skeletal Muscle Tissue Meta-analysis****

**Overview:**

This repository contains Python scripts used as part of an original research paper [insert DOI here] where proteomics datasets from different experiments with focus on skeletal muscle tissues, its 2D and 3D _in vitro_ models, in vivo muscle and its adjacent tissues including tendons, are collected from Proteomics Identification Database (PRIDE) and are combined and used for training machine learning tools. The goal is to use domain knowledge and expertise, cellular compositional data fetched from online databases such as Uniprot and Ensembl in specific, for categoriziation as an alternative to routinely used feature selection methods. This approach was shown to uncover biologically relevant patterns that traditional analysis methods could not reveal, facilitating a more wholistic understanding of the target tissue.

**Contents:**

<ins>Code1_id_fetching.py:</ins> Preprocesses the individual proteomics datasets, annotating them with information fetched from Uniprot and Ensembl Databases, and integrating them to one dataset for further analysis.

<ins>Code2_cell_component_analysis.py:</ins> Analyzes the combined dataset through dimensionality reduction, Correlation Network Analysis, and more while considering cellular composition categories.

<ins>Code3_basic_ML.py:</ins> Trains a Randfom Forest Machine Learning model using the high dimensional proteomics dataset and compares the effect of feature selection and cellular composition categorization on the performance of the model.

<ins>Code4_testing_pretrained_model.py:</ins> Evaluates the perfrmance of the Random Forest model in classifying new proteomics datasets not used in training of the model by selecting the top two most similar conditions from the training datasets and calculating similairty scores.


**Dependencies:**
The Python scripts are run on Google Colab environment and need to be ran consecutively. Installed libraries include mygene, pandas, tqdm, rpy2, networkx, and matplotlib.

**Contributions:**
Contributions are welcome. Please open an issue or submit a pull request for suggestions or improvements.

**Credits:**
This project is part of Evolved.Bio's effort in developing biofabrication methods and software stacks required for regeneraitve medicine applications. Evolved.Bio is a Canadian biotechnology startup, driving a radically new approach to regenerative therapeutics by working at the confluence of cell sheet engineering, machine learning, and biomanufacturing to build the world’s first true tissue foundry.

**License:**
This work is part of an open access publication [insert DOI] under license [insert license details].

**Contact:**
For queries or discussions, please contact Alireza Shahin (alireza@itsevolved.com).
