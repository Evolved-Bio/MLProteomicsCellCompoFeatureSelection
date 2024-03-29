****Machine Learning and Proteomics: Leveraging Domain Knowledge for Feature Selection in a Skeletal Muscle Tissue Meta-analysis****

**Overview:**

This repository houses the code and data supporting an original research publication [insert DOI here]. Proteomics datasets from diverse skeletal muscle tissue experiments (including 2D/3D in vitro models, in vivo muscle, and adjacent tissues) retrieved from the Proteomics Identification Database (PRIDE) were combined.  This novel approach leveraged domain knowledge and cellular composition data from Uniprot and Ensembl to categorize proteins, guiding feature selection for machine learning analysis. This strategy uncovers biologically meaningful patterns that traditional methods may overlook, offering a deeper understanding of skeletal muscle tissue.

**Contents:**

<ins>Code1_id_fetching.py:</ins> Preprocesses proteomics datasets, integrates annotations from Uniprot and Ensembl, and creates a unified dataset.

<ins>Code2_cell_component_analysis.py:</ins> Employs dimensionality reduction, correlation network analysis, and other techniques to explore the combined dataset, emphasizing cellular composition categories.

<ins>Code3_basic_ML.py:</ins> Trains a Random Forest model on the proteomics data, examining how feature selection and cellular composition categorization impact performance.

<ins>Code4_testing_pretrained_model.py:</ins> Assesses the model's ability to classify new proteomics datasets by identifying similar conditions from the training set and computing similarity scores.


**Dependencies:**
Google Colab environment. Python libraries: mygene, pandas, tqdm, rpy2, networkx, matplotlib

**Contributions:**
We welcome contributions to enhance this research. Please open issues for discussions or submit pull requests for code improvements.

**Credits:**
This project aligns with Evolved.Bio's mission to advance regenerative medicine through cell sheet engineering, machine learning, and biomanufacturing. As a Canadian biotechnology startup, Evolved.Bio pioneers innovative approaches to create a world-leading tissue foundry.

**License:**
This work is published under [insert license details] as part of an open access publication [insert DOI].

**Contact:**
For questions or collaborations, please reach out to Alireza Shahin (alireza@itsevolved.com).
