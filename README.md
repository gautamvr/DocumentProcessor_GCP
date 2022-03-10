# DocumentProcessor_GCP

## Overview

A complete pipeline for Document AI process which is fully automated from training to deployment using Google Cloud Platform. It's trained on multiple machine learning models then deployed together alongside processing steps.

The  end-to-end document AI pipeline consists of two components:

1. A training pipeline which formats the training data and uses AutoML to build Image Classification, Entity Extraction, Text Classification, and Object Detection models.

2. A prediction pipeline which takes PDF documents from a specified Cloud Storage bucket, uses the AutoML models to extract the relevant data from the documents, and stores the extracted data in BigQuery for further analysis.

### 1. Training pipeline

- Training data is pulled from the BigQuery public dataset. The training BigQuery table includes links to PDF files in Cloud Storage of patents from the United States and European Union.

- The PDF files are converted to PNG files and uploaded to a new Cloud Storage bucket in your own project. The PNG files will be used to train the AutoML Vision models.

- The PNG files are run through the Cloud Vision API to create TXT files containing the raw text from the converted PDFs. These TXT files are used to train the AutoML Natural Language models.

- The links to the PNG or TXT files are combined with the labels and features from the BigQuery table into a CSV file in the training data format required by AutoML. This CSV is then uploaded to a Cloud Storage bucket. Note: This format is different for each type of AutoML model.

- This CSV is used to create an AutoML dataset and model. Both are named in the format patent_demo_data_%m%d%Y_%H%M%S. Some AutoML models can sometimes take hours to train.

### 2. Prediction pipeline

- This pipeline uses the AutoML models previously trained by the pipeline above. For predictions, the following steps occur:

- The patent PDFs are collected from the prescribed bucket and converted to PNG and TXT files with the Cloud Vision API.

- The AutoML Image Classification model is called on the PNG files to classify each patent as either a US or EU patent. The results are uploaded to a BigQuery table.

- The AutoML Object Detection model is called on the PNG files to determine the location of any figures on the patent document. The resulting relative x, y coordinates of the bounding box are then uploaded to a BigQuery table.

- The AutoML Text Classification model is called on the TXT files to classify the topic of the patent content as medical technology, computer vision, cryptocurrency or other. The results are then uploaded to a BigQuery table.

- The AutoML Entity Extraction model is called to extract predetermined entities from the patent. The extracted entities are applicant, application number, international classification, filing date, inventor, number, publication date and title. These entities are then uploaded to a BigQuery table.

- Finally, the BigQuery tables above are joined to produce a final results table with all the properties above.
