Supplementary Material 1. Training data and model code

**1. Overview**

This folder provides the training data and model code used in this study. The files support model training, accuracy comparison, feature importance analysis, feature subset sensitivity analysis, and regional forest-type prediction based on AlphaEarth Foundation (AEF) embedding features.



**2. File list**

training data.csv

Training samples used in this study. The file contains harmonized sample information and AEF embedding features.

01\_model\_training\_accuracy\_comparison.py

Model training and accuracy comparison under different classification strategies.

02\_feature\_importance\_and\_subset\_sensitivity\_ET\_KNN.py

Feature importance ranking and feature subset sensitivity analysis based on the ET + KNN model combination.

03\_regional\_prediction\_ET\_KNN\_top28.py

Regional forest-type prediction using selected AEF features and the ET + KNN model combination.

README\_Supplementary\_Material\_1.txt

Description of the files included in Supplementary Material 1.



**3. Training data**

The file "training data.csv" contains 123,626 samples and 70 fields. Each record includes sample identifiers, geographic coordinates, original land-cover class information, and 64-dimensional AEF embedding features.

Main fields:

id--Sample identifier.

point\_id--Original point identifier.

lon, lat--Longitude and latitude of the sample point.

lc\_code--Original land-cover class code.

type--Original land-cover class name.

A00-A63--The 64-dimensional AlphaEarth Foundation embedding features extracted for each sample point.



**4. Code description**

01\_model\_training\_accuracy\_comparison.py

This script compares different classification strategies and machine learning models. It includes the hierarchical classification strategy, direct six-class classification strategy, and fine-class training with six-class evaluation.

02\_feature\_importance\_and\_subset\_sensitivity\_ET\_KNN.py

This script performs feature importance ranking and feature subset sensitivity analysis. The ET model is used in the forest/non-forest classification stage, and the KNN model is used in the forest-type classification stage. The script evaluates model performance under different numbers of selected AEF embedding features.

03\_regional\_prediction\_ET\_KNN\_top28.py

This script applies the selected AEF features and the ET + KNN model combination to regional forest-type prediction. It supports block-wise raster prediction and output of forest-type mapping results.



**5. Running order**

Step 1:

Run 01\_model\_training\_accuracy\_comparison.py to compare classification strategies and models.

Step 2:

Run 02\_feature\_importance\_and\_subset\_sensitivity\_ET\_KNN.py to generate feature importance rankings, feature subset sensitivity results, and the selected model bundle.

Step 3:

Run 03\_regional\_prediction\_ET\_KNN\_top28.py to generate regional forest-type mapping results.



**6. Notes**

Before running the scripts, users should modify the input and output path placeholders according to their own data storage locations.

The training data have already been prepared and include both class information and AEF embedding features. Therefore, no additional sample preprocessing script is provided in this supplementary material.

The AEF feature columns are named from A00 to A63. These columns are used as model input variables.

The scripts are provided to support reproducibility of the model training, feature analysis, and regional prediction workflow described in the manuscript.



