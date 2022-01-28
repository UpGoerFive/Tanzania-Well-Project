# Tanzania Well Project

Tanzania is a country located in Eastern Africa with a population of about 59,678,000 people. Access to water remains a major challenge in Tanzania especially in its rural villages. The hope is that by predicting the functional status of wells, we can maintain the existing systems and identify the neglected as well as the development of new delivery mechanisms.

## Business Problem

The purpose of this project is to use Machine Learning Classification models to predict the functional status of water wells in Tanzania. These models include Decision Trees, Random Forest, and Gradient Boost using RandomizedSearchCV to identify the best parameters for each. The different status groups for classification are functional, non functional, and functional but needs repair.

The hope is that by gaining a better understanding of the factors that impact water wells, we are able to improve maintenance and ensure all wells stay functional.  The importance of having accessibility of water for these Tanzanian communities is that it decreases diseases, increases education, and increases the economy overall.

## Data
The data that was used to train our models was provided by Taarifa and the Tanzanian Ministry of Water. You may also able to download the datasets [here](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/). This is a large dataset of around fifty thousand entries, with forty features to use in prediction.

## Pipeline
We've created a Class to organize our modeling process and allow quick use of our chosen pipeline objects. This class performs our train test split, to make sure all models are consistent with each other, and also has an optional default preprocessor to make it easier to keep track of how data is being manipulated in each model pipeline. We also have an evaluation report method and permutation importance method to assess our best performing models. We're able to quickly select and train models, and compare them this way. The Class also includes logging functionality to make it easier to preserve modeling information.

## Modeling

- K Nearest Neighbors
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Cat Boost

We evaluated the accuracy of these models in predicting our target. Ultimately our best performing model was a tuned Gradient Boost with an accuracy of around 81%. This was achieved through hyper parameter tuning with RandomizedSearchCV in SciKit-Learn.

## Recommendations

The mean feature importance averages of our model are given here:

We're recommending a focus on waterpoint_type, extraction_type, and payment_type as these are both important for prediction, and also controllable by the Government of Tanzania. Specifically we recommend:

- Building primarily single communal standpipes
- point 2
- point 3

These have the highest percentage of functional wells out of the available options for each feature.

## Repository Structure


* [data/](./Tanzania-Well-Project/data)
  * [Joined_values_labels.csv](./Tanzania-Well-Project/data/Joined_values_labels.csv)
  * [SubmissionFormat.csv](./Tanzania-Well-Project/data/SubmissionFormat.csv)
  * [Test-set-values.csv](./Tanzania-Well-Project/data/Test-set-values.csv)
  * [Training-set-labels.csv](./Tanzania-Well-Project/data/Training-set-labels.csv)
  * [Training-set-values.csv](./Tanzania-Well-Project/data/Training-set-values.csv)
* [Baseline Models.ipynb](./Tanzania-Well-Project/Baseline Models.ipynb)
* [EDA.ipynb](./Tanzania-Well-Project/EDA.ipynb)
* [Join Labels & Values.ipynb](./Tanzania-Well-Project/Join Labels & Values.ipynb)
* [PlottingNotebook.ipynb](./Tanzania-Well-Project/PlottingNotebook.ipynb)
* [README.md](./Tanzania-Well-Project/README.md)
* [REVIEW_BaselineModels.ipynb](./Tanzania-Well-Project/REVIEW_BaselineModels.ipynb)
* [ourfunctions.py](./Tanzania-Well-Project/ourfunctions.py)


**Authors:** Grace Arina, Nathaniel Martin, Valeria Viscarra Fossati