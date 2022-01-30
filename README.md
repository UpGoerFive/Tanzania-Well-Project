# Tanzania Well Project

Tanzania is located in Eastern Africa with a population of about 59,678,000 people. Access to water remains a major challenge in Tanzania especially in its rural villages. The hope is that by predicting the functional status of wells, Tanzania can maintain the existing systems and identify the neglected wells as well as the development of new water delivery mechanisms.

## Business Problem

The purpose of this project is to use Machine Learning Classification models to predict the functional status of water wells in Tanzania. These models include Decision Trees, Random Forest, and Gradient Boost using RandomizedSearchCV to identify the best parameters for each. The different status groups for classification are functional, non functional, and functional but needs repair.

The hope is that by gaining a better understanding of the factors that impact water wells, we are able to improve maintenance and ensure all wells stay functional.  The importance of having accessibility of water for these Tanzanian communities is that it decreases diseases, increases education, and increases the economy overall.

## Data
The data that was used to train our models was provided by Taarifa and the Tanzanian Ministry of Water.
The data sets are as follow:
- Training-set-values: 59,400 observations, 40 variables
- Training-set-labels: 59,400 observations; contains status group labels

We perform a train test split on this set in order to properly test our models on previously unseen data.

- Test-set: 14,850 observations, 40 variables

This is provided by the competition website to generate predictions on, and rank the outputs.

You can download the datasets [here](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/).

Additionaly, we used SQL to join the Training-set-values and Training-set-labels to make the Joined_values_labels csv to make plotting easier on Tableau.

## Pipeline
We created a Class to organize our modeling process and allow quick use of our chosen pipeline objects. This class performs our train test split, to make sure all models are consistent with each other, and also has an optional default preprocessor to make it easier to keep track of how data is being manipulated in each model pipeline. We also have an evaluation report method and permutation importance method to assess our best performing models. We're able to quickly select and train models, and compare them this way. The Class also includes logging functionality to make it easier to preserve modeling information.

## Modeling

- K Nearest Neighbors
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Cat Boost

We evaluated the accuracy of these models in predicting our target. Ultimately our best performing model was a tuned Gradient Boost with an accuracy of around 81%. This was achieved through hyper parameter tuning with RandomizedSearchCV in SciKit-Learn.

## Recommendations

We created based on the mean permutation feature importances of our best model. We're recommending a focus on waterpoint_type, extraction_type, and payment_type as these are both important for prediction, and also controllable by the Government of Tanzania. Specifically we recommend:

- Building and maintaining primarily single communal standpipes
- Building and maintaining primarily gravity wells
- Charging annually or monthly

These have the highest percentage of functional wells out of the available options for each feature.

## Repository Structure

* [data/](./Tanzania-Well-Project/data)
  * [Joined_values_labels.csv](./Tanzania-Well-Project/data/Joined_values_labels.csv)
  * [SubmissionFormat.csv](./Tanzania-Well-Project/data/SubmissionFormat.csv)
  * [Test-set-values.csv](./Tanzania-Well-Project/data/Test-set-values.csv)
  * [Training-set-labels.csv](./Tanzania-Well-Project/data/Training-set-labels.csv)
  * [Training-set-values.csv](./Tanzania-Well-Project/data/Training-set-values.csv)
* [presentation-items/](./Tanzania-Well-Project/presentation-items)
  * [Inferential Plots.twb](<./Tanzania-Well-Project/presentation-items/Inferential Plots.twb>)
  * [Model-Performance.png](./Tanzania-Well-Project/presentation-items/Model-Performance.png)
  * [Tanzania Well Project.pdf](<./Tanzania-Well-Project/presentation-items/Tanzania Well Project.pdf>)
* [.gitignore](./Tanzania-Well-Project/.gitignore)
* [BaselineModels.ipynb](./Tanzania-Well-Project/BaselineModels.ipynb)        ~Our Non-Boosted Models
* [BoostModels.ipynb](./Tanzania-Well-Project/BoostModels.ipynb)        ~Our Boosted Models
* [EDA.ipynb](./Tanzania-Well-Project/EDA.ipynb)
* [ModelClass.py](./Tanzania-Well-Project/ModelClass.py)
* [PlottingNotebook.ipynb](./Tanzania-Well-Project/PlottingNotebook.ipynb)        ~Just for plotting from both model notebooks
* [README.md](./Tanzania-Well-Project/README.md)


**Authors:** Grace Arina, Nathaniel Martin, Valeria Viscarra Fossati
