# Tanzania Well Project

Tanzania is a country located in Eastern Africa with a population of about 59,678,000 people. Access to water remains a major challenge in Tanzania especially in its rural villages. The hope is that by predicting the functional status of wells, we can maintain the existing systems and identify the neglected as well as the development of new delivery mechanisms.

## Business Problem

The purpose of this project is to use Machine Learning Classification models to predict the functional status of water wells in Tanzania. These models include Decision Trees, Bagged Trees, Random Forest, and Gradient Boost using RandomizedSearchCV to identify the best parameters for each. The different status groups for classification are functional, non functional, and functional but needs repair.

The hope is that by gaining a better understanding of the factors that impact water pumps we are able to improve maintenance and ensure all pumps stay functional.  The importance of having accessibility of water for these Tanzanian communities is that it decreases diseases, increases education, and increases the economy overall.

## Data
The data that was used to train our models was provided by Taarifa and the Tanzanian Ministry of Water. You may also able to download the datasets [here](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/).

## Pipeline
We've created a Class to organize our modeling process and allow quick use of our chosen pipeline objects. We're able to quickly select and train models, and compare them this way. The Class also includes logging functionality to make it easier to preserve modeling information.

## Modeling

- K Nearest Neighbors
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XG Boost
- Cat Boost

## Recommendations

**Authors:** Grace Arina, Nathaniel Martin, Valeria Viscarra Fossati