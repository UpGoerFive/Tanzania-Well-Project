import pandas as pd
import numpy as np
import seaborn as sns
import logging
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer

#########################Valeria###########################



#########################Grace#############################



#########################Nathaniel#########################
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('model-run.log')
c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
# logger.addHandler(c_handler)
logger.addHandler(f_handler)
logger.info("Logger set up")


class Modeler:
    """
    Modeling pipeline. It has basic defaults and can accept new models and transformers.
    Models should be added in the form of:

    {'classifier': <classifier>,
     'preprocessor': <preprocessor>}

    preprocessor can be None if the default preprocessor is acceptable(Need to implement). This class also
    logs model output to a default model-run.log file. Each train or test method also has an optional print
    keyword argument that will print output if desired, as well as log it to the output file. This defaults
    to True for single runs, and False for multiple runs.

    When instantiating the class, if the prep(default preprocessor) argument is left out,
    the Modeler will generate a default one from it's input data, so X and y must be given
    in that instance.
    """
    def __init__(self, models={}, prep=None, X=pd.DataFrame(), y=pd.DataFrame(), log='model-run.log'):
        self._models=models
        self._preprocessor=prep

        for name in self._models:
            self._models[name]['output'] = None
            self._models[name]['fit_classifier'] = None
            self._models[name]['time_ran'] = None

        if not X.empty and not y.empty:
            self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(X, y, test_size=0.25, random_state = 829941045)
        else:
            self._X_train, self._X_test, self._y_train, self._y_test = None, None, None, None

        if not prep:
            self._preprocessor = self.create_default_prep(X)
            
    def create_default_prep(self, X):
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = X.select_dtypes(exclude=['int64', 'float64']).columns

        numeric_transformer = Pipeline(
            steps=[('imputer', SimpleImputer(strategy='median'))]
        )

        categorical_transformer = Pipeline(
            steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Missing'))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, num_cols),
                ("categorical", categorical_transformer, cat_cols)
            ]
        )

        return preprocessor

    def add_model(self, name, model):
        """
        Basic mechanism to add a model, model should provide classifier and preprocessor fields.
        To Do: implement default preprocessor assignment.
        """
        self._models[name] = model
        self._models[name]['output'] = None
        self._models[name]['fit_classifier'] = None
        self._models[name]['time_ran'] = None

    def change_prep(self, name, prep):
        """
        Basic reassingment of preprocessor pipeline object.
        """
        self._models[name]['preprocessor'] = prep

    def show_model(self, name):
        """
        Shows all model information.
        To Do: add printing/logging options.
        """
        print(f"{name}: {self._models[name]}")

    def train_model(self, name, X_train=pd.DataFrame(), y_train=pd.DataFrame(), print=True):
        """
        Train a single model. Fits all preprocessing transformers for later testing.
        Records and outputs cross validate scores, but also trains the classifier
        for later testing. Optional printing ability.
        """
        if print:
            logger.addHandler(c_handler)

        if X_train.empty:
            X_train = self._X_train
        if y_train.empty:
            y_train = self._y_train
        model = self._models[name]

        X_train_processed = model['preprocessor'].fit_transform(X_train)

        model['fit_classifier'] = model['classifier'].fit(X_train_processed, y_train)
        logger.info(f"{name} has been fit.")

        model['output'] = cross_val_score(
            estimator=model['classifier'],
            X=X_train_processed,
            y=y_train
        )
        logger.info(f"Cross validate scores for {name}: {model['output']}")

        if print:
            logger.removeHandler(c_handler)

    def train_all(self, X_train=pd.DataFrame(), y_train=pd.DataFrame(), print=False):
        """
        Train all available models. Fits all preprocessing transformers for later testing.
        Records and outputs cross validate scores, but also trains the classifier
        for later testing. Optional printing ability.
        """
        if print:
            logger.addHandler(c_handler)

        if X_train.empty:
            X_train = self._X_train
        if y_train.empty:
            y_train = self._y_train

        for model in self._models:
            self.train_model(model, X_train, y_train, print)

        if print:
            logger.removeHandler(c_handler)

    def test_model(self, name, X_test=pd.DataFrame(), y_test=pd.DataFrame(), print=True):
        """
        Test a single model. Uses already fitted preprocessor pipeline and classifier.
        Raises an exception if there is no fit classifier for the model. Optional printing.
        """
        if print:
            logger.addHandler(c_handler)

        if X_test.empty:
            X_test = self._X_test
        if y_test.empty:
            y_test = self._y_test
        model = self._models[name]

        X_test_processed = model['preprocessor'].transform(X_test)

        if not model['fit_classifier']: # Should add auto train fitting
            raise Exception("This model has not been fit yet.")

        model['test_output'] = model['fit_classifier'].score(X_test_processed, y_test)
        logger.info(f"{name} test score: {model['test_output']}")

        if print:
            logger.removeHandler(c_handler)

    def test_all(self, X_test=pd.DataFrame(), y_test=pd.DataFrame(), print=False):
        """
        Test all available models. Uses already fitted preprocessor pipelines and classifiers.
        Raises an exception if there is no fit classifier for a model. Optional printing.
        """
        if print:
            logger.addHandler(c_handler)

        if X_test.empty:
            X_test = self._X_test
        if y_test.empty:
            y_test = self._y_test

        for model in self._models:
            self.test_model(model, X_test, y_test, print)

        if print:
            logger.removeHandler(c_handler)

    def plot_models(self):
        """Skylar slide style."""
        pass

