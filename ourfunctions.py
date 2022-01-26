import pandas as pd
import numpy as np
import seaborn as sns
import logging
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
# Current version of sklearn is still too old I think, might try to upgrade to use below option
# from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV# ,  HalvingRandomSearchCV
from sklearn.compose import ColumnTransformer, make_column_selector

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

    preprocessor can be None if the default preprocessor is acceptable. This class also
    logs model output to a default model-run.log file. Each train or test method also has an optional print
    keyword argument that will print output if desired, as well as log it to the output file. This defaults
    to True for single runs, and False for multiple runs.
    """
    def __init__(self, models={}, X=pd.DataFrame(), y=pd.DataFrame()):
        self._models=models
        self._tuning = {}

        for name in self._models:
            if not self._models[name]['preprocessor']:
                self._models[name]['preprocessor'] = self.create_default_prep()
            self._models[name]['output'] = None
            self._models[name]['fit_classifier'] = None

        if not X.empty and not y.empty:
            self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(X, y, test_size=0.25, random_state = 829941045)
        else:
            self._X_train, self._X_test, self._y_train, self._y_test = None, None, None, None

    def create_default_prep(self, cat_add=None, num_add=None):
        """
        Creates a default preprocessing object, uses all columns and imputes with median for numeric and 'missing' for categorical.
        Can accept extra steps with cat_add and num_add, which must be lists of tuples (steps). Currently only adds them to the order.
        """
        def to_object(x):
            return pd.DataFrame(x).astype(str)

        string_transformer = FunctionTransformer(to_object)

        if num_add:
            numeric_transformer = Pipeline(
                steps=[('imputer', SimpleImputer(strategy='median')),
                        *num_add]
            )
        else:
            numeric_transformer = Pipeline(
                steps=[('imputer', SimpleImputer(strategy='median'))]
            )

        if cat_add:
            categorical_transformer = Pipeline(
                steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                    ('casting', string_transformer),
                    ('one_hot_encode', OneHotEncoder(handle_unknown='ignore')),
                    *cat_add]
            )
        else:
            categorical_transformer = Pipeline(
                steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                    ('casting', string_transformer),
                    ('one_hot_encode', OneHotEncoder(handle_unknown='ignore'))]
            )

        preprocessor = ColumnTransformer(
                            transformers=[
                                ("numeric", numeric_transformer, make_column_selector(dtype_include=np.number)),
                                ("categorical", categorical_transformer, make_column_selector(dtype_exclude=np.number))
                            ]
                        )

        return preprocessor

    def add_model(self, name, model):
        """
        Basic mechanism to add a model, model should provide classifier and preprocessor fields.
        Model can have None as the preprocessor, in which case a default will be provided, however
        the 'preprocessor' key must still be provided.
        """
        if not model['preprocessor']:
                model['preprocessor'] = self.create_default_prep()

        self._models[name] = model
        self._models[name]['output'] = None
        self._models[name]['fit_classifier'] = None

    def remove_model(self, name):
        """
        Files your taxes.
        """
        del self._models[name]

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

    def get_model(self, name):
        """
        Access a model to use.
        """
        return self._models[name]

    def train_model(self, name, X_train=pd.DataFrame(), y_train=pd.DataFrame(), print=True, cv=True, train=True):
        """
        Train a single model. Fits all preprocessing transformers for later testing.
        Records and outputs cross validate scores. The cv_only option determines if the method will 
        fit a classifier, which is required before testing. Optional printing ability.
        """
        if print:
            logger.addHandler(c_handler)

        if X_train.empty:
            X_train = self._X_train
        if y_train.empty:
            y_train = np.array(self._y_train).ravel()
        model = self._models[name]

        X_train_processed = model['preprocessor'].fit_transform(X_train)

        if train:
            model['fit_classifier'] = model['classifier'].fit(X_train_processed, y_train)
            logger.info(f"{name} has been fit.")
            self._models[name]['time_fit'] = time.asctime()

        if cv:
            model['output'] = cross_val_score(
                estimator=model['classifier'],
                X=X_train_processed,
                y=y_train
            )
            logger.info(f"Cross validate scores for {name}: {model['output']}")
            self._models[name]['time_cross_val'] = time.asctime()

        if print:
            logger.removeHandler(c_handler)

    def train_all(self, X_train=pd.DataFrame(), y_train=pd.DataFrame(), print=False, cv=True, train=True):
        """
        Train all available models. Fits all preprocessing transformers for later testing.
        Records and outputs cross validate scores. The cv_only option determines if the method will 
        fit a classifier, which is required before testing. Optional printing ability.
        """

        if X_train.empty:
            X_train = self._X_train
        if y_train.empty:
            y_train = self._y_train

        for model in self._models:
            self.train_model(model, X_train, y_train, print, cv, train)

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
            y_test = np.array(self._y_test).ravel()
        model = self._models[name]

        X_test_processed = model['preprocessor'].transform(X_test)

        if not model['fit_classifier']: # Should add auto train fitting
            raise Exception("This model has not been fit yet.")

        model['test_output'] = model['fit_classifier'].score(X_test_processed, y_test)
        self._models[name]['time_tested'] = time.asctime()
        logger.info(f"{name} test score: {model['test_output']}")

        if print:
            logger.removeHandler(c_handler)

    def test_all(self, X_test=pd.DataFrame(), y_test=pd.DataFrame(), print=False):
        """
        Test all available models. Uses already fitted preprocessor pipelines and classifiers.
        Raises an exception if there is no fit classifier for a model. Optional printing.
        """

        if X_test.empty:
            X_test = self._X_test
        if y_test.empty:
            y_test = self._y_test

        for model in self._models:
            self.test_model(model, X_test, y_test, print)

    def hyper_search(self, name, searcher=RandomizedSearchCV, params=None, searcher_kwargs=None, print=False, set_to_train=False):
        """
        Hyper parameter tuning function, defaults to RandomizedSearchCV, but any search function
        you want can be passed in. searcher_kwargs should be a dictionary of the keyword argument you want to pass
        to the search object:

            searcher_kwargs = {'n_jobs': 3, 'refit': True, 'cv': 10}

        The keys need to be the exact arguments of the object. Note that this should not include things like 
        param_distributions, as this should be filled in the params argument.
        """
        if print:
            logger.addHandler(c_handler)

        if not params and 'param_distro' in self._models[name].keys():
            params = self._models[name]['param_distro']
        elif params:
            self._models[name]['param_distro'] = params

        if searcher_kwargs:
            search_object = searcher(self._models[name]['classifier'], params, **searcher_kwargs)
        else:
            search_object = searcher(self._models[name]['classifier'], params)
        
        X_train_processed = self._models[name]['preprocessor'].fit_transform(self._X_train)
        search_object.fit(X_train_processed, np.array(self._y_train).ravel())
        logger.info(f"For model {name}, {searcher} with{params} produced:")
        logger.info(f"Params: {search_object.best_params_}")
        logger.info(f"{search_object.best_score_}" if 'refit' not in searcher_kwargs.keys() else "refit = False")

        self._models[name]['search_classifier'] = search_object.best_estimator_ if 'refit' not in searcher_kwargs.keys() else None
        self._models[name]['search_best_params'] = search_object.best_params_
        self._models[name]['search_performed_at'] = time.asctime()

        if set_to_train:
            self._models[name]['fit_classifier'] = search_object.best_estimator_
            self._models[name]['time_fit'] = time.asctime()

        if print:
            logger.removeHandler(c_handler)


    def plot_models(self, sns_style='darkgrid', sns_context='talk', palette='coolwarm', save=None):
        """
        Skylar slide style, with thanks to Matt. Has options for seaborn plotting. If you want to save the plot,
        give the save option a filename, exactly as would be done with plt.savefig()
        """
        logger.removeHandler(c_handler)
        logger.removeHandler(f_handler)

        xticklabels = list(self._models.keys())
        y = [model['test_output'] for model in self._models.values()]

        sns.set_style(sns_style)
        sns.set_context(sns_context)
        fig, ax = plt.subplots(figsize=(20, 10))

        fig.set_tight_layout(True)

        sns.barplot(x=xticklabels, y=y, palette=palette)
        ax.set(ylim=(0, 1))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

        # ax2 = ax.twinx()
        # sns.lineplot(x=xticklabels, y=x_error, linewidth=5)
        # ax2.set(ylim=(0, 300000))
        # ax2.set_yticks(np.linspace(0,300000,num=6))
        # ax2.set_yticklabels(np.linspace(0,300,num=6,dtype=int))

        ax.set_ylabel('Accuracy Score')
        # ax2.set_ylabel('Error, USD (thousands)')
        ax.set_title('Model Effectiveness');

        if save:
            plt.savefig(save)
        
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

