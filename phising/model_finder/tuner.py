from sklearn.ensemble import RandomForestClassifier
from utils.logger import App_Logger
from utils.model_utils import Model_Utils
from utils.read_params import read_params
from xgboost import XGBClassifier


class Model_Finder:
    """
    Description :   This class shall  be used to find the model with best accuracy and AUC score.
    Written by  :   iNeuron Intelligence
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self, log_file):
        self.log_file = log_file

        self.class_name = self.__class__.__name__

        self.config = read_params()

        self.cv = self.config["model_utils"]["cv"]

        self.verbose = self.config["model_utils"]["verbose"]

        self.model_utils = Model_Utils()

        self.log_writer = App_Logger()

        self.rf_model = RandomForestClassifier()

        self.xgb_model = XGBClassifier(objective="binary:logistic")

    def get_best_params_for_random_forest(self, train_x, train_y):
        """
        Method Name :   get_best_params_for_random_forest
        Description :   get the parameters for Random Forest Algorithm which give the best accuracy.
                        Use Hyper Parameter Tuning.
        
        Output      :   The model with the best parameters
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Written by  :   iNeuron Intelligence
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_best_params_for_random_forest.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            self.rf_model_name = self.model_utils.get_model_name(
                self.rf_model, self.log_file
            )

            self.rf_best_params = self.model_utils.get_model_params(
                self.rf_model,
                model_key_name="rf_model",
                x_train=train_x,
                y_train=train_y,
            )

            self.criterion = self.rf_best_params["criterion"]

            self.max_depth = self.rf_best_params["max_depth"]

            self.max_features = self.rf_best_params["max_features"]

            self.n_estimators = self.rf_best_params["n_estimators"]

            self.log_writer.log(
                log_file,
                f"{self.rf_model_name} model best params are {self.rf_best_params}",
            )

            self.rf_model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                criterion=self.criterion,
                max_depth=self.max_depth,
                max_features=self.max_features,
            )

            self.log_writer.log(
                log_file,
                f"Initialized {self.rf_model_name} with {self.rf_best_params} as params",
            )

            self.rf_model.fit(train_x, train_y)

            self.log_writer.log(
                log_file,
                f"Created {self.rf_model_name} based on the {self.rf_best_params} as params",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

            return self.rf_model

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )

    def get_best_params_for_xgboost(self, train_x, train_y):
        """
        Method Name :   get_best_params_for_xgboost
        Description :   get the parameters for XGBoost Algorithm which give the best accuracy.
                        Use Hyper Parameter Tuning.
        
        Output      :   The model with the best parameters
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Written by  :   iNeuron Intelligence
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_best_params_for_xgboost.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            self.xgb_model_name = self.model_utils.get_model_name(
                self.xgb_model, self.log_file
            )

            self.xgb_best_params = self.model_utils.get_model_params(
                self.xgb_model,
                model_key_name="xgb_model",
                x_train=train_x,
                y_train=train_y,
            )

            self.learning_rate = self.xgb_best_params["learning_rate"]

            self.max_depth = self.xgb_best_params["max_depth"]

            self.n_estimators = self.rf_best_params["n_estimators"]

            self.log_writer.log(
                log_file,
                f"{self.rf_model_name} model best params are {self.rf_best_params}",
            )

            self.xgb_model = XGBClassifier(
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
            )

            self.log_writer.log(
                log_file,
                f"Initialized {self.xgb_model_name} model with best params as {self.xgb_best_params}",
            )

            self.xgb_model.fit(train_x, train_y)

            self.log_writer.log(
                log_file,
                f"Created {self.xgb_model_name} model with best params as {self.xgb_best_params}",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

            return self.xgb_model

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )

    def get_trained_models(self, train_x, train_y, test_x, test_y):
        """
        Method Name :   get_trained_models
        Description :   Find out the Model which has the best score.
        
        Output      :   The best model name and the model object
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Written by  :   iNeuron Intelligence
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_trained_models.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name,
        )

        try:
            self.xgb_model = self.get_best_params_for_xgboost(train_x, train_y)

            self.xgb_model_score = self.model_utils.get_model_score(
                self.xgb_model, test_x=test_x, test_y=test_y,
            )

            self.rf_model = self.get_best_params_for_random_forest(train_x, train_y)

            self.rf_model_score = self.model_utils.get_model_score(
                self.rf_model, test_x=test_x, test_y=test_y,
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name,
            )

            return (
                self.xgb_model,
                self.xgb_model_score,
                self.rf_model,
                self.rf_model_score,
            )

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name,
            )
