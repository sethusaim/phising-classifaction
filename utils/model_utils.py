from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from utils.logger import App_Logger
from utils.read_params import read_params


class Model_Utils:
    """
    Description :   This class is used for all the model utils
    Written by  :   iNeuron Intelligence
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self):
        self.log_writer = App_Logger()

        self.config = read_params()

        self.tuner_params = self.config["model_utils"]

        self.class_name = self.__class__.__name__

    def get_model_score(self, model, test_x, test_y, log_file):
        """
        Method Name :   get_model_score
        Description :   This method gets model score againist the test data

        Output      :   A model score is returned 
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """

        method_name = self.get_model_score.__name__

        self.log_writer.start_log(
            "start", log_file, self.class_name, method_name=method_name
        )

        try:
            model_name = model.__class__.__name__

            preds = model.predict(test_x)

            self.log_writer.log(
                log_file, f"Used {model_name} model to get predictions on test data",
            )

            if len(test_y.unique()) == 1:
                model_score = accuracy_score(test_y, preds)

                self.log_writer.log(
                    log_file, f"Accuracy for {model_name} is {model_score}",
                )

            else:
                model_score = roc_auc_score(test_y, preds)

                self.log_writer.log(
                    log_file, f"AUC score for {model_name} is {model_score}"
                )

            self.log_writer.start_log("exit", log_file, self.class_name, method_name)

            return model_score

        except Exception as e:
            self.log_writer.exception_log(e, log_file, self.class_name, method_name)

    def get_model_params(self, model, model_key_name, x_train, y_train, log_file):
        """
        Method Name :   get_model_params
        Description :   This method gets the model parameters based on model_key_name and train data

        Output      :   Best model parameters are returned
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """

        method_name = self.get_model_params.__name__

        self.log_writer.start_log("start",self.log_file,self.class_name,method_name)

        try:
            model_name = model.__class__.__name__

            model_param_grid = self.config[model_name]

            model_grid = GridSearchCV(model, model_param_grid, **self.tuner_params)

            self.log_writer.log(
                log_file,
                f"Initialized {model_grid.__class__.__name__}  with {model_param_grid} as params",
            )

            model_grid.fit(x_train, y_train)

            self.log_writer.log(
                log_file,
                f"Found the best params for {model_name} model based on {model_param_grid} as params",
            )

            self.log_writer.start_log("exit",self.log_file,self.class_name,method_name)

            return model_grid.best_params_

        except Exception as e:
            self.log_writer.exception_log(e,self.log_file,self.class_name,method_name)
