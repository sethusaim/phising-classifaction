from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from utils.logger import app_logger
from utils.read_params import read_params

log_writer = app_logger()

config = read_params()


def get_model_name(model, table_name):
    """
    Method Name :   get_model_name
    Description :   This method is used for getting the actual model name

    Version     :   1.0
    Revisions   :   None
    """
    method_name = get_model_name.__name__

    log_writer.start_log(
        key="start",
        class_name=__file__,
        method_name=method_name,
        table_name=table_name,
    )

    try:
        model_name = model.__class__.__name__

        log_writer.log(
            table_name=table_name, log_message=f"Got the {model} model_name",
        )

        log_writer.start_log(
            key="exit",
            class_name=__file__,
            method_name=method_name,
            table_name=table_name,
        )

        return model_name

    except Exception as e:
        log_writer.exception_log(
            error=e,
            class_name=__file__,
            method_name=method_name,
            table_name=table_name,
        )


def get_model_param_grid(model_key_name, table_name):
    """
    Method Name :   get_model_param_grid
    Description :   This method is used for getting the param dict from params.yaml file

    Version     :   1.2
    Revisions   :   moved setup to cloud
    """
    method_name = get_model_param_grid.__name__

    log_writer.start_log(
        key="start",
        class_name=__file__,
        method_name=method_name,
        table_name=table_name,
    )

    try:
        model_grid = {}

        model_param_name = config["model_params"][model_key_name]

        params_names = list(model_param_name.keys())

        for param in params_names:
            model_grid[param] = model_param_name[param]

        log_writer.log(
            table_name=table_name,
            log_message=f"Inserted {model_key_name} params to model_grid dict",
        )

        log_writer.start_log(
            key="exit",
            class_name=__file__,
            method_name=method_name,
            table_name=table_name,
        )

        return model_grid

    except Exception as e:
        log_writer.exception_log(
            error=e,
            class_name=__file__,
            method_name=method_name,
            table_name=table_name,
        )


def get_model_score(model, test_x, test_y, table_name):
    """
    Method Name :   get_model_score
    Description :   This method is used for calculating the best score for the model based on the test data

    Version     :   1.2
    Revisions   :   moved setup to cloud
    """
    method_name = get_model_score.__name__

    log_writer.start_log(
        key="start",
        class_name=__file__,
        method_name=method_name,
        table_name=table_name,
    )

    try:
        model_name = get_model_name(model=model, table_name=table_name)

        preds = model.predict(test_x)

        log_writer.log(
            table_name=table_name,
            log_message=f"Used {model_name} model to get predictions on test data",
        )

        if len(test_y.unique()) == 1:
            model_score = accuracy_score(test_y, preds)

            log_writer.log(
                table_name=table_name,
                log_message=f"Accuracy for {model_name} is {model_score}",
            )

        else:
            model_score = roc_auc_score(test_y, preds)

            log_writer.log(
                table_name=table_name,
                log_message=f"AUC score for {model_name} is {model_score}",
            )

            log_writer.start_log(
                key="exit",
                class_name=__file__,
                method_name=method_name,
                table_name=table_name,
            )

        return model_score

    except Exception as e:
        log_writer.exception_log(
            error=e,
            class_name=__file__,
            method_name=method_name,
            table_name=table_name,
        )


def get_model_params(model, model_key_name, x_train, y_train, table_name):
    """
    Method Name :   get_model_params
    Description :   This method is used for finding the best params for the given model

    Version     :   1.2
    Revisions   :   moved setup to cloud
    """
    method_name = get_model_params.__name__

    log_writer.start_log(
        key="start",
        class_name=__file__,
        method_name=method_name,
        table_name=table_name,
    )

    try:
        cv = config["model_utils"]["cv"]

        verbose = config["model_utils"]["verbose"]

        n_jobs = config["model_utils"]["n_jobs"]

        model_name = get_model_name(model=model, table_name=table_name)

        model_param_grid = get_model_param_grid(
            model_key_name=model_key_name, table_name=table_name
        )

        model_grid = GridSearchCV(
            estimator=model,
            param_grid=model_param_grid,
            cv=cv,
            verbose=verbose,
            n_jobs=n_jobs,
        )

        log_writer.log(
            table_name=table_name,
            log_message=f"Initialized {model_grid.__class__.__name__}  with {model_param_grid} as params",
        )

        model_grid.fit(x_train, y_train)

        log_writer.log(
            table_name=table_name,
            log_message=f"Found the best params for {model_name} model based on {model_param_grid} as params",
        )

        log_writer.start_log(
            key="exit",
            class_name=__file__,
            method_name=method_name,
            table_name=table_name,
        )

        return model_grid.best_params_

    except Exception as e:
        log_writer.exception_log(
            error=e,
            class_name=__file__,
            method_name=method_name,
            table_name=table_name,
        )
