from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from utils.logger import App_Logger
from utils.read_params import read_params

log_writer = App_Logger()

config = read_params()

file_name = "model_utils.py"


def get_model_name(model, db_name, collection_name):
    """
    Method Name :   get_model_name
    Description :   This method is used for getting the actual model name

    Version     :   1.0
    Revisions   :   None
    """
    method_name = get_model_name.__name__

    log_writer.start_log(
        key="start",
        class_name=file_name,
        method_name=method_name,
        db_name=db_name,
        collection_name=collection_name,
    )

    try:
        model_name = model.__class__.__name__

        log_writer.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message=f"Got the {model} model_name",
        )

        log_writer.start_log(
            key="exit",
            class_name=file_name,
            method_name=method_name,
            db_name=db_name,
            collection_name=collection_name,
        )

        return model_name

    except Exception as e:
        log_writer.raise_exception_log(
            error=e,
            file_name=file_name,
            method_name=method_name,
            db_name=db_name,
            collection_name=collection_name,
        )


def get_model_param_grid(model_key_name, db_name, collection_name):
    """
    Method Name :   get_model_param_grid
    Description :   This method is used for getting the param dict from params.yaml file

    Version     :   1.0
    Revisions   :   None
    """
    method_name = get_model_param_grid.__name__

    log_writer.start_log(
        key="start",
        class_name=file_name,
        method_name=method_name,
        db_name=db_name,
        collection_name=collection_name,
    )

    try:
        model_grid = {}

        model_param_name = config["model_params"][model_key_name]

        params_names = list(model_param_name.keys())

        for param in params_names:
            model_grid[param] = model_param_name[param]

        log_writer.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message=f"Inserted {model_key_name} params to model_grid dict",
        )

        log_writer.start_log(
            key="exit",
            class_name=file_name,
            method_name=method_name,
            db_name=db_name,
            collection_name=collection_name,
        )

        return model_grid

    except Exception as e:
        log_writer.raise_exception_log(
            error=e,
            class_name=file_name,
            method_name=method_name,
            db_name=db_name,
            collection_name=collection_name,
        )


def get_best_score_for_model(model, test_x, test_y, db_name, collection_name):
    """
    Method Name :   get_best_score_for_model
    Description :   This method is used for calculating the best score for the model based on the test data

    Version     :   1.0
    Revisions   :   None
    """
    method_name = get_best_score_for_model.__name__

    log_writer.start_log(
        key="start",
        class_name=file_name,
        method_name=method_name,
        db_name=db_name,
        collection_name=collection_name,
    )

    try:
        preds = model.predict(test_x)

        log_writer.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message=f"Used {model.__class__.__name__} model to get predictions on test data",
        )

        if len(test_y.unique()) == 1:
            model_score = accuracy_score(test_y, preds)

            log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Accuracy for {model.__class__.__name__} is {model_score}",
            )

        else:
            model_score = roc_auc_score(test_y, preds)

            log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"AUC score for {model.__class__.__name__} is {model_score}",
            )

            log_writer.start_log(
                key="exit",
                class_name=file_name,
                method_name=method_name,
                db_name=db_name,
                collection_name=collection_name,
            )

        return model_score

    except Exception as e:
        log_writer.raise_exception_log(
            error=e,
            class_name=file_name,
            method_name=method_name,
            db_name=db_name,
            collection_name=collection_name,
        )


def get_best_params_for_model(
    model, model_key_name, x_train, y_train, db_name, collection_name
):
    method_name = get_best_params_for_model.__name__

    try:
        cv = config["model_utils"]["cv"]

        verbose = config["model_utils"]["verbose"]

        model_param_grid = get_model_param_grid(
            model_key_name=model_key_name,
            db_name=db_name,
            collection_name=collection_name,
        )

        model_grid = GridSearchCV(
            estimator=model, param_grid=model_param_grid, cv=cv, verbose=verbose,
        )

        log_writer.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message=f"Initialized {model_param_grid.__class__.__name__} model with {model_param_grid} as params",
        )

        model_grid.fit(x_train, y_train)

        log_writer.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message=f"Found the best params for {model.__class__.__name__} model based on {model_param_grid} as params",
        )

        return model_grid.best_params_

    except Exception as e:
        log_writer.raise_exception_log(
            error=e,
            class_name=file_name,
            method_name=method_name,
            db_name=db_name,
            collection_name=collection_name,
        )
