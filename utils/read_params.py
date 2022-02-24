import yaml


def read_params(config_path="params.yaml"):
    """
    Method Name :   starread_paramst_log
    Description :   This method is used for read the params from yaml file

    Version     :   1.2
    Revisions   :   moved setup to cloud
    """
    method_name = read_params.__name__

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        return config

    except Exception as e:
        raise Exception(
            f"Exception occured in {__file__}, Method : {method_name}, Error : {str(e)}"
        )
