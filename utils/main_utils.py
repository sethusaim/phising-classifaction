import os

from phising.s3_bucket_operations.s3_operations import S3_Operation

s3 = S3_Operation()


def upload_logs(log_path, bucket):
    try:
        log_dir = os.listdir(log_path)

        for log in log_dir:
            abs_f = log_path + "/" + log

            s3.upload_file(abs_f, abs_f, bucket, "train_upload_log.txt")

        os.removedirs(log_path)

    except Exception as e:
        raise e
