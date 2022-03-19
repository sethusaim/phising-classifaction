from kneed import KneeLocator
from matplotlib import pyplot as plt
from phising.s3_bucket_operations.s3_operations import S3_Operation
from sklearn.cluster import KMeans
from utils.logger import App_Logger
from utils.read_params import read_params


class KMeans_Clustering:
    def __init__(self, log_file):
        self.s3 = S3_Operation()

        self.log_file = log_file

        self.config = read_params()

        self.input_files = self.config["s3_bucket"]["input_files"]

        self.model_bucket = self.config["s3_bucket"]["model"]

        self.kmeans_params = self.config["KMeans"]

        self.model_save_format = self.config["model_save_format"]

        self.knee_params = self.config["knee"]

        self.max_clusters = self.config["kmeans_cluster"]["max_clusters"]

        self.trained_model_dir = self.config["trained_model_dir"]

        self.elbow_plot_file = self.config["elbow_plot_fig"]

        self.log_writer = App_Logger()

        self.class_name = self.__class__.__name__

    def draw_elbow_plot(self, data):
        method_name = self.draw_elbow_plot.__name__

        self.log_writer.start_log("start", self.log_file, self.class_name, method_name)

        try:
            wcss = []

            for i in range(1, self.max_clusters):
                kmeans = KMeans(n_clusters=i, **self.kmeans_params)

                kmeans.fit(data)

                wcss.append(kmeans.inertia_)

            plt.plot(range(1, self.max_clusters), wcss)

            plt.title("The Elbow Method")

            plt.xlabel("Number of clusters")

            plt.ylabel("WCSS")

            plt.savefig(self.elbow_plot_file)

            self.log_writer.log(
                self.log_file, "Saved draw_elbow_plot fig and local copy is created",
            )

            self.s3.upload_file(
                self.elbow_plot_file,
                self.elbow_plot_file,
                self.input_files,
                self.log_file,
            )

            self.kn = KneeLocator(range(1, self.max_clusters), wcss, **self.knee_params)

            self.log_writer.log(
                self.log_file, f"The optimum number of clusters is {str(self.kn.knee)}",
            )

            self.log_writer.start_log(
                "exit", self.log_file, self.class_name, method_name
            )

            return self.kn.knee

        except Exception as e:
            self.log_writer.exception_log(
                e, self.log_file, self.class_name, method_name
            )

    def create_clusters(self, data, num_clusters):
        method_name = self.create_clusters.__name__

        self.log_writer.start_log("start", self.log_file, self.class_name, method_name)

        try:
            self.kmeans = KMeans(n_clusters=num_clusters, **self.kmeans_params)

            self.log_writer.log(
                self.log_file, f"Initialized {self.kmeans.__class__.__name__} model"
            )

            self.y_kmeans = self.kmeans.fit_predict(data)

            self.s3.save_model(
                self.kmeans,
                self.trained_model_dir,
                self.model_save_format,
                self.model_bucket,
                self.log_file,
            )

            data["Cluster"] = self.y_kmeans

            self.log_writer.log(
                self.log_file, f"Successfully created {str(self.kn.knee)} clusters",
            )

            self.log_writer.start_log(
                "exit", self.log_file, self.class_name, method_name
            )

            return data

        except Exception as e:
            self.log_writer.exception_log(
                e, self.log_file, self.class_name, method_name
            )
