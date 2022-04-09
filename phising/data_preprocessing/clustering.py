from kneed import KneeLocator
from matplotlib.pyplot import plot, savefig, title, xlabel, ylabel
from phising.s3_bucket_operations.s3_operations import S3_Operation
from sklearn.cluster import KMeans
from utils.logger import App_Logger
from utils.read_params import read_params


class KMeans_Clustering:
    """
    Description :   This class shall be used to divide the data into clusters before training.
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self, log_file):
        self.log_file = log_file

        self.config = read_params()

        self.input_files_bucket = self.config["s3_bucket"]["input_files"]

        self.model_bucket = self.config["s3_bucket"]["wafer_model"]

        self.random_state = self.config["base"]["random_state"]

        self.trained_model_dir = self.config["model_dir"]["trained"]

        self.kmeans_params = self.config["KMeans"]

        self.model_save_format = self.config["model_save_format"]

        self.knee_params = self.config["knee"]

        self.max_clusters = self.config["kmeans_cluster"]["max_clusters"]

        self.elbow_plot = self.config["elbow_plot_fig"]

        self.s3 = S3_Operation()

        self.log_writer = App_Logger()

        self.class_name = self.__class__.__name__

    def draw_elbow_plot(self, data):
        """
        Method Name :   draw_elbow_plot
        Description :   This method saves the plot to s3 bucket and decides the optimum number of clusters to the file.
        
        Output      :   An elbow plot figure saved to input files bucket
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   Moved to setup to cloud 
        """
        method_name = self.draw_elbow_plot.__name__

        self.log_writer.start_log("start", self.class_name, method_name, self.log_file)

        try:
            wcss = []

            for i in range(1, self.max_clusters):
                kmeans = KMeans(n_clusters=i, **self.kmeans_params)

                kmeans.fit(data)

                wcss.append(kmeans.inertia_)

            plot(range(1, self.max_clusters), wcss)

            title("The Elbow Method")

            xlabel("Number of clusters")

            ylabel("WCSS")

            savefig(self.elbow_plot)

            self.log_writer.log(
                "Saved elbow plot fig and local copy is created", self.log_file
            )

            self.s3.upload_file(
                self.elbow_plot,
                self.elbow_plot,
                self.input_files_bucket,
                self.log_file,
            )

            self.kn = KneeLocator(range(1, self.max_clusters), wcss, **self.knee_params)

            self.log_writer.log(
                f"The optimum number of clusters is {str(self.kn.knee)}", self.log_file
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file
            )

            return self.kn.knee

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file
            )

    def create_clusters(self, data, num_clusters):
        """
        Method Name :   create_clusters
        Description :   Create a new dataframe consisting of the cluster information.
        
        Output      :   A dataframe with cluster column
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   Moved to setup to cloud 
        """
        method_name = self.create_clusters.__name__

        self.log_writer.start_log("start", self.class_name, method_name, self.log_file)

        try:
            self.kmeans = KMeans(n_clusters=num_clusters, **self.kmeans_params)

            self.y_kmeans = self.kmeans.fit_predict(data)

            self.s3.save_model(
                self.kmeans, self.trained_model_dir, self.model_bucket, self.log_file
            )

            data["Cluster"] = self.y_kmeans

            self.log_writer.log(
                f"Successfully created {str(self.kn.knee)} clusters", self.log_file
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file
            )

            return data, self.kmeans

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file
            )
