


# standard library imports
from tqdm.auto import tqdm
import os

# related third-party
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# local application/library specific imports
from APP_CONFIG import Config

# define configuration proxy
working_dir = os.path.dirname(os.getcwd())
configProxy = Config(working_dir)

# get configuration
CONFIG = configProxy.return_config()

# get global constants configuration
GLOBAL_CONSTANTS = configProxy.return_global_constants()

BERT_EMBEDDING_MODEL_NAME = CONFIG['BERT_EMBEDDING_MODEL_NAME']
PREPROCESS_DEVELOPMENT_MODE = GLOBAL_CONSTANTS['PREPROCESS_DEVELOPMENT_MODE']
DISPLAY_CLUSTERS_PLOTS = GLOBAL_CONSTANTS['DISPLAY_CLUSTERS_PLOTS']
GENERATE_VALIDATION_DATASET = GLOBAL_CONSTANTS['GENERATE_VALIDATION_DATASET']
RANDOM_STATE = GLOBAL_CONSTANTS['RANDOM_STATE']
NO_CLUSTERS = GLOBAL_CONSTANTS['KMEANS_NO_CLUSTERS']


class KMeansClustering:
    def __init__(self, corpus):
        self.no_clusters = NO_CLUSTERS
        self.corpus = corpus
        self.embedder = SentenceTransformer(BERT_EMBEDDING_MODEL_NAME)
        self.corpus_embeddings = None
        self.raw_clusters = []
        self.clustered_sentences = {}

    def __encode_corpus(self):
        """
        Encodes the corpus using the provided embedder, and stores the resulting embeddings in the 'corpus_embeddings'
        attribute of the object.

        Returns:
            None
        """
        self.corpus_embeddings = self.embedder.encode(self.corpus, normalize_embeddings=True)

    def __perform_clustering(self):
        """
        Performs k-means clustering on the encoded corpus embeddings using the specified number of clusters,
        and stores the resulting cluster assignments in the 'raw_clusters' attribute of the object.

        This method uses the 'KMeans' algorithm from scikit-learn library to perform clustering on the 'corpus_embeddings'
        attribute, which contains the encoded embeddings of the corpus sentences.

        Additionally, this method processes the raw clusters to create a mapping of cluster IDs to the sentences in each cluster,
        which is stored in the 'clustered_sentences' attribute of the object.

        Returns:
            None
        """
        # Perform k-means clustering
        clustering_model = KMeans(n_clusters=self.no_clusters, random_state=RANDOM_STATE)
        self.raw_clusters = clustering_model.fit_predict(self.corpus_embeddings)
        # Process raw clusters
        for sentence_id, cluster_id in tqdm(enumerate(self.raw_clusters)):
            if cluster_id not in self.clustered_sentences:
                self.clustered_sentences[cluster_id] = []
            self.clustered_sentences[cluster_id].append(self.corpus[sentence_id])

    def display_cluster_results(self):
        """
        Displays the results of clustering by writing the cluster information to a log file.

        This method writes the cluster information to a log file, as specified by the 'LOG_TRAINING_CORPUS_CLUSTERS_PATH'
        configuration parameter, in the format:
        "Cluster <cluster_id> -> <number_of_editorials_in_cluster>"
        "<editorial_text>"
        ...

        Returns:
            None
        """
        with open(CONFIG["LOG_TRAINING_CORPUS_CLUSTERS_PATH"], 'w') as log_file:
            for i, cluster in self.clustered_sentences.items():
                log_file.write(f"Cluster {i} -> {len(cluster)}\n")
                for editorial in cluster:
                    log_file.write(f'{editorial}\n')
                log_file.write("\n")
        print(f"""\nClusters were printed at {CONFIG["LOG_TRAINING_CORPUS_CLUSTERS_PATH"]}""")

    def display_elbow_plot(self, l_clusters, r_clusters):
        """
        Displays the elbow plot for determining the optimal number of clusters.

        Args:
            l_clusters (int): The lower bound of the range of clusters to consider.
            r_clusters (int): The upper bound of the range of clusters to consider.

        Returns:
            None
        """
        # Instantiate the clustering model and visualizer
        model = KMeans(random_state=RANDOM_STATE)
        visualizer = KElbowVisualizer(model, k=(l_clusters, r_clusters))

        visualizer.fit(self.corpus_embeddings)  # Fit the data to the visualizer

        visualizer.show(outpath=CONFIG["ELBOW_PLOT_PATH"], clear_figure=True)  # Finalize and render the figure
        print(f"""\nElbow plot was saved at {CONFIG["ELBOW_PLOT_PATH"]}""")

    def display_silhouette_plot(self, l_clusters, r_clusters):
        """
        Displays the silhouette plot for evaluating cluster quality.

        Args:
            l_clusters (int): The lower bound of the range of clusters to consider.
            r_clusters (int): The upper bound of the range of clusters to consider.

        Returns:
            None
        """
        K = [i for i in range(l_clusters, r_clusters)]
        sil = []

        # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
        for k in K:
            model = KMeans(n_clusters=k, random_state=RANDOM_STATE)
            labels = model.fit_predict(self.corpus_embeddings)
            sil.append(silhouette_score(self.corpus_embeddings, labels, metric='euclidean'))

        plt.title(f"Silhouette Score for KMeans Clustering")
        plt.plot(K, sil, marker='o')
        plt.xlabel('k')
        plt.ylabel('silhouette score')
        plt.savefig(CONFIG["SILHOUETTE_PLOT_PATH"], bbox_inches='tight')
        plt.clf()
        print(f"""\nSilhouette plot was saved at {CONFIG["SILHOUETTE_PLOT_PATH"]}""")

    def display_cluster_scatter(self):
        """
        Displays a scatter plot for visualizing clustered data.

        Returns:
            None
        """
        decomp_method = TSNE(n_components=2, random_state=RANDOM_STATE)
        decomp_2d_result = decomp_method.fit_transform(self.corpus_embeddings)  # output shape: [N, 2]

        data = pd.DataFrame(decomp_2d_result, columns=['c1', 'c2'])
        data['kmeans_labels'] = self.raw_clusters
        plt.title(f"Cluster Analysis of Editorials")
        plt.scatter(x=data['c1'], y=data['c2'], c=data['kmeans_labels'], cmap='gist_rainbow')
        plt.savefig(CONFIG["CLUSTERS_SCATTER_PLOT_PATH"], bbox_inches='tight')
        plt.clf()
        print(f"""\nClusters scatter plot was saved at {CONFIG["CLUSTERS_SCATTER_PLOT_PATH"]}""")

    def fit_predict(self, display_elbow_plot=False, display_silhouette_plot=False, display_cluster_results=False,
                    display_cluster_scatter=False):
        """
        Performs k-means clustering on the embedded corpus and provides various visualization options.

        This method performs k-means clustering on the embedded corpus using the pre-trained embeddings. It provides options
        to display an elbow plot, a silhouette plot, cluster results, and cluster scatter plot for visualizing the
        clustering results.

        Args:
            display_elbow_plot (bool, optional): Whether to display an elbow plot for determining the optimal number of
                clusters. Defaults to False.
            display_silhouette_plot (bool, optional): Whether to display a silhouette plot for evaluating the quality of
                clustering. Defaults to False.
            display_cluster_results (bool, optional): Whether to display cluster results in a log file. Defaults to False.
            display_cluster_scatter (bool, optional): Whether to display a scatter plot for visualizing the clustered data.
                Defaults to False.

        Returns:
            None
        """
        self.__encode_corpus()
        if display_elbow_plot:
            self.display_elbow_plot(1, 100)
        if display_silhouette_plot:
            self.display_silhouette_plot(5, 40)
        if not PREPROCESS_DEVELOPMENT_MODE:
            self.__perform_clustering()
            if DISPLAY_CLUSTERS_PLOTS:
                if display_cluster_results:
                    self.display_cluster_results()
                if display_cluster_scatter:
                    self.display_cluster_scatter()
        return self.clustered_sentences
