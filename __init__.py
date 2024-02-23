import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types
from fiftyone.brain import Similarity
from pprint import pprint
from fiftyone import ViewField as F


import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types
import fiftyone.core.utils as fou
import numpy as np
import fiftyone.zoo as foz

sklearn = fou.lazy_import("sklearn")
umap = fou.lazy_import("umap")




class ClusterSamples(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="cluster_samples",
            label="Cluster Samples",
            description="Cluster your samples with various algorithms!",
            icon="/assets/binoculars.svg",
            dynamic=True,

        )
    
    def resolve_input(self, ctx):
        inputs = types.Object()
    
        ready = _cluster_inputs(ctx,inputs)

        if ready:
            _execution_mode(ctx, inputs)
        

        return types.Property(inputs)
    
    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)
    
    def execute(self, ctx):
        
        _cluster(ctx)
    
        return {}
    
def _cluster_inputs(ctx, inputs):

    target_view = get_target_view(ctx, inputs)

    model_choices = ["alexnet-imagenet-torch",
                     "classification-transformer-torch",
                     "clip-vit-base32-torch",
                     "densenet121-imagenet-torch",
                     "densenet161-imagenet-torch",
                     "densenet169-imagenet-torch",
                     "densenet201-imagenet-torch",
                     "detection-transformer-torch",
                     "dinov2-vitb14-torch",
                     "dinov2-vitg14-torch",
                     "dinov2-vitl14-torch",
                     "dinov2-vits14-torch",
                     "googlenet-imagenet-torch",
                     "inception-resnet-v2-imagenet-tf1",
                     "inception-v3-imagenet-torch",
                     "inception-v4-imagenet-tf1",
                     "mobilenet-v2-imagenet-tf1",
                     "mobilenet-v2-imagenet-torch",
                     "open-clip-torch",
                     "resnet-v1-50-imagenet-tf1",
                     "resnet-v2-50-imagenet-tf1",
                     "resnet101-imagenet-torch",
                     "resnet152-imagenet-torch",
                     "resnet18-imagenet-torch",
                     "resnet34-imagenet-torch",
                     "resnet50-imagenet-torch",
                     "resnext101-32x8d-imagenet-torch",
                     "resnext50-32x4d-imagenet-torch",
                     "vgg11-bn-imagenet-torch",
                     "vgg11-imagenet-torch",
                     "vgg13-bn-imagenet-torch",
                     "vgg13-imagenet-torch",
                     "vgg16-bn-imagenet-torch",
                     "vgg16-imagenet-tf1",
                     "vgg16-imagenet-torch",
                     "vgg19-bn-imagenet-torch",
                     "vgg19-imagenet-torch",
                     "wide-resnet101-2-imagenet-torch",
                     "wide-resnet50-2-imagenet-torch",
                     "zero-shot-classification-transformer-torch",
                     "zero-shot-detection-transformer-torch"
                      ] 

    model_radio_group = types.RadioGroup()

    for choice in model_choices:
        model_radio_group.add_choice(choice, label=choice)

    inputs.enum(
        "model_radio_group",
        model_radio_group.values(),
        label="Embedding model to use",
        description="Choose what model will generate your embeddings:",
        view=types.DropdownView(),
        default='clip-vit-base32-torch',
        required=False,
        )
    
    inputs.bool(
            "force_embeddings",
            label="Force generate new embeddings?",
            description="This will overwrite any previous embeddings",
            view=types.SwitchView(),
            default=False
            )


    cluster_alg_choices = ["K-means",
                            "Affinity Propagation",
                            "Mean Shift",
                            "Agglomerative (Hierarchical)",
                            "DBSCAN" ,
                            "HDBSCAN" ,
                            "OPTICS" ,
                            "BIRCH",
                             ]


    cluster_radio_group = types.RadioGroup()

    for choice in cluster_alg_choices:
        cluster_radio_group.add_choice(choice, label=choice)

    inputs.enum(
    "cluster_radio_group",
    cluster_radio_group.values(),
    label="Cluster Algorithm",
    description="Choose what algorithm to use:",
    view=types.DropdownView(),
    default='K-means',
    )

    alg = ctx.params.get("cluster_radio_group", None)

    if alg == "K-means":
        inputs.int(
                "n_clusters",
                label="n_clusters",
                description="The number of clusters to form as well as the number of centroids to generate.",
                view=types.SliderView(componentsProps={'slider': {'min': 1, 'max': 200, "step": 1,}}),
                default=8,
        )
        inputs.int(
                "max_iter",
                label="max_iter",
                description="Maximum number of iterations of the k-means algorithm for a single run",
                view=types.SliderView(componentsProps={'slider': {'min': 1, "max":1000, "step": 1, "default": 300}}),
                default=300,
        )
        inputs.float(
                "tol",
                label="tol",
                description="Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence",
                view=types.SliderView(componentsProps={'slider': {'min': 0, "max": .001, "step": .00001, "default": 0.0001}}),
                default=0.0001,
        )
        inputs.int(
                "random_state",
                label="random_state",
                description="Determines random number generation for centroid initialization.",
                view=types.SliderView(componentsProps={'slider': {'min': 1, "max":100, "step": 1, "default": 51}}),
                default=51,
        )
        alg_choices = ["lloyd", "elkan"]

        alg_radio_group = types.RadioGroup()

        for choice in alg_choices:
            alg_radio_group.add_choice(choice, label=choice)

        inputs.enum(
            "algorithm",
            alg_radio_group.values(),
            label="algorithm",
            description="K-means algorithm to use. The classical EM-style algorithm is \"lloyd\". The \"elkan\" \
                        variation can be more efficient on some datasets with well-defined clusters, by using the \
                        triangle inequality. However it\'s more memory intensive due to the allocation of an extra array \
                        of shape (n_samples, n_clusters).",
            view=types.RadioView(),
        )
    elif alg == "Affinity Propagation":

        inputs.float(
                "damping",
                label="damping",
                description="Damping factor in the range [0.5, 1.0) is the extent to which the current value is maintained relative to incoming values (weighted 1 - damping). ",
                view=types.SliderView(componentsProps={'slider': {'min': 0.5, "max": 0.99, "step": 0.05, "default": 0.5}}),
                default=0.5,
        )
        inputs.int(
                "max_iter",
                label="max_iter",
                description="Maximum number of iterations of the k-means algorithm for a single run",
                view=types.SliderView(componentsProps={'slider': {'min': 1, "step": 1, "default": 200}}),
                default=200,
        )
        inputs.int(
                "convergence_iter",
                label="convergence_iter",
                description="Number of iterations with no change in the number of estimated clusters that stops the convergence",
                view=types.SliderView(componentsProps={'slider': {'min': 1, "step": 1, "default": 15}}),
                default=15,
        )
        inputs.float(
                "preference",
                label="preference",
                description="Preferences for each point - points with larger values of preferences are more likely to be \
                    chosen as exemplars. The number of exemplars, ie of clusters, is influenced by the input preferences value.\
                        If the preferences are not passed as arguments, they will be set to the median of the input similarities.",
                view=types.SliderView(),
                required=False
        )

        inputs.int(
                "random_state",
                label="random_state",
                description="Determines random number generation for centroid initialization.",
                view=types.SliderView(componentsProps={'slider': {'min': 1, "step": 1, "default": 51}}),
                default=51,
        )
        alg_choices = ["euclidean", "precomputed"]

        alg_radio_group = types.RadioGroup()

        for choice in alg_choices:
            alg_radio_group.add_choice(choice, label=choice)

        inputs.enum(
            "affinity",
            alg_radio_group.values(),
            label="affinity",
            description="Which affinity to use. At the moment precomputed and euclidean are supported. \
                    euclidean uses the negative squared euclidean distance between points.",
            view=types.RadioView(),
        )
    elif alg == "Mean Shift":
        inputs.float(
                "bandwidth_quantile",
                label="bandwidth_quantile",
                description="Should be between [0, 1] 0.5 means that the median of all pairwise distances is used.",
                view=types.SliderView(componentsProps={'slider': {'min': 0, "max": 1, "step": 0.1, "default": 0.3}}),
                default=0.3,
        )
        inputs.int(
                "bandwidth_n_samples",
                label="bandwidth_n_samples",
                description="The number of samples to use. If not given, all samples are used.",
                view=types.SliderView(componentsProps={'slider': {'min': 1,}}),
                required=False
        )
        inputs.bool(
                "bin_seeding",
                label="bin_seeding",
                description="If true, initial kernel locations are not locations of all points, \
                    but rather the location of the discretized version of points, where points are binned \
                    onto a grid whose coarseness corresponds to the bandwidth. Setting this option to True\
                    will speed up the algorithm because fewer seeds will be initialized.",
                view=types.CheckboxView(),
            )
        inputs.bool(
                "cluster_all",
                label="cluster_all",
                description="If true, then all points are clustered, even those orphans that are not within \
                    any kernel. Orphans are assigned to the nearest kernel. If false, then orphans are given \
                    cluster label -1.",
                view=types.CheckboxView(),
            )
        inputs.int(
                "max_iter",
                label="max_iter",
                description="Maximum number of iterations, per seed point before the clustering operation \
                    terminates (for that seed point), if has not converged yet.",
                view=types.SliderView(componentsProps={'slider': {'min': 1, "step": 1, "default": 300}}),
                default=300,
        )

        inputs.int(
                "random_state",
                label="random_state",
                description="Determines random number generation for centroid initialization.",
                view=types.SliderView(componentsProps={'slider': {'min': 1, "step": 1, "default": 51}}),
                default=51,
        )

    elif  alg == "Agglomerative (Hierarchical)":

        inputs.int(
                "n_clusters",
                label="n_cluster",
                description="The number of clusters to find. ",
                view=types.SliderView(componentsProps={'slider': {'min': 0, "step": 1, "default": 2}}),
                default=2,
        )
        alg_choices = ["ward", "complete", "single", "average"]

        alg_radio_group = types.RadioGroup()

        for choice in alg_choices:
            alg_radio_group.add_choice(choice, label=choice)

        inputs.enum(
            "linkage",
            alg_radio_group.values(),
            label="linkage",
            description="Which linkage criterion to use. The linkage criterion determines which distance \
                to use between sets of observation. The algorithm will merge the pairs of cluster that minimize \
                this criterion. \
                Ward minimizes the variance of the clusters being merged. \
                Average uses the average of the distances of each observation of the two sets. \
                Complete linkage uses the maximum distances between all observations of the two sets. \
                Single uses the minimum of the distances between all observations of the two sets.",
            view=types.RadioView(),
        )
    elif alg == "DBSCAN":

        inputs.float(
            "eps",
            label="eps",
            description="The maximum distance between two samples for one to be considered \
                as in the neighborhood of the other. This is not a maximum bound on the distances \
                of points within a cluster. This is the most important DBSCAN parameter to choose \
                appropriately for your data set and distance function.",
            view=types.SliderView(componentsProps={'slider': {'min': 0, "step": 0.1, "default": 0.5}}),
            default=.5,
        )

        inputs.int(
            "min_samples",
            label="min_samples",
            description="The number of samples (or total weight) in a neighborhood for a point to \
                be considered as a core point. This includes the point itself. If min_samples is \
                set to a higher value, DBSCAN will find denser clusters, whereas if it is set to\
                a lower value, the found clusters will be more sparse.",
            view=types.SliderView(componentsProps={'slider': {'min': 1, "step": 1, "default": 5}}),
            default=5,
        )

        alg_choices = ["auto", "ball_tree", "kd_tree", "brute"]

        alg_radio_group = types.RadioGroup()

        for choice in alg_choices:
            alg_radio_group.add_choice(choice, label=choice)

        inputs.enum(
            "algorithm",
            alg_radio_group.values(),
            label="algorithm",
            description="The algorithm to be used by the NearestNeighbors module to compute \
                pointwise distances and find nearest neighbors.",
            view=types.RadioView(),
        )
        inputs.int(
            "leaf_size",
            label="leaf_size",
            description="Leaf size passed to BallTree or cKDTree. This can affect the speed of the construction and query,\
                as well as the memory required to store the tree. The optimal value depends on the nature of the problem.",
            view=types.SliderView(componentsProps={'slider': {'min': 1, "step": 1, "default": 30}}),
        )
        inputs.float(
            "p",
            label="p",
            description="The power of the Minkowski metric to be used to calculate distance between points. If None, then p=2 (equivalent to the Euclidean distance)",
            view=types.SliderView(componentsProps={'slider': {'min': 0, "step": 1, "default": 2}}),
        )
    elif alg == "HDBSCAN":
        inputs.int(
            "min_cluster_size",
            label="min_cluster_size",
            description="The minimum number of samples in a group for that group to be considered a \
                cluster; groupings smaller than this size will be left as noise.",
            view=types.SliderView(componentsProps={'slider': {'min': 1, "step": 1, "default": 5}}),
        )

        inputs.int(
            "min_samples",
            label="min_samples",
            description="The number of samples in a neighborhood for a point to be considered as a core \
                point. This includes the point itself.",
            view=types.SliderView(componentsProps={'slider': {'min': 1, "step": 1, "default": 5}}),
            default=5,
        )

        inputs.float(
            "cluster_selection_epsilon",
            label="cluster_selection_epsilon",
            description="The number of samples in a neighborhood for a point to be considered as a core \
                point. This includes the point itself.",
            view=types.SliderView(componentsProps={'slider': {'min': 0, "step": 0.1, "default": 0.0}}),
            default=0,
        )
        inputs.int(
            "max_cluster_size",
            label="max_cluster_size",
            description="A limit to the size of clusters returned by the eom cluster selection algorithm. \
                There is no limit when max_cluster_size=None",
            view=types.SliderView(componentsProps={'slider': {'min': 1, "step": 1, "default": None}}),
            required=False
        )
        inputs.float(
            "alpha",
            label="alpha",
            description="A distance scaling parameter as used in robust single linkage.",
            view=types.SliderView(componentsProps={'slider': {'min': 0, "step": 0.1, "default": 1.0}}),
            default=1,
        )
        alg_choices = ["auto", "ball_tree", "kd_tree", "brute"]

        alg_radio_group = types.RadioGroup()

        for choice in alg_choices:
            alg_radio_group.add_choice(choice, label=choice)

        inputs.enum(
            "algorithm",
            alg_radio_group.values(),
            label="algorithm",
            description="Exactly which algorithm to use for computing core distances",
            view=types.RadioView(),
        )
        inputs.int(
            "leaf_size",
            label="alpha",
            description="Leaf size for trees responsible for fast nearest neighbour queries \
                when a KDTree or a BallTree are used as core-distance algorithms",
            view=types.SliderView(componentsProps={'slider': {'min': 1, "step": 1, "default": 40}}),
        )

    elif alg == "OPTICS":
        inputs.int(
            "min_samples",
            label="min_samples",
            description="The number of samples in a neighborhood for a point to be considered as a core point. \
                Also, up and down steep regions can't have more than min_samples consecutive non-steep points. ",
            view=types.SliderView(componentsProps={'slider': {'min': 1, "step": 1, "default": 5}}),
            default=5,
        )
        inputs.float(
            "p",
            label="p",
            description="Parameter for the Minkowski metric from pairwise_distances. When p = 1, this is equivalent \
                to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, \
                minkowski_distance (l_p) is used.",
            view=types.SliderView(componentsProps={'slider': {'min': 0, "step": 1, "default": 2}}),
            default=2,
        )

        alg_choices = ["auto", "ball_tree", "kd_tree", "brute"]

        alg_radio_group = types.RadioGroup()

        for choice in alg_choices:
            alg_radio_group.add_choice(choice, label=choice)

        inputs.enum(
            "algorithm",
            alg_radio_group.values(),
            label="algorithm",
            description="Exactly which algorithm to use for computing core distances",
            view=types.RadioView(),
        )
        inputs.int(
            "leaf_size",
            label="alpha",
            description="Leaf size for trees responsible for fast nearest neighbour queries \
                when a KDTree or a BallTree are used as core-distance algorithms",
            view=types.SliderView(componentsProps={'slider': {'min': 1, "step": 1, "default": 40}}),
        )
    elif alg == "BIRCH":

        inputs.float(
            "threshold",
            label="threshold",
            description="The radius of the subcluster obtained by merging a new sample and the \
                closest subcluster should be lesser than the threshold. Otherwise a new subcluster\
                is started. Setting this value to be very low promotes splitting and vice-versa.",
            view=types.SliderView(componentsProps={'slider': {'min': 0, "step": 0.1, "default": 0.5}}),
            default=0.5,
        )
        inputs.int(
            "branching_factor",
            label="branching_factor",
            description="Maximum number of CF subclusters in each node. If a new samples enters such \
                that the number of subclusters exceed the branching_factor then that node is split \
                into two nodes with the subclusters redistributed in each. The parent subcluster of\
                that node is removed and two new subclusters are added as parents of the 2 split nodes.",
            view=types.SliderView(componentsProps={'slider': {'min': 1, "step": 1, "default": 50}}),
            default=50,
        )
        inputs.int(
            "n_clusters",
            label="n_clusters",
            description="Number of clusters after the final clustering step, which treats the subclusters\
                from the leaves as new samples.",
            view=types.SliderView(componentsProps={'slider': {'min': 2, "step": 1, "default": 3}}),
            default=3,
        )


    inputs.str(
            "tag_name",
            label="What would like the tag to be?",
            description="Name the tag for your clusters (i.e. Cluster (X) )",
            default="Cluster"
        )


    inputs.bool(
        "filter_by_class",
        label="Filter by class?",
        description="Turn on filtering on a specific class or not.",
        view=types.SwitchView(),
        )

    by_class = ctx.params.get("filter_by_class", False)

    if by_class:
        labels = []
        field_names = list(target_view.get_field_schema().keys())
        for name in field_names:
            if type(target_view.get_field(name)) == fo.core.fields.EmbeddedDocumentField:
                if "detections" in  list(target_view.get_field(name).get_field_schema().keys()):
                    labels.append(name + ".detections")
                elif "label" in list(target_view.get_field(name).get_field_schema().keys()):
                    labels.append(name)

        if labels == []:
            inputs.view(
            "error", 
            types.Error(label="No labels found on this dataset", description="Add labels to be able to filter by them")
        )
        else:

            label_radio_group = types.RadioGroup()

            for choice in labels:
                label_radio_group.add_choice(choice, label=choice)

            inputs.enum(
                "label_radio_group",
                label_radio_group.values(),
                label="Choose Field",
                description="Choose what label field to filter on:",
                view=types.DropdownView(),
                required=True,
                default=None
                )


            field = ctx.params.get("label_radio_group")
            if field == None:
                inputs.view(
                    "warning", 
                    types.Error(label="Choose a field first!", description="Pick a label field to filter on first")
                )
            else:
                classes = target_view.distinct(field + ".label")
                class_radio_group = types.RadioGroup()

                for choice in classes:
                    class_radio_group.add_choice(choice, label=choice)

                inputs.enum(
                "class_radio_group",
                class_radio_group.values(),
                label="Choose Class",
                description="Choose what class to filter on:",
                view=types.DropdownView(),
                required=True,
                )

    return True



def _cluster(ctx):
    alg = ctx.params.get("cluster_radio_group")
    field = ctx.params.get("label_radio_group")
    by_class = ctx.params.get("filter_by_class")
    target = ctx.params.get("target", None)
    target_view = _get_target_view(ctx, target)
    tag_name = ctx.params.get("tag_name", None)
    model_choice = ctx.params.get("model_radio_group")
    force_embeddings = ctx.params.get("force_embeddings")

    

    model = foz.load_zoo_model(model_choice)

    if by_class:
        cls = ctx.params.get("class_radio_group")
        target_view = target_view.filter_labels(
            field, (F("label") == cls)
        )


    if "embeddings" not in list(target_view.get_field_schema().keys()) or force_embeddings:
        target_view.compute_embeddings(model, embeddings_field="embeddings")


    embeddings = np.array(target_view.values("embeddings"))
    mapper = umap.UMAP().fit(embeddings)
    embeddings = mapper.embedding_

    if alg == "K-means":

        from sklearn.cluster import KMeans

        n_clusters = ctx.params.get("n_clusters")
        max_iter = ctx.params.get("max_iter")
        tol = ctx.params.get("tol")
        random_state = ctx.params.get("random_state")
        alg_choice = ctx.params.get("algorithm")
        kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            algorithm=alg_choice,
        ).fit(embeddings)
        for sample, cluster in zip(target_view,kmeans.labels_):
            sample.tags.append(tag_name + " " +  str(cluster))
            sample.save()
    elif alg == "Affinity Propagation":

        from sklearn.cluster import AffinityPropagation


        damping = ctx.params.get("damping")
        max_iter = ctx.params.get("max_iter")
        convergence_iter = ctx.params.get("convergence_iter")
        preference = ctx.params.get("preference")
        random_state = ctx.params.get("random_state")
        affinity = ctx.params.get("affinity")

        af = AffinityPropagation(
            damping=damping,
            max_iter=max_iter,
            convergence_iter=convergence_iter,
            preference=preference,
            affinity=affinity,
            random_state=random_state
        ).fit(embeddings)
        for sample, cluster in zip(target_view,af.labels_):
            sample.tags.append(tag_name + " " +  str(cluster))
            sample.save()

    elif alg == "Mean Shift":

        from sklearn.cluster import  MeanShift, estimate_bandwidth

        bandwidth_quantile = ctx.params.get("bandwidth_quantile")
        bandwidth_n_samples = ctx.params.get("bandwidth_n_samples")
        bin_seeding = ctx.params.get("bin_seeding")
        cluster_all = ctx.params.get("cluster_all")
        max_iter = ctx.params.get("max_iter")
        random_state = ctx.params.get("random_state")

        bandwidth = estimate_bandwidth(embeddings, quantile=bandwidth_quantile, n_samples=bandwidth_n_samples)
        ms = MeanShift(
            bandwidth=bandwidth,
            bin_seeding=bin_seeding,
            cluster_all=cluster_all,
            max_iter=max_iter,
            random_state=random_state
            ).fit(embeddings)
        for sample, cluster in zip(target_view,ms.labels_):
            sample.tags.append(tag_name + " " +  str(cluster))
            sample.save()
        

    elif  alg == "Agglomerative (Hierarchical)":

        from sklearn.cluster import AgglomerativeClustering

        n_clusters = ctx.params.get("n_clusters")
        linkage = ctx.params.get("linkage")
        ag = AgglomerativeClustering(linkage=linkage,n_clusters=n_clusters).fit(embeddings)
        for sample, cluster in zip(target_view,ag.labels_):
            sample.tags.append(tag_name + " " + str(cluster))
            sample.save()
        
    elif alg == "DBSCAN":

        from sklearn.cluster import DBSCAN

        eps = ctx.params.get("eps")
        min_samples = ctx.params.get("min_samples")
        algorithm = ctx.params.get("algorithm")
        leaf_size = ctx.params.get("leaf_size")
        p = ctx.params.get("p")

        db = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p
        ).fit(embeddings)
        for sample, cluster in zip(target_view,db.labels_):
            sample.tags.append(tag_name + " " + str(cluster))
            sample.save()

    elif alg == "HDBSCAN":

        from sklearn.cluster import HDBSCAN

        min_cluster_size = ctx.params.get("min_cluster_size")
        min_samples = ctx.params.get("min_samples")

        cluster_selection_epsilon = ctx.params.get("cluster_selection_epsilon" )
        max_cluster_size = ctx.params.get("max_cluster_size")
        alpha = ctx.params.get("alpha")
        algorithm = ctx.params.get("algorithm")
        leaf_size = ctx.params.get("leaf_size")

        hdb = sklearn.cluster.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            max_cluster_size=max_cluster_size,
            alpha=alpha,
            algorithm=algorithm,
            leaf_size=leaf_size,
        ).fit(embeddings)
        for sample, cluster in zip(target_view,hdb.labels_):
            sample.tags.append(tag_name + " " + str(cluster))
            sample.save()


    elif alg == "OPTICS":

        from sklearn.cluster import OPTICS

        min_samples = ctx.params.get("min_samples")
        p = ctx.params.get("p")
        algorithm = ctx.params.get("algorithm")
        leaf_size = ctx.params.get("leaf_size")

        op = OPTICS(
            min_samples=min_samples,
            p=p,
            algorithm=algorithm,
            leaf_size=leaf_size
        ).fit(embeddings)
        for sample, cluster in zip(target_view,op.labels_):
            sample.tags.append(tag_name + " " + str(cluster))
            sample.save()

    elif alg == "BIRCH":

        from sklearn.cluster import Birch

        threshold = ctx.params.get("threshold")
        branching_factor = ctx.params.get("branching_factor")
        n_clusters = ctx.params.get("n_clusters")

        birch = sklearn.cluster.BIRCH(
            threshold=threshold,
            branching_factor=branching_factor,
            n_clusters=n_clusters
        ).fit(embeddings)

        for sample, cluster in zip(target_view,op.labels_):
            sample.tags.append(tag_name + " " + str(cluster))      
            sample.save()  


    

    return True




def _execution_mode(ctx, inputs):
    delegate = ctx.params.get("delegate", False)

    if delegate:
        description = "Uncheck this box to execute the operation immediately"
    else:
        description = "Check this box to delegate execution of this task"

    inputs.bool(
        "delegate",
        default=False,
        required=True,
        label="Delegate execution?",
        description=description,
        view=types.CheckboxView(),
    )

    if delegate:
        inputs.view(
            "notice",
            types.Notice(
                label=(
                    "You've chosen delegated execution. Note that you must "
                    "have a delegated operation service running in order for "
                    "this task to be processed. See "
                    "https://docs.voxel51.com/plugins/index.html#operators "
                    "for more information"
                )
            ),
        )



def get_target_view(ctx, inputs):
    has_view = ctx.view != ctx.dataset.view()
    has_selected = bool(ctx.selected)
    default_target = None

    if has_view or has_selected:
        target_choices = types.RadioGroup(orientation="horizontal")
        target_choices.add_choice(
            "DATASET",
            label="Entire dataset",
            description="Process the entire dataset",
        )

        if has_view:
            target_choices.add_choice(
                "CURRENT_VIEW",
                label="Current view",
                description="Process the current view",
            )
            default_target = "CURRENT_VIEW"

        if has_selected:
            target_choices.add_choice(
                "SELECTED_SAMPLES",
                label="Selected samples",
                description="Process only the selected samples",
            )
            default_target = "SELECTED_SAMPLES"

        inputs.enum(
            "target",
            target_choices.values(),
            default=default_target,
            required=True,
            label="Target view",
            view=target_choices,
        )

    target = ctx.params.get("target", default_target)

    return _get_target_view(ctx, target)

def _get_target_view(ctx, target):
    if target == "SELECTED_SAMPLES":
        return ctx.view.select(ctx.selected)

    if target == "DATASET":
        return ctx.dataset

    return ctx.view

def register(plugin):
    plugin.register(ClusterSamples)
