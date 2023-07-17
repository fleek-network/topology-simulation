from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_array, check_random_state
from sklearn.metrics.pairwise import (
    pairwise_distances,
    pairwise_distances_argmin,
)
from sklearn.utils.extmath import cartesian
from sklearn.utils.validation import check_is_fitted
import warnings
import numpy as np
from ortools.graph.python.min_cost_flow import SimpleMinCostFlow


def _compute_inertia(distances):
    """Compute inertia of new samples. Inertia is defined as the sum of the
    sample distances to closest cluster centers.

    Parameters
    ----------
    distances : {array-like, sparse matrix}, shape=(n_samples, n_clusters)
        Distances to cluster centers.

    Returns
    -------
    Sum of sample distances to closest cluster centers.
    """

    # Define inertia as the sum of the sample-distances
    # to closest cluster centers
    inertia = np.sum(np.min(distances, axis=1))

    return inertia


def minimum_cost_flow_problem_graph(num_nodes, num_medoids, D, size_min, size_max):
    # Setup minimum cost flow formulation graph
    # Vertices indexes:
    # X-nodes: [0, n(x)-1], C-nodes: [n(X), n(X)+n(C)-1], C-dummy nodes:[n(X)+n(C), n(X)+2*n(C)-1],
    # Artificial node: [n(X)+2*n(C), n(X)+2*n(C)+1-1]

    # Create indices of nodes
    n_X = num_nodes
    n_C = num_medoids
    X_ix = np.arange(n_X)
    C_dummy_ix = np.arange(X_ix[-1] + 1, X_ix[-1] + 1 + n_C)
    C_ix = np.arange(C_dummy_ix[-1] + 1, C_dummy_ix[-1] + 1 + n_C)
    art_ix = C_ix[-1] + 1

    # Edges
    edges_X_C_dummy = cartesian([X_ix, C_dummy_ix])  # All X's connect to all C dummy nodes (C')
    edges_C_dummy_C = np.stack([C_dummy_ix, C_ix], axis=1)  # Each C' connects to a corresponding C (centroid)
    edges_C_art = np.stack([C_ix, art_ix * np.ones(n_C)], axis=1)  # All C connect to artificial node

    edges = np.concatenate([edges_X_C_dummy, edges_C_dummy_C, edges_C_art])

    # Costs
    costs_X_C_dummy = D.reshape(D.size)
    costs = np.concatenate([costs_X_C_dummy, np.zeros(edges.shape[0] - len(costs_X_C_dummy))])

    # Capacities - can set for max-k
    capacities_C_dummy_C = size_max * np.ones(n_C)
    cap_non = n_X  # The total supply and therefore wont restrict flow
    capacities = np.concatenate([
        np.ones(edges_X_C_dummy.shape[0]),
        capacities_C_dummy_C,
        cap_non * np.ones(n_C)
    ])

    # Sources and sinks
    supplies_X = np.ones(n_X)
    supplies_C = -1 * size_min * np.ones(n_C)  # Demand node
    supplies_art = -1 * (n_X - n_C * size_min)  # Demand node
    supplies = np.concatenate([
        supplies_X,
        np.zeros(n_C),  # C_dummies
        supplies_C,
        [supplies_art]
    ])

    # All arrays must be of int dtype for `SimpleMinCostFlow`
    edges = edges.astype('int32')
    costs = np.around(costs * 1000, 0).astype('int32')  # Times by 1000 to give extra precision
    capacities = capacities.astype('int32')
    supplies = supplies.astype('int32')

    return edges, costs, capacities, supplies, n_C, n_X


def solve_min_cost_flow_graph(edges, costs, capacities, supplies, n_C, n_X):
    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = SimpleMinCostFlow()

    if (edges.dtype != 'int32') or (costs.dtype != 'int32') \
            or (capacities.dtype != 'int32') or (supplies.dtype != 'int32'):
        raise ValueError("`edges`, `costs`, `capacities`, `supplies` must all be int dtype")

    N_edges = edges.shape[0]
    N_nodes = len(supplies)

    # Add each edge with associated capacities and cost
    min_cost_flow.add_arcs_with_capacity_and_unit_cost(edges[:, 0], edges[:, 1], capacities, costs)

    # Add node supplies
    for count, supply in enumerate(supplies):
        min_cost_flow.set_node_supply(count, supply)

    # Find the minimum cost flow between node 0 and node 4.
    if min_cost_flow.solve() != min_cost_flow.OPTIMAL:
        raise Exception('There was an issue with the min cost flow input.')

    # Assignment
    labels_M = np.array([min_cost_flow.flow(i) for i in range(n_X * n_C)]).reshape(n_X, n_C).astype('int32')

    labels = labels_M.argmax(axis=1)
    return labels


class ConstrainedKMedoids(BaseEstimator, ClusterMixin, TransformerMixin):
    """k-medoids clustering.

    Read more in the :ref:`User Guide <k_medoids>`.

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of medoids to
        generate.

    metric : string, or callable, optional, default: 'euclidean'
        What distance metric to use. See :func:metrics.pairwise_distances
        metric can be 'precomputed', the user must then feed the fit method
        with a precomputed kernel matrix and not the design matrix X.

    method : {'alternate', 'pam'}, default: 'alternate'
        Which algorithm to use. 'alternate' is faster while 'pam' is more accurate.

    init : {'random', 'heuristic', 'k-medoids++', 'build'}, or array-like of shape
        (n_clusters, n_features), optional, default: 'heuristic'
        Specify medoid initialization method. 'random' selects n_clusters
        elements from the dataset. 'heuristic' picks the n_clusters points
        with the smallest sum distance to every other point. 'k-medoids++'
        follows an approach based on k-means++_, and in general, gives initial
        medoids which are more separated than those generated by the other methods.
        'build' is a greedy initialization of the medoids used in the original PAM
        algorithm. Often 'build' is more efficient but slower than other
        initializations on big datasets and it is also very non-robust,
        if there are outliers in the dataset, use another initialization.
        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        .. _k-means++: https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf

    max_iter : int, optional, default : 300
        Specify the maximum number of iterations when fitting. It can be zero in
        which case only the initialization is computed which may be suitable for
        large datasets when the initialization is sufficiently efficient
        (i.e. for 'build' init).

    random_state : int, RandomState instance or None, optional
        Specify random state for the random number generator. Used to
        initialise medoids when init='random'.

    Attributes
    ----------
    cluster_centers_ : array, shape = (n_clusters, n_features)
            or None if metric == 'precomputed'
        Cluster centers, i.e. medoids (elements from the original dataset)

    medoid_indices_ : array, shape = (n_clusters,)
        The indices of the medoid rows in X

    labels_ : array, shape = (n_samples,)
        Labels of each point

    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    Examples
    --------
    >>> from sklearn_extra.cluster import KMedoids
    >>> import numpy as np

    >>> X = np.asarray([[1, 2], [1, 4], [1, 0],
    ...                 [4, 2], [4, 4], [4, 0]])
    >>> kmedoids = KMedoids(n_clusters=2, random_state=0).fit(X)
    >>> kmedoids.labels_
    array([0, 0, 0, 1, 1, 1])
    >>> kmedoids.predict([[0,0], [4,4]])
    array([0, 1])
    >>> kmedoids.cluster_centers_
    array([[1., 2.],
           [4., 2.]])
    >>> kmedoids.inertia_
    8.0

    See scikit-learn-extra/examples/plot_kmedoids_digits.py for examples
    of KMedoids with various distance metrics.

    References
    ----------
    Maranzana, F.E., 1963. On the location of supply points to minimize
      transportation costs. IBM Systems Journal, 2(2), pp.129-135.
    Park, H.S.and Jun, C.H., 2009. A simple and fast algorithm for K-medoids
      clustering.  Expert systems with applications, 36(2), pp.3336-3341.

    See also
    --------

    KMeans
        The KMeans algorithm minimizes the within-cluster sum-of-squares
        criterion. It scales well to large number of samples.

    Notes
    -----
    Since all pairwise distances are calculated and stored in memory for
    the duration of fit, the space complexity is O(n_samples ** 2).

    """

    def __init__(
        self,
        n_clusters=8,
        metric="euclidean",
        method="alternate",
        init="heuristic",
        min_cluster_size=2,
        max_cluster_size=12,
        max_iter=300,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.metric = metric
        self.method = method
        self.init = init
        self.max_iter = max_iter
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.random_state = random_state


    def _check_nonnegative_int(self, value, desc, strict=True):
        """Validates if value is a valid integer > 0"""
        if strict:
            negative = (value is None) or (value <= 0)
        else:
            negative = (value is None) or (value < 0)
        if negative or not isinstance(value, (int, np.integer)):
            raise ValueError(
                "%s should be a nonnegative integer. "
                "%s was given" % (desc, value)
            )

    def _check_init_args(self):
        """Validates the input arguments."""

        # Check n_clusters and max_iter
        self._check_nonnegative_int(self.n_clusters, "n_clusters")
        self._check_nonnegative_int(self.max_iter, "max_iter", False)

        # Check init
        init_methods = ["random", "heuristic", "k-medoids++", "build"]
        if not (
            hasattr(self.init, "__array__")
            or (isinstance(self.init, str) and self.init in init_methods)
        ):
            raise ValueError(
                "init needs to be one of "
                + "the following: "
                + "%s" % (init_methods + ["array-like"])
            )

        # Check n_clusters
        if (
            hasattr(self.init, "__array__")
            and self.n_clusters != self.init.shape[0]
        ):
            warnings.warn(
                "n_clusters should be equal to size of array-like if init "
                "is array-like setting n_clusters to {}.".format(
                    self.init.shape[0]
                )
            )
            self.n_clusters = self.init.shape[0]

    def fit(self, X, y=None):
        """Fit K-Medoids to the provided data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features), \
                or (n_samples, n_samples) if metric == 'precomputed'
            Dataset to cluster.

        y : Ignored

        Returns
        -------
        self
        """
        random_state_ = check_random_state(self.random_state)

        self._check_init_args()
        X = check_array(
            X, accept_sparse=["csr", "csc"], dtype=[np.float64, np.float32]
        )
        self.n_features_in_ = X.shape[1]
        if self.n_clusters > X.shape[0]:
            raise ValueError(
                "The number of medoids (%d) must be less "
                "than the number of samples %d."
                % (self.n_clusters, X.shape[0])
            )

        D = pairwise_distances(X, metric=self.metric)

        medoid_idxs = self._initialize_medoids(
            D, self.n_clusters, random_state_, X
        )
        labels = None

        if self.method == "pam":
            # Compute the distance to the first and second closest points
            # among medoids.

            if self.n_clusters == 1 and self.max_iter > 0:
                # PAM SWAP step can only be used for n_clusters > 1
                warnings.warn(
                    "n_clusters should be larger than 2 if max_iter != 0 "
                    "setting max_iter to 0."
                )
                self.max_iter = 0
            elif self.max_iter > 0:
                Djs, Ejs = np.sort(D[medoid_idxs], axis=0)[[0, 1]]

        # Continue the algorithm as long as
        # the medoids keep changing and the maximum number
        # of iterations is not exceeded

        for self.n_iter_ in range(0, self.max_iter):
            old_medoid_idxs = np.copy(medoid_idxs)
            labels = np.argmin(D[medoid_idxs, :], axis=0)

            # min flow graph
            if self.n_iter_ > 0:
                num_nodes = len(labels)
                num_medoids = len(medoid_idxs)
                
                dist_to_medoids = np.zeros((num_nodes, num_medoids))
                for i in range(num_nodes):
                    for j in range(len(medoid_idxs)):
                        dist_to_medoids[i, j] = X[i, medoid_idxs[j]]

                edges, costs, capacities, supplies, n_C, n_X = minimum_cost_flow_problem_graph(num_nodes, num_medoids, dist_to_medoids, self.min_cluster_size, self.max_cluster_size)
                labels = solve_min_cost_flow_graph(edges, costs, capacities, supplies, n_C, n_X)

            if self.method == "alternate":
                # Update medoids with the new cluster indices
                self._update_medoid_idxs_in_place(D, labels, medoid_idxs)
            elif self.method == "pam":
                not_medoid_idxs = np.delete(np.arange(len(D)), medoid_idxs)
                optimal_swap = _compute_optimal_swap(
                    D,
                    medoid_idxs.astype(np.intc),
                    not_medoid_idxs.astype(np.intc),
                    Djs,
                    Ejs,
                    self.n_clusters,
                )
                if optimal_swap is not None:
                    i, j, _ = optimal_swap
                    medoid_idxs[medoid_idxs == i] = j

                    # update Djs and Ejs with new medoids
                    Djs, Ejs = np.sort(D[medoid_idxs], axis=0)[[0, 1]]

            else:
                raise ValueError(
                    f"method={self.method} is not supported. Supported methods "
                    f"are 'pam' and 'alternate'."
                )

            if np.all(old_medoid_idxs == medoid_idxs):
                break
            elif self.n_iter_ == self.max_iter - 1:
                warnings.warn(
                    "Maximum number of iteration reached before "
                    "convergence. Consider increasing max_iter to "
                    "improve the fit.",
                    ConvergenceWarning,
                )



        # Set the resulting instance variables.
        if self.metric == "precomputed":
            self.cluster_centers_ = None
        else:
            self.cluster_centers_ = X[medoid_idxs]

        # Expose labels_ which are the assignments of
        # the training data to clusters
        self.labels_ = np.argmin(D[medoid_idxs, :], axis=0)
        self.medoid_indices_ = medoid_idxs
        self.inertia_ = _compute_inertia(self.transform(X))

        # min flow graph
        num_nodes = len(self.labels_)
        num_medoids = len(self.medoid_indices_)
        
        dist_to_medoids = np.zeros((num_nodes, num_medoids))
        for i in range(num_nodes):
            for j in range(len(self.medoid_indices_)):
                dist_to_medoids[i, j] = X[i, self.medoid_indices_[j]]

        edges, costs, capacities, supplies, n_C, n_X = minimum_cost_flow_problem_graph(num_nodes, num_medoids, dist_to_medoids, self.min_cluster_size, self.max_cluster_size)
        self.labels_ = solve_min_cost_flow_graph(edges, costs, capacities, supplies, n_C, n_X)

        # Return self to enable method chaining
        return self

    def _update_medoid_idxs_in_place(self, D, labels, medoid_idxs):
        """In-place update of the medoid indices"""

        # Update the medoids for each cluster
        for k in range(self.n_clusters):
            # Extract the distance matrix between the data points
            # inside the cluster k
            cluster_k_idxs = np.where(labels == k)[0]

            if len(cluster_k_idxs) == 0:
                warnings.warn(
                    "Cluster {k} is empty! "
                    "self.labels_[self.medoid_indices_[{k}]] "
                    "may not be labeled with "
                    "its corresponding cluster ({k}).".format(k=k)
                )
                continue

            in_cluster_distances = D[
                cluster_k_idxs, cluster_k_idxs[:, np.newaxis]
            ]

            # Calculate all costs from each point to all others in the cluster
            in_cluster_all_costs = np.sum(in_cluster_distances, axis=1)

            min_cost_idx = np.argmin(in_cluster_all_costs)
            min_cost = in_cluster_all_costs[min_cost_idx]
            curr_cost = in_cluster_all_costs[
                np.argmax(cluster_k_idxs == medoid_idxs[k])
            ]

            # Adopt a new medoid if its distance is smaller then the current
            if min_cost < curr_cost:
                medoid_idxs[k] = cluster_k_idxs[min_cost_idx]

    def _compute_cost(self, D, medoid_idxs):
        """Compute the cose for a given configuration of the medoids"""
        return _compute_inertia(D[:, medoid_idxs])

    def transform(self, X):
        """Transforms X to cluster-distance space.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Data to transform.

        Returns
        -------
        X_new : {array-like, sparse matrix}, shape=(n_query, n_clusters)
            X transformed in the new space of distances to cluster centers.
        """
        X = check_array(
            X, accept_sparse=["csr", "csc"], dtype=[np.float64, np.float32]
        )

        if self.metric == "precomputed":
            check_is_fitted(self, "medoid_indices_")
            return X[:, self.medoid_indices_]
        else:
            check_is_fitted(self, "cluster_centers_")

            Y = self.cluster_centers_
            kwargs = {}
            if self.metric == "seuclidean":
                kwargs["V"] = np.var(np.vstack([X, Y]), axis=0, ddof=1)
            DXY = pairwise_distances(X, Y=Y, metric=self.metric, **kwargs)

            return DXY

    def predict(self, X):
        """Predict the closest cluster for each sample in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            New data to predict.

        Returns
        -------
        labels : array, shape = (n_query,)
            Index of the cluster each sample belongs to.
        """
        X = check_array(
            X, accept_sparse=["csr", "csc"], dtype=[np.float64, np.float32]
        )

        if self.metric == "precomputed":
            check_is_fitted(self, "medoid_indices_")
            return np.argmin(X[:, self.medoid_indices_], axis=1)
        else:
            check_is_fitted(self, "cluster_centers_")

            # Return data points to clusters based on which cluster assignment
            # yields the smallest distance
            kwargs = {}
            if self.metric == "seuclidean":
                kwargs["V"] = np.var(
                    np.vstack([X, self.cluster_centers_]), axis=0, ddof=1
                )
            pd_argmin = pairwise_distances_argmin(
                X,
                Y=self.cluster_centers_,
                metric=self.metric,
                metric_kwargs=kwargs,
            )

            return pd_argmin

    def _initialize_medoids(self, D, n_clusters, random_state_, X=None):
        """Select initial mediods when beginning clustering."""

        if hasattr(self.init, "__array__"):  # Pre assign cluster
            medoids = np.hstack(
                [np.where((X == c).all(axis=1)) for c in self.init]
            ).ravel()
        elif self.init == "random":  # Random initialization
            # Pick random k medoids as the initial ones.
            medoids = random_state_.choice(len(D), n_clusters, replace=False)
        elif self.init == "k-medoids++":
            medoids = self._kpp_init(D, n_clusters, random_state_)
        elif self.init == "heuristic":  # Initialization by heuristic
            # Pick K first data points that have the smallest sum distance
            # to every other point. These are the initial medoids.
            medoids = np.argpartition(np.sum(D, axis=1), n_clusters - 1)[
                :n_clusters
            ]
        elif self.init == "build":  # Build initialization
            medoids = _build(D, n_clusters).astype(np.int64)
        else:
            raise ValueError(f"init value '{self.init}' not recognized")

        return medoids

    # Copied from sklearn.cluster.k_means_._k_init
    def _kpp_init(self, D, n_clusters, random_state_, n_local_trials=None):
        """Init n_clusters seeds with a method similar to k-means++

        Parameters
        -----------
        D : array, shape (n_samples, n_samples)
            The distance matrix we will use to select medoid indices.

        n_clusters : integer
            The number of seeds to choose

        random_state : RandomState
            The generator used to initialize the centers.

        n_local_trials : integer, optional
            The number of seeding trials for each center (except the first),
            of which the one reducing inertia the most is greedily chosen.
            Set to None to make the number of trials depend logarithmically
            on the number of seeds (2+log(k)); this is the default.

        Notes
        -----
        Selects initial cluster centers for k-medoid clustering in a smart way
        to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
        "k-means++: the advantages of careful seeding". ACM-SIAM symposium
        on Discrete algorithms. 2007

        Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
        which is the implementation used in the aforementioned paper.
        """
        n_samples, _ = D.shape

        centers = np.empty(n_clusters, dtype=int)

        # Set the number of local seeding trials if none is given
        if n_local_trials is None:
            # This is what Arthur/Vassilvitskii tried, but did not report
            # specific results for other than mentioning in the conclusion
            # that it helped.
            n_local_trials = 2 + int(np.log(n_clusters))

        center_id = random_state_.randint(n_samples)
        centers[0] = center_id

        # Initialize list of closest distances and calculate current potential
        closest_dist_sq = D[centers[0], :] ** 2
        current_pot = closest_dist_sq.sum()

        # pick the remaining n_clusters-1 points
        for cluster_index in range(1, n_clusters):
            rand_vals = (
                random_state_.random_sample(n_local_trials) * current_pot
            )
            candidate_ids = np.searchsorted(
                stable_cumsum(closest_dist_sq), rand_vals
            )

            # Compute distances to center candidates
            distance_to_candidates = D[candidate_ids, :] ** 2

            # Decide which candidate is the best
            best_candidate = None
            best_pot = None
            best_dist_sq = None
            for trial in range(n_local_trials):
                # Compute potential when including center candidate
                new_dist_sq = np.minimum(
                    closest_dist_sq, distance_to_candidates[trial]
                )
                new_pot = new_dist_sq.sum()

                # Store result if it is the best local trial so far
                if (best_candidate is None) or (new_pot < best_pot):
                    best_candidate = candidate_ids[trial]
                    best_pot = new_pot
                    best_dist_sq = new_dist_sq

            centers[cluster_index] = best_candidate
            current_pot = best_pot
            closest_dist_sq = best_dist_sq

        return centers
