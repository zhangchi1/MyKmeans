## Data Mining project 1, my random k-means and k-means ++ functions
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy import spatial

def compute_cosine_similarity(vector1, vector2):
  '''Computes the cosine similarity of the two input vectors.

  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

  Returns:
    A scalar similarity value.
  '''
  dot_product = np.dot(vector1, vector2)
  vector1_len = np.sum(vector1 ** 2) ** 0.5
  vector2_len = np.sum(vector2 ** 2) ** 0.5
  dot_len = vector1_len * vector2_len
  if dot_len == 0:
      return 0
  return dot_product/ dot_len
# get the distance method between x and y, default method is euclidian
def get_distMethod(x, y, method='euclidian'):
    dist = 0
    if method == 'l1':
        dist= spatial.distance.cityblock(x,y)
    elif method == 'euclidian':
        dist= np.linalg.norm(x - y)
    elif method == 'euclidian_squared':
        dist= euclidean_distances(x, y, squared=True)
    elif method =='cosine':
        dist = compute_cosine_similarity(x, y)
    return dist

# choose k number of points from points, so that each of them are as far apart
# from each other.
# input:
# @points is the given data points as an np array
# @k select k number of points
# @distance is the method if distance
# ouput: k data points as an np array
def choose_farthest_points(points, k, distance='euclidian'):
    m = points.shape[0]
    remain_points = points
    # init k selected points 
    k_points = []
    choice = np.random.choice(m, 1, replace=False)
    first_point = remain_points[choice,:]
    remain_points = np.delete(remain_points, choice, axis=0)
    k_points.append(first_point)
    for num_k in range(k-1):
        distances = []
        # get distances between first point with all other points
        for p_index, point in enumerate(remain_points):
            distances.append(get_distMethod(point, k_points[0]))
            for k_index, k_point in enumerate(k_points):
                distances[p_index] = min(distances[p_index], get_distMethod(point, k_point,distance))
        next_point_index = distances.index(max(distances))
        next_point = remain_points[next_point_index,:]
        remain_points = np.delete(remain_points, next_point_index, axis=0)       
        k_points.append(next_point)
    return np.array(k_points)
    
# find closest centriod for each sample,
# output:
# @current_clusters: for each sample m x1 
# @cost: the cost for all samples
def find_closest_centriods(cleanFile, current_centroids, distance='euclidian'):
    cost = 0
    current_clusters = [] #init current_clusters m x 1
    # find closest centriod for each sample
    for sample_index, sample in enumerate(cleanFile):
        closest_centroid_id = 0
        closest_centroid_dist = get_distMethod(sample, current_centroids[0], distance)
        for centroid_index, centroid in enumerate(current_centroids):
            current_dist = get_distMethod(sample, centroid, distance)
            #sign each sample to its closest centroid
            if(closest_centroid_dist > current_dist):
                closest_centroid_id = centroid_index
                closest_centroid_dist = current_dist
        # save each sample's cluster in current_clusters
        current_clusters.append(closest_centroid_id)
        # upate cost for each iteration, add cost for each sample
        cost += get_distMethod(sample, current_centroids[closest_centroid_id], distance)
    return np.array(current_clusters), cost

# update each centroids position
# save each sample in its corresponding cluster, as a dictionary,
# where the cluster index is the key 
# ouput: 
# @cluster_centroids: an array of centroids: k x features(n)
def update_centroids(cleanFile, clusters, features, k):  
    samples_dict = {}
    cluster_centroids = [] 
    # init samples_dict
    for i in range(k):
        samples_dict[i] = np.empty((0,features))
    # update samples_dict based on clusters list
    for cluster_index, cluster in enumerate(clusters):
        samples_dict[cluster] = np.append(samples_dict[cluster], 
                    cleanFile[cluster_index].reshape(1, features), axis=0)
    for i in range(k):
        cluster_centroids.append(np.mean(samples_dict[i], axis=0))
    return np.array(cluster_centroids)
    
# the random k-means algorithm 
# inputs: 
# @cleanFile is the dataset of the clean source file. Datatype: np array
# @k is the k number of clusters
# @max_iter is the maximum iteration of this algorithm for a single run
# @distance is the method to calculate the distance, default euclidian
# @epsilon: The minimum error to be used in the stop condition (optional, default == 0.1)
# ouptput:
# @ an array, the same length as dataset X, containing the cluster id of each item.
def myRandomKmeans(cleanFile, k, max_iter=500, distance='euclidian', epsilon=0.1):
    m = cleanFile.shape[0]
    features = cleanFile.shape[1]
    # choose random samples init centroids
    choice = np.random.choice(m, k, replace=False)
    init_centroids = cleanFile[choice,:]
    current_centroids = init_centroids
    # init clusters for each sample: a m x 1 vector
    clusters = []
    # random init the costs 
    costs = [888, 777]
    iteration = 0
    centroid_history = []
    # update samples' clusters, then update centroids
    while (iteration < max_iter) & ((costs[-2] - costs[-1]) > epsilon):
        centroid_history.append(current_centroids)
        iteration +=1
        current_clusters, cost = find_closest_centriods(cleanFile, current_centroids, distance)
        # update clusters for each iteration    
        clusters = current_clusters
        # add cost for each iteration to costs
        costs.append(cost)       
        # update each centroids position      
        current_centroids = update_centroids(cleanFile, clusters, features, k)
    return clusters, current_centroids, costs, centroid_history

# the k-means ++ algorithm 
# inputs: 
# @cleanFile is the dataset of the clean source file. Datatype: np array
# @k is the k number of clusters
# @max_iter is the maximum iteration of this algorithm for a single run
# @distance is the method to calculate the distance, default euclidian
# @epsilon: The minimum error to be used in the stop condition (optional, default == 0.1)
# ouptput:
# @ an array, the same length as dataset X, containing the cluster id of each item.
def myKmeansPlusPlus(cleanFile, k, max_iter=500, distance='euclidian', epsilon=0.1):
    features = cleanFile.shape[1]    
    # choose k farthest points from each other 
    init_centroids = choose_farthest_points(cleanFile, k, distance)    
    current_centroids = init_centroids
    # init clusters for each sample: a m x 1 vector
    clusters = []
    # random init the costs 
    costs = [888, 777]
    iteration = 0
    centroid_history = []
    # update samples' clusters, then update centroids
    while (iteration < max_iter) & ((costs[-2] - costs[-1]) > epsilon):
        centroid_history.append(current_centroids)
        iteration +=1
        current_clusters, cost = find_closest_centriods(cleanFile, current_centroids, distance)
        # update clusters for each iteration    
        clusters = current_clusters
        # add cost for each iteration to costs
        costs.append(cost)
        # update each centroids position      
        current_centroids = update_centroids(cleanFile, clusters, features, k)
    return clusters, current_centroids, costs, centroid_history
