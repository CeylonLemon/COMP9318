import pickle
import numpy as np

# How to run your implementation for Part 1


def move_centroids(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    output = []
    for k in range(centroids.shape[0]):
        if not len(points[closest == k]):
            output.append(centroids[k])
        else:
            output.append(np.median(points[closest == k], axis=0))
    return np.array(output)


def closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.abs(abs(points - centroids[:, np.newaxis]).sum(axis=2))
    return np.argmin(distances, axis=0)


def compute_distances(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.abs(abs(points - centroids[:, np.newaxis])).sum(axis=2)
    return distances


def pq(data,P, init_centroids, max_iter):
    shape1 = int(data.shape[0])
    shape2 = int(data.shape[1]/P)
    data = data.reshape(P,shape1, shape2)
    Data = [[] for i in range(P)]
    for x in range(P*data.shape[1]):
        Data[x%P].append(data[int(x/data.shape[1])][int(x%data.shape[1])])
    Data = np.array(Data)
    i = 0
    centro = init_centroids

    while(i<max_iter):
        clust = []
        for par in range(P):
            clust = closest_centroid(Data[par],centro[par])
            centro[par] = move_centroids(Data[par],clust, centro[par])
        i+=1
    code = []
    for par in range(P):
        code.append(closest_centroid(Data[par], centro[par]))
    code = np.swapaxes(np.array(code),0,1)
    return centro.astype('float32'),code.astype('uint8')


def query(quries, codebooks, codes, T):
    p = codebooks.shape[0]
    candidate = []
    shape1 = int(quries.shape[0])
    shape2 = int(quries.shape[1]/p)

    Query = quries.reshape(p,shape1, shape2)
    Data = [[] for i in range(p)]
    for x in range(p*Query.shape[1]):
        Data[x%p].append(Query[int(x/Query.shape[1])][int(x%Query.shape[1])])
    Query = np.array(Data)

    # to compute distance

    for q in range(Query.shape[1]):
        p_dist_index = []
        p_dist = []
        for par in range(p):
            dist = list(compute_distances(Query[par][q], codebooks[par]))
            index = [i for i in range(len(codebooks[par]))]
            p_dist.append(dist)
            p_dist_index.append(sorted(zip(dist,index)))
        #p_dist = np.swapaxes(p_dist,1,0)
        p_dist = [[p[0] for p in p_] for p_ in p_dist]
        virtual_dist = []
        for n in range(len(codes)):
            dist = 0
            for par in range(p):
                d = p_dist[par][int(codes[n][par])]
                dist += d
            virtual_dist.append((dist,n))
        virtual_dist.sort()
        count = -1
        cand_this_term = []
        while (count < T-1):
            count+=1
            cand_loc = virtual_dist[count][1]
            cand_this_term.append(cand_loc)
            while(virtual_dist[count+1][0]==virtual_dist[count][0]):
                count+=1
                cand_loc = virtual_dist[count][1]
                cand_this_term.append(cand_loc)
        candidate.append(set(cand_this_term))

    return candidate

