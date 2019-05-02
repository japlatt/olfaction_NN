import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from itertools import cycle

import matplotlib.pyplot as plt

plt.style.use('ggplot')

def load_wlc_data(prefix, num_runs = 3):
    spikes_t_arr = []
    spikes_i_arr = []
    I_arr = []
    trace_V_arr = []
    trace_t_arr = []
    for i in range(num_runs):
        spikes_t_arr.append(np.load(prefix+'spikes_t_'+str(i)+'.npy'))
        spikes_i_arr.append(np.load(prefix+'spikes_i_'+str(i)+'.npy'))
        I_arr.append(np.load(prefix+'I_'+str(i)+'.npy'))
        trace_V_arr.append(np.load(prefix+'trace_V_'+str(i)+'.npy'))
        trace_t_arr.append(np.load(prefix+'trace_t_'+str(i)+'.npy'))
    return spikes_t_arr, spikes_i_arr, I_arr, trace_V_arr, trace_t_arr

def doPCA(trace_V_arr, k = 3):

    length = len(trace_V_arr[0][0])

    data = np.hstack(trace_V_arr)

    # svd decomposition and extract eigen-values/vectors
    pca = PCA(n_components=k)
    pca.fit(data.T)
    # wk = pca.explained_variance_
    # vk = pca.components_

    # Save the pca data into each odor/conc
    Xk = pca.transform(data.T)

    pca1 = Xk[0:length]
    pca2 = Xk[length:2*length]
    pca3 = Xk[2*length:3*length]

    return [pca1, pca2, pca3]

def plotPCA(PCAdata, N, el = 30, az = 30, skip = 1, start = 50):

    pca1, pca2, pca3 = PCAdata
    #Plot PCA
    cycol = cycle(['#f10c45','#069af3','#02590f','#ab33ff','#ff8c00','#ffd700'])
    # marker = cycle(['^','o','s','p'])


    fig = plt.figure(figsize = (10,7)).gca(projection='3d')
    c = next(cycol)
    # m = next(marker)

    name = [pca1[::skip], pca2[::skip], pca3[::skip]]
    for j in range(3):
        fig.scatter(name[j][start:,0], name[j][start:,1], name[j][start:,2],s=10, color = c)
        c = next(cycol)

    plt.title('PCA ' + str(N) + ' neuron')
    fig.view_init(elev = el, azim = 0)
    # fig.view_init(elev = 30)
    fig.figure.savefig('PCA_' + str(N) + '.pdf', bbox_inches = 'tight')
    # fig.figure.savefig('PCA_' + str(N) + '.png', bbox_inches = 'tight', dpi = 400)