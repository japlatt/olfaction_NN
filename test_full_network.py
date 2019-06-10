from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../../')
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import pickle

import neuron_models as nm
import lab_manager as lm
import experiments as ex

import time

from brian2 import *

plt.style.use('ggplot')


comp = False
# C++ standalone code to run quicker
#TODO: not working yet
if comp:
    set_device('cpp_standalone', build_on_run=False)

# Parallelize using OpenMP. Might not work....
# prefs.devices.cpp_standalone.openmp_threads = 4


start_scope()

'''To run you need to download
the MNIST dataset from http://yann.lecun.com/exdb/mnist/'''

#path to the data folder
MNIST_data_path = 'data_set/'
# MNIST_data_path = '/home/jplatt/Mothnet/MNIST_data/'

#path to folder to save data
prefix = 'total_data/'

#doesn't work if timestep > 0.05ms
defaultclock.dt = .05*ms

#number of images to run
#num_examples = int(raw_input('Number of images to test: ')) #len(training)

#plot some diagnostics at the end
plot = True

lfp_syn = True

#-----------------------------------------------------------
#tunable params
tunable_params = pickle.load( open(prefix + "connections/tunable_params.p", "rb"))


#0-10
numbers_to_inc = tunable_params['numbers_to_inc']

#size of network
N_AL = tunable_params['N_AL'] #doesn't work if you change this number for now
N_KC = tunable_params['N_KC']
N_BL = tunable_params['N_BL']

#learning rate
eta = tunable_params['eta']

"""
Amount of inhibition between AL Neurons.
Enforces WLC dynamics and needs to be scaled
with the size of the network
"""
in_AL = tunable_params['in_AL']
# didn't save as tunable_param yet
#lfp_syn = tunable_params['lfp_syn']

'''Excititation between AL -> KCs'''
ex_ALKC = tunable_params['ex_ALKC']

#excitation kenyon cells to beta lobes
ex_KCBL = tunable_params['ex_KCBL']

#Lateral inhibition beta lobe
in_BLBL = tunable_params['in_BLBL']


#excitation KC->GGN
ex_KCGGN = tunable_params['ex_KCGGN']
#inhibition GGN->KC
in_GGNKC = tunable_params['in_GGNKC']

PAL = tunable_params['PAL']
PALKC = tunable_params['PALKC']
PKCBL = tunable_params['PKCBL']


input_intensity = tunable_params['input_intensity'] #scale input
time_per_image = tunable_params['time_per_image'] #ms
reset_time = tunable_params['reset_time'] #ms

bin_thresh = tunable_params['bin_thresh'] #threshold for binary

taupre = tunable_params['taupre'] #width of STDP
taupost = tunable_params['taupost']

S_AL_conn = np.load(prefix+'connections/S_AL.npz')
S_ALKC_conn = np.load(prefix + 'connections/S_ALKC.npz')
S_KCBL_conn = np.load(prefix + 'connections/S_KCBL.npz')

#--------------------------------------------------------
# Rebuild same network for testing
al_para = dict(N = N_AL,
               g_syn = in_AL,
               neuron_class = nm.n_FitzHugh_Nagumo,
               syn_class = nm.s_FitzHughNagumo_inh,
               p = PAL,
               mon = [],
               S_AL_conn = S_AL_conn,
               lfp_syn = lfp_syn
              )

kc_para = dict( N = N_KC,
                neuron_class = nm.n_lif,
                mon = []
                )

ggn_para = dict(N = 1,
                neuron_class = nm.n_li,
                mon = ['v'])

bl_para = dict(N = N_BL,
               g_syn = in_BLBL,
               neuron_class = nm.n_lif,
               syn_class = nm.s_lif_in,
               mon = ['v'])

conn_para = dict(synALKC_class = nm.s_lif_ex,
                 ex_ALKC = ex_ALKC,
                 synKCGGN_class = nm.s_lif_ex,
                 ex_KCGGN = ex_ALKC,
                 synGGNKC_class = nm.s_gapjunc_in,
                 in_GGNKC = in_GGNKC,
                 synKCBL_class = nm.s_lifSTDP_ex,
                 ex_KCBL = ex_KCBL,
                 etaSTDP = eta,
                 taupreSTDP = taupre,
                 taupostSTDP = taupost,
                 PALKC = PALKC,
                 PKCBL = PKCBL,
                 S_ALKC_conn = S_ALKC_conn,
                 S_KCBL_conn = S_KCBL_conn)
# Current used
I = '2*(0.5*(1-tanh(-3.5*(t-tstart)/tr)) - 0.5)*0.5*(1-tanh(-3.5*(width+tr-(t-tstart))/tf))*input_intensity*nA'
@network_operation()
def f(t):
    G_AL.I_inj = I

net = Network(f)

G_AL, S_AL, trace_AL, spikes_AL, G_LFP, S_LFP, trace_LFP = lm.get_AL(al_para, net, train = False)

G_KC, trace_KC, spikes_KC = lm.get_KCs(kc_para, net)

G_GGN, trace_GGN = lm.get_GGN(ggn_para, net)

G_BL, S_BL, trace_BL, spikes_BL = lm.get_BL(bl_para, net)


states = [G_AL, G_KC, G_GGN, G_BL]

S_ALKC, S_KCGGN, S_GGNKC, S_KCBL = lm.connect_network(conn_para, states, net, train = False)

#-------------------------------------------------
start = time.time()
testing = np.load(prefix+'input.npy')

#testing = ex.get_labeled_data(MNIST_data_path + 'testing', MNIST_data_path, bTrain = False)
end = time.time()
print('time needed to load test set:', end - start)

#n_input = testing['rows']*testing['cols'] #28x28=784

#num_tot_images = len(testing['x'])
#imgs = testing['x']
#labels = testing['y']

#-----------------------------------------------------
pred_vec = []

num_classes = np.shape(testing)[0]
samples_per_class = 1
tr = 10*ms
tf = 10*ms
time_per_image = 50
width = time_per_image*ms
time_per_image = width + tr + tf
tstart = 0*ms
num_examples = int(num_classes*samples_per_class)

#net.restore(name = 'trained', filename = prefix + 'connections/trained')
#spikes_ABL = SpikeMonitor(G_BL)
#net.add(spikes_ABL)
#net.store(name = 'Add monitors')

# Random Input
for i in range(num_examples):
    net.restore(name = 'trained', filename=prefix+'connections/trained')
    spikes_BL_test = SpikeMonitor(G_BL)
    net.add(spikes_BL_test)
    #G_AL.active_ = 0
    #net.run(reset_time*ms)
    print('After net.restore: {}'.format(spikes_BL_test.count[1]))
    G_AL.active_ = testing[i%num_classes,:]
    net.run(time_per_image,report='text')

    print('After running: {}'.format(spikes_BL_test.count[1]))
    # net.restore prevents need for reset?
    tstart = tstart + time_per_image

    max_act = 0
    pred = -1
    trains = spikes_BL_test.spike_trains()
    for k in range(len(trains)):
        if len(trains[k]) > max_act:
            pred = k
            max_act = len(trains[k])
    net.remove(spikes_BL_test)
    pred_vec.append((i%num_classes, pred))
"""
# MNIST
j = 0
for i in range(num_tot_images):
    if labels[i][0] in numbers_to_inc:
        net.restore(name = 'trained', filename = prefix + 'connections/trained')
        print('image: ' + str(j))
        j = j+1
        # print(labels[i][0])


        #right now creating binary image
        rates = np.where(imgs[i%10000,:,:] > bin_thresh, 1, 0)*input_intensity
        linear = np.ravel(rates)

        padding = N_AL - n_input
        I = np.pad(linear, (0,padding), 'constant', constant_values=(0,0))
        G_AL.I_inj = I*nA

        net.run(time_per_image*ms, report = 'text')

        max_act = 0
        pred = -1
        trains = spikes_BL.spike_trains()
        for k in range(len(trains)):
            if len(trains[k]) > max_act:
                pred = k
                max_act = len(trains[k])

        pred_vec.append((labels[i][0], pred))
    if j == num_examples:
        break
"""
# run if built in C++ standalone
if comp:
    device.build(directory=prefix+'run_dir', compile=True, run=True, debug=False)

print(pred_vec)
acc = 0
for (label, pred) in pred_vec:
    if label == pred:
        acc+=1
acc = acc*1.0/num_examples
print(acc)

"""
plt.plot(spikes_AAL.t/ms, spikes_AAL.i, '.')
plt.title('Spikes AL')
plt.xlabel('TIme (ms)')
plt.ylabel('Neuron Number')
plt.ylim(-0.5,N_AL-0.5)
"""


#plt.plot(spikes_BL.t/ms, spikes_ABL.i, '.')
#plt.plot(spikes_BL.t/ms, spikes_BL.i, '.')
#plt.title('Spikes BL')
#plt.xlabel('Time (ms)')
#plt.ylabel('Neuron Number')
#lt.ylim(-0.5, N_BL-0.5)
#plt.show()

np.savetxt(prefix+'predictions.txt', pred_vec)
