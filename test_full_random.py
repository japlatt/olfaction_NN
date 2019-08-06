from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
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

#path to folder to save data
prefix = 'total_data/'

#doesn't work if timestep > 0.05ms
defaultclock.dt = .05*ms

#plot some diagnostics at the end
plot = True

lfp_syn = True

#-----------------------------------------------------------
#tunable params
net = Network()

tunable_params = pickle.load( open(prefix + "connections/tunable_params.p", "rb"))

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
time_per_image = tunable_params['time_per_image']
reset_time = tunable_params['reset_time'] #ms
width = tunable_params['width'] #ms
tr = tunable_params['tr'] #ms
tf = tunable_params['tf'] #ms
max_inp = input_intensity*nA

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

G_AL, S_AL, trace_AL, spikes_AL, G_LFP, S_LFP, trace_LFP = lm.get_AL(al_para, net, train = False)

G_KC, trace_KC, spikes_KC = lm.get_KCs(kc_para, net)

G_GGN, trace_GGN = lm.get_GGN(ggn_para, net)

G_BL, S_BL, trace_BL, spikes_BL = lm.get_BL(bl_para, net)

states = [G_AL, G_KC, G_GGN, G_BL]

S_ALKC, S_KCGGN, S_GGNKC, S_KCBL = lm.connect_network(conn_para, states, net, train = False)

#-------------------------------------------------
start = time.time()
testing = np.load(prefix+'input.npy')
print(np.shape(testing))
#testing = ex.get_labeled_data(MNIST_data_path + 'testing', MNIST_data_path, bTrain = False)
end = time.time()
print('time needed to load test set:', end - start)

#-----------------------------------------------------
pred_vec = []

num_classes = np.shape(testing)[0]
samples_per_class = 1
tstart = 0*ms
num_examples = int(num_classes*samples_per_class)

# Find some way to automate this?
I = ex.get_gradual_current()
G_run = G_AL.run_regularly('I_inj = {}'.format(I),dt = 0.02*ms)
net.add(G_run)

# Random Input
#@network_operation(dt = 10*ms)
#def f2(t):
#    print(G_AL.I_inj[0])
#net.add(f2)

net.restore(name = 'trained', filename = prefix + 'connections/trained')

# Adds a new spike monitor for counts
spikes_BL_test = SpikeMonitor(G_BL)
net.add(spikes_BL_test)


# set to zero
G_AL.scale = 0.0
net.run(20*ms)
tstart = trace_GGN.t[-1]
net.store(name = 'test')

neuron_classes = []

# Set the neurons for each class by running a test
for j in range(num_classes):
    net.restore(name='test')

    G_AL.scale = testing[j,:]
    net.run(time_per_image*ms,report='text')

    max_act = 0
    pred = -1
    trains = spikes_BL_test.spike_trains()
    for k in range(len(trains)):
        if len(trains[k]) > max_act:
            pred = k
            max_act = len(trains[k])
    neuron_classes.append(pred)

for i in range(num_examples):
    net.restore(name = 'test')

    print('After net.restore: {}'.format(spikes_BL_test.count[1]))
    G_AL.scale = testing[i%num_classes,:]
    net.run(time_per_image*ms,report='text')

    print('After running: {}'.format(spikes_BL_test.count[1]))

    max_act = 0
    pred = -1
    trains = spikes_BL_test.spike_trains()
    for k in range(len(trains)):
        if len(trains[k]) > max_act:
            pred = k
            max_act = len(trains[k])
    pred_vec.append((neuron_classes[i%num_classes], pred))

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
np.savetxt(prefix+'predictions.txt', pred_vec)

# Only worth plotting new trackers
if plot:
    fig3 = plt.figure()
    plt.plot(spikes_BL.t/ms, spikes_BL.i, '.')
    plt.title('Spikes BL')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Number')
    plt.ylim(-0.5, N_BL-0.5)
    fig3.savefig(prefix+'images/spikes_BL_test.png', bbox_inches = 'tight')
