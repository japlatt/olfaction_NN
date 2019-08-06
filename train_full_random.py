import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import sys
import time

import matplotlib.pyplot as plt

if sys.version_info[0] < 3:
    import cPickle as pickle

else:
    import pickle


import neuron_models as nm
import lab_manager as lm
import experiments as ex
from sklearn.utils import shuffle as rshuffle

import time
# import skimage

from brian2 import *

plt.style.use('ggplot')

#compile into C++ standalone
#works, but asking it to compile too much will freeze computer -- uses all cores to compile
# Note: The documentations states this only works for a fixed number of run statements
# and not loops. This doesn't appear to be the case, although maybe it works with
# only a few number of loops? I've gotten 4 runs to work.
# tstart variable also doesn't work because it uses a single value for it.
comp = False

# C++ standalone code to run quicker
if comp:
    set_device('cpp_standalone', debug=True, build_on_run=False)

# Parallelize using OpenMP. Might not work....
# also only works with C++ standalone/comp = True
# prefs.devices.cpp_standalone.openmp_threads = 4


start_scope()
#path to the data folder
MNIST_data_path = 'data_set/'
# MNIST_data_path = '/home/jplatt/Mothnet/MNIST_data/'

#path to folder to save data
prefix = 'total_data/'

#doesn't work if timestep > 0.05ms
defaultclock.dt = .05*ms



#number of images to run
#num_examples = int(raw_input('Number of images to train: ')) #len(training)

#plot some diagnostics at the end
plot = True

# A way to use a synapse to store LFP, useful when you don't want to
# trace the voltages of all AL neurons.
lfp_syn = True
#-----------------------------------------------------------
#Tunable Parameters: the parameters to change in the network


#0-10
numbers_to_inc = frozenset([0, 1])

#size of network
N_AL = 1000 #must be >= 784
N_KC = 10000
N_BL = 3 #should be the same as the number of classes

#learning rate
# 0.1
eta = 0.01 #fraction of total conductance per spike

"""
Amount of inhibition between AL Neurons.
Enforces WLC dynamics and needs to be scaled
with the size of the network
"""
#0.1
in_AL = 0.1

'''Excititation between AL -> KCs'''
#0.2
ex_ALKC = .25


#excitation kenyon cells to beta lobes
#1.5
ex_KCBL = 0.5

#Lateral inhibition beta lobe
#4
in_BLBL = 1.0


#excitation KC->GGN
#0.01
ex_KCGGN = 0.001

#inhibition GGN->KC
#0.2
in_GGNKC = 0.3

#probability inter-AL connections
#0.5
PAL = 0.5

#AL->KCs
#0.01
PALKC = 0.02

#KCs->BLs
#0.3
PKCBL = 0.3

taupre = 15*ms #width of STDP
taupost = taupre

input_intensity = 0.3 #scale input
reset_time = 30 #ms

# needed for gradual current
tr = 20*ms # rise time
tf = 20*ms # fall time
width = 150*ms # duration of constat input
max_inp = input_intensity*nA # max current, can be just unit if amplitude is specified with active_

# total run time, too lazy to change name
time_per_image = (width + tr + tf)/ms

#initialize tstart - this is because the reset is called first
tstart = reset_time*ms

#save the parameters to load in for testing
tunable_params = {'N_AL': N_AL,
                  'N_KC': N_KC,
                  'N_BL': N_BL,
                  'eta': eta,
                  'in_AL': in_AL,
                  'ex_ALKC': ex_ALKC,
                  'ex_KCBL': ex_KCBL,
                  'in_BLBL': in_BLBL,
                  'ex_KCGGN': ex_KCGGN,
                  'in_GGNKC': in_GGNKC,
                  'PAL': PAL,
                  'PALKC': PALKC,
                  'PKCBL': PKCBL,
                  'input_intensity': input_intensity,
                  'time_per_image': time_per_image,
                  'reset_time': reset_time,
                  'width': width,
                  'tr': tr,
                  'tf': tf,
                  'taupre': taupre,
                  'taupost': taupost}

pickle.dump( tunable_params, open(prefix+"connections/tunable_params.p", "wb" ) )

#--------------------------------------------------------

#Antennal Lobe parameters
al_para = dict(N = N_AL,
               g_syn = in_AL,
               neuron_class = nm.n_FitzHugh_Nagumo,
               syn_class = nm.s_FitzHughNagumo_inh,
               PAL = PAL,
               mon = [],
               lfp_syn = lfp_syn
              )

#Kenyon cell parameters
kc_para = dict( N = N_KC,
                neuron_class = nm.n_lif,
                mon = []
                )

#GGN parameters
ggn_para = dict(N = 1,
				neuron_class = nm.n_li,
				mon = ['v'])

#Beta lobe neuron parameters
bl_para = dict(N = N_BL,
			   g_syn = in_BLBL,
			   neuron_class = nm.n_lif,
			   syn_class = nm.s_lif_in,
			   mon = ['v'])

#connect all the layers together
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
                 PKCBL = PKCBL)



t_sim = time.time()

#create the network object
net = Network()
# monlfp if using synapse to monitor
G_AL, S_AL, trace_AL, spikes_AL, G_LFP, S_LFP, trace_LFP = lm.get_AL(al_para, net)

G_KC, trace_KC, spikes_KC = lm.get_KCs(kc_para, net)

G_GGN, trace_GGN = lm.get_GGN(ggn_para, net)

G_BL, S_BL, trace_BL, spikes_BL = lm.get_BL(bl_para, net)

states = [G_AL, G_KC, G_GGN, G_BL]

S_ALKC, S_KCGGN, S_GGNKC, S_KCBL = lm.connect_network(conn_para, states, net)

#----------------------------------------------------------
'''
There are a few components to using a time dependent current.

(1) The string I. This is just a string which describes the function and units
of the time dependent current.

(2) The run_regularly function. This exists in the namespace of the NeuronGroup
you're adding it to. It can execute code strings at regular intervals. This
object needs to be assigned a name and added to the network separately if the
neuron group has previous been added.

(3) Additional parameters: If there are any additional parameters in the string
(rise time, fall time, etc), they need to be definied in the execution script.

(4) Random input - This can be combined with get_rand_I to select the neurons
that receive this time dependent current. If the input is scaled by max_inp,
the last argument in get_rand_I should be 1. Do not double scale input!

'''

# tr, tf, width, max_inp need to be defined
I = ex.get_gradual_current()


# dt here can be smaller than default clock. This is dt of the data provided.
G_run = G_AL.run_regularly('I_inj = {}'.format(I),dt = 0.05*ms)
net.add(G_run)

# troubleshooting function
#@network_operation(dt=5*ms)
#def f2(t):
#    print(G_AL.I_inj[0])
#net.add(f2)



# random input
num_classes = 3
samples_per_class = 750
n_samples = int(samples_per_class*num_classes)

p_inj = 0.3
X = np.zeros((num_classes,N_AL))
for j in range(num_classes):
    X[j,:] = ex.get_rand_I(N_AL,p_inj,1)

# troubleshooting array
#test_array = np.zeros(N_AL)
#test_array[0] = 0.5
#test_array[999] = 1.0
#run the network
scale_noise = 0.05
# Run random input with gradual current
for i in range(n_samples):
    # turns off all neurons
    G_AL.active_ = 0
    net.run(reset_time*ms)

    G_AL.active_ = X[i%num_classes,:] + scale_noise*input_intensity*np.random.uniform(low=0,high=1,size=len(X[i%num_classes,:]))
    #G_AL.active_ = test_array
    net.run(time_per_image*ms, report='text')
    tstart = tstart + time_per_image*ms + reset_time*ms

print('Simulation time: {0} seconds'.format(time.time()-t_sim))

# run if built in C++ standalone -- this takes a LONG time
if comp:
    print("Compiling...")
    device.build(directory=prefix+'run_dir', compile=True, run=True, debug=True)
# store function not defined for C++ standalone
else:
    net.store(name = 'trained', filename = prefix+'connections/trained')


#save some of the data
np.savez(prefix+'connections/S_AL.npz', i = S_AL.i, j = S_AL.j)
np.savez(prefix+'connections/S_KCBL.npz', i = S_KCBL.i, j = S_KCBL.j)
np.savez(prefix+'connections/S_ALKC.npz', i = S_ALKC.i, j = S_ALKC.j)

np.save(prefix+'output/spikesAL_t' ,spikes_AL.t/ms)
np.save(prefix+'output/spikesAL_i',spikes_AL.i)
# np.save(prefix+'traceAL_V', trace_AL.V)

np.save(prefix+'output/spikesKC_t' ,spikes_KC.t/ms)
np.save(prefix+'output/spikesKC_i',spikes_KC.i)
# np.save(prefix+'traceKC_V', trace_KC.v)

np.save(prefix+'output/spikesBL_t' ,spikes_BL.t/ms)
np.save(prefix+'output/spikesBL_i',spikes_BL.i)
np.save(prefix+'output/traceBL_V', trace_BL.v)
np.save(prefix+'output/trace_t', trace_BL.t)

np.save(prefix+'output/traceGGN_V', trace_GGN.v)

np.save(prefix+'output/weights', S_KCBL.w_syn)

np.save(prefix+'input',X)
#plot diagnostics
if plot:
    fig1 = plt.figure()
    plt.plot(spikes_AL.t/ms, spikes_AL.i, '.')
    plt.title('Spikes AL')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Number')
    fig1.savefig(prefix+'images/spikes_AL_train.png', bbox_inches = 'tight')

    fig2 = plt.figure()
    plt.plot(spikes_KC.t/ms, spikes_KC.i, '.')
    plt.title('Spikes KC')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Number')
    fig2.savefig(prefix+'images/spikes_KC_train.png', bbox_inches = 'tight')

    fig3 = plt.figure()
    plt.plot(spikes_BL.t/ms, spikes_BL.i, '.')
    plt.title('Spikes BL')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Number')
    plt.ylim(-0.5, N_BL-0.5)
    fig3.savefig(prefix+'images/spikes_BL_train.png', bbox_inches = 'tight')

    fig4 = plt.figure()
    hist(S_KCBL.w_syn / ex_KCBL, 20)
    xlabel('Weight / gmax')
    plt.title('Weights BL')
    fig4.savefig(prefix+'images/weights_BL_train.png', bbox_inches = 'tight')

    fig5 = plt.figure()
    plt.plot(trace_GGN.t/ms,trace_GGN.v[0]/mV)
    plt.title('Trace GGN')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Voltage (mV)')
    fig5.savefig(prefix+'images/trace_GGN_train.png', bbox_inches = 'tight')

    fig7 = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(trace_BL.t/ms, trace_BL.v[0]/mV)
    plt.subplot(2,1,2)
    plt.plot(trace_BL.t/ms, trace_BL.v[1], 'b')
    plt.suptitle('Trace BL train')
    fig7.savefig(prefix+'images/trace_BL_train.png', bbox_inches = 'tight')

    #fig8 = plt.figure()
    #plt.plot(trace_AL.t/ms, mean(trace_AL.I_inj, axis=0)/nA)
    #plt.title('Injected current')
    #plt.xlabel('Time (ms)')
    #plt.ylabel('Avg Current (nA)')

    if lfp_syn:
        fig9 = plt.figure()
        plt.plot(trace_LFP.t/ms, trace_LFP.V[0]/mV)
        plt.title('LFP AL')
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Voltage (mV)')
        fig9.savefig(prefix+'images/lfp_AL_train.png', bbox_inches = 'tight')

    else:
        fig6 = plt.figure()
        plt.plot(trace_AL.t/ms, mean(trace_AL.V, axis = 0)/mV)
        plt.title('LFP AL')
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Voltage (mV)')
        fig6.savefig(prefix+'images/lfp_AL_train.png', bbox_inches = 'tight')

    #plt.show()
