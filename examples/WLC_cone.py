from brian2 import *

import sys
sys.path.append("..")

import neuron_models as nm
import lab_manager as lm
import experiments as ex
import analysis as anal
from itertools import cycle

import matplotlib.pyplot as plt

from __future__ import division

# plt.style.use('ggplot')

np.random.seed(10)

defaultclock.dt = .05*ms

N_AL = 1000
in_AL = .1
PAL = 0.5

inp = 0.15
noise = 2.0

run_time = 100*ms

prefix = 'data/'

#Antennal Lobe parameters
al_para = dict(N = N_AL,
               g_syn = in_AL,
               neuron_class = nm.n_FitzHugh_Nagumo,
               syn_class = nm.s_FitzHughNagumo_inh,
               PAL = PAL,
               mon = ['V']
              )

#create the network object
net = Network()

G_AL, S_AL, trace_AL, spikes_AL = lm.get_AL(al_para, net)
# set just because
G_AL.active_ = 1

net.store()

num_odors = 1
I_arr = []

#create the base odors
for i in range(num_odors):
    I = ex.get_rand_I(N_AL, p = 0.33, I_inp = inp)*nA
    I_arr.append(I)

num_trials = 2

n = 0 #Counting index
start = 100 #skip the first 15 ms to remove transients
for j in range(num_odors):
    for k in range(num_trials):
        net.restore()

        if k == 0:
            G_AL.I_inj = I_arr[j]
        else:
            G_AL.I_inj = I_arr[j]+noise*inp*(2*np.random.random(N_AL)-1)*nA

        net.run(run_time, report = 'text')

        np.save(prefix+'spikes_t_'+str(n) ,spikes_AL.t)
        np.save(prefix+'spikes_i_'+str(n) ,spikes_AL.i)
        np.save(prefix+'I_'+str(n), I)
        np.save(prefix+'trace_V_'+str(n), trace_AL.V[:,start:])
        np.save(prefix+'trace_t_'+str(n), trace_AL.t[start:])
        n = n+1

spikes_t_arr, spikes_i_arr, I_arr, trace_V_arr, trace_t_arr = anal.load_wlc_data(prefix, num_runs = num_odors*num_trials)

base_od = []
for i in range(num_odors):
    base_od.append(trace_V_arr[i*num_trials])


pca_arr = anal.doPCA(trace_V_arr, k = 2)
# EV = anal.getPrincEV(base_od, k = 2)

# # am = angle_mat(vk_arr)
# pca_arr = anal.pcaEV(trace_V_arr, EV)

title = 'Noise ' + str(np.rint(100*noise/sqrt(3)))+'%'
name = 'Noise_' + str(np.rint(100*noise/sqrt(3)))+'%.pdf'
# title = 'Principle Vector Analysis Noise '+ str(np.rint(100*noise/sqrt(3)))+'%'
# name = 'Principle_Vector_Analysis_noise.pdf'
anal.plotPCA2D(pca_arr, title, name, num_trials, skip = 2)

# ang_mat = anal.angle_mat(trace_V_arr)
# np.savetxt('angle_mat.txt', np.round(ang_mat, 2), fmt = '%1.2f')
# print(ang_mat)

plt.show()

# PCAdata = anal.doPCA(trace_V_arr, k = 3)
#% noise:  100*(inp*noise*std(Uniform(-1.0, 1.0)))/inp
# anal.plotPCA3D(PCAdata, N_AL, title = 'Trans RMS Noise Level ' + str(np.round(100*noise/sqrt(3))) + '%', el = 0, az = 0, skip = 1, start = 100)

# plt.show()


# # anal.getMIM(trace_V_arr) #takes a long time
# MIM = np.load('MIM.npy')
# data = np.hstack(trace_V_arr).T
# length = len(trace_V_arr[0][0])

# InCAdata = anal.doInCA(MIM, data, length, skip = 2, k = 3)
# anal.plotInCA(InCAdata, N_AL, start = 200)

# col = ['#f10c45','#069af3','#02590f','#ab33ff','#ff8c00','#ffd700']

# fig1 = plt.figure()
# plt.plot(spikes_t_arr[0]/ms, spikes_i_arr[0], '.', color = col[0])
# plt.plot(spikes_t_arr[1]/ms, spikes_i_arr[1], '.', color = col[1])
# plt.plot(spikes_t_arr[2]/ms, spikes_i_arr[2], '.', color = col[2])
# plt.title('Hodgkin Huxley AL', fontsize = 22)
# plt.xlabel('Time (ms)', fontsize = 16)
# plt.ylabel('Neuron Number', fontsize = 16)
# fig1.savefig('spikes_AL.pdf', bbox_inches = 'tight')

# fig2 = plt.figure(figsize = (8, 6))
# plt.plot(trace_t_arr[0]/ms, mean(trace_V_arr[0], axis = 0)/mV, linewidth = 2)
# plt.title('LFP AL ' +str(N_AL)+' Neuron HH', fontsize = 22)
# plt.xlabel('Time (ms)', fontsize = 16)
# plt.ylabel('LFP (mV)', fontsize = 16)
# plt.ylim(-50, -80)
# fig2.savefig('lfp_AL.pdf', bbox_inches = 'tight')
