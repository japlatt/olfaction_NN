from brian2 import *

import sys
sys.path.append("..")

import neuron_models as nm
import lab_manager as lm
import experiments as ex

import matplotlib.pyplot as plt

plt.style.use('ggplot')

defaultclock.dt = .02*ms

N_AL = 1000
in_AL = .35*msiemens/cm**2
PAL = 0.5

#Antennal Lobe parameters
al_para = dict(N = N_AL,
               g_syn = in_AL,
               neuron_class = nm.n_HH, 
               syn_class = nm.s_GABA_inh,
               PAL = PAL,
               mon = ['V']
              )

#create the network object
net = Network()

G_AL, S_AL, trace_AL, spikes_AL = lm.get_AL(al_para, net)



G_AL.I_inj = (ex.get_rand_I(N_AL, p = 0.33, I_inp = 15)+15)*uA/cm**2

net.run(100*ms, report = 'text')


fig1 = plt.figure()
plt.plot(spikes_AL.t/ms, spikes_AL.i, '.')
plt.title('Spikes AL')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron Number')
fig1.savefig('spikes_AL.pdf', bbox_inches = 'tight')

fig2 = plt.figure()
plt.plot(trace_AL.t/ms, mean(trace_AL.V, axis = 0)/mV)
plt.title('Trace AL')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Voltage (mV)')
fig2.savefig('lfp_AL.pdf', bbox_inches = 'tight')

plt.show()
