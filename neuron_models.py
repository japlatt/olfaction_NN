from brian2 import *



#----------------------------------------------------------
#NEURONS

'''
naming convention:  all neurons start with n followed by the name
of the neuron

Each neuron class needs to have the following functions defined

def eqs(self):
    return string of equations in Brian format

def namespace(self):
    return dictionary of variables defined in eqs/threshold...etc

def threshold(self):
    return string 'V > thresh' with voltage that neuron spikes or None

def refractory(self):
    return string or number defining length of time that voltage above
    threshold does not register as spike.  Note this does not change
    behavior of membrane potential unless specified.

def reset(self):
    return voltage that neuron returns to after reaching threshold

def method(self):
    return string 'rk4' or other method of integration

def state_mon(self):
    return array of variables in string format e.g ['V'] that tell
    the simulation to record the value of that variable

def init_cond(self):
    return dict(v = self.vr) dictionary of variables and their initial
    conditions
'''
class n_FitzHugh_Nagumo:

    #global variables for all instances of the class
    nu = -1.5*mV
    a = 0.7
    b = 0.8
    t1 = 0.08*ms
    t2 = 3.1*ms

    thresh = 0*mV
    refrac = -0.5*mV

    states_to_mon = ['V']


    def __init__(self, mon):
        self.states_to_mon = mon
        return
    
    def eqs(self):
        eqns_AL = '''
                    I_inj : amp
                    I_syn : 1
                    dV/dt = (V-(V**3)/(3*mV**2) - w - z*(V - nu) + 0.35*mV + I_inj*Mohm)/t1 : volt
                    dw/dt = (V - b*w + a*mV)/ms : volt
                    dz/dt = (I_syn - z)/t2 : 1
                    '''
        return eqns_AL

    def namespace(self):
        namespace = {'nu': self.nu,
                     'a' : self.a,
                     'b' : self.b,
                     't1': self.t1,
                     't2': self.t2,
                     'refrac': self.refrac,
                     'thresh': self.thresh}
                    
        return namespace

    def threshold(self):
        return 'V > thresh'

    def refractory(self):
        return 'V >= refrac'

    def reset(self):
        return None

    def method(self):
        return 'rk4'

    def state_mon(self):
        return self.states_to_mon
    
    def init_cond(self):
        return  dict(V = -1.2*mV, w = -0.62*mV, z = 'rand()')



#leaky integrate and fire
#non-linear excitatory synapses
#gap junction inhibitory synapses
class n_lif:
    taum = 10*ms #membrane 

    vt = -50*mV #threshold for spike
    vr = -74*mV #resting/reset potential

    Ee = 0*mV
    Ei = -100*mV

    taue = 5*ms #1 ms diehl/cook
    taui = 2*ms

    refKC = 2*ms

    

    def __init__(self, mon):
        self.states_to_mon = mon
        return

    def eqs(self):
        eqns_KCs = '''
                dv/dt = ((vr - v) + (I_synE-I_synI) / nS) / taum  : volt (unless refractory)
                I_synE = ge * nS * (Ee - v)                        : amp
                dge/dt = -ge/taue                                  : 1
                dgi/dt = -gi/taui                                 : 1
                I_synI                                            : amp
                '''
        return eqns_KCs

    def namespace(self):
        ns = dict(  taum = self.taum,
                    vt = self.vt,
                    vr = self.vr,
                    Ee = self.Ee,
                    Ei = self.Ei,
                    taue = self.taue,
                    taui = self.taui,
                    refrac = self.refKC)
        return ns

    def threshold(self):
        return 'v > vt'

    def refractory(self):
        return 'refrac'

    def reset(self):
        return 'v = vr'

    def method(self):
        return 'rk4'

    def state_mon(self):
        return self.states_to_mon

    def init_cond(self):
        return dict(v = self.vr)


#leaky integrate (non spiking)
#Non-linear excitatory synapses
class n_li:

    #GGN

    taum = 10*ms #membrane 
    vr = -74*mV #resting/reset potential

    Ee = 0*mV

    taue = 5*ms #1 ms diehl/cook

    refGGN = 0*ms

    def __init__(self, mon):
        self.states_to_mon = mon
        return

    def eqs(self):
        eqns_GGN = '''
                    dv/dt = ((vr - v) + I_synE/nS) / taum  : volt
                    I_synE = ge * nS * (Ee - v) : amp
                    dge/dt = -ge/taue : 1
                    '''
        return eqns_GGN

    def namespace(self):
        ns = dict(  taum = self.taum,
                    vr = self.vr,
                    Ee = self.Ee,
                    taue = self.taue,
                    refrac = self.refGGN)
        return ns

    def threshold(self):
        return None

    def refractory(self):
        return 'refrac'

    def reset(self):
        return None

    def method(self):
        return 'rk4'

    def state_mon(self):
        return self.states_to_mon

    def init_cond(self):
        return dict(v = self.vr)
#-----------------------------------------------------------------
#SYNAPSES
'''
naming convention:  all synapses start with s, followed by the name of
the synapse, followed by excitation or inhibition

def eqs(self):
    return string of equations in Brian format

def onpre(self):
    return string of equations to execute in the event of a pre-synaptic spike

def onpost(self):
    return string of equations to execute in the event of a post-synaptic spike

def namespace(self):
    return dictionary of the variables and their values

def init_cond(self):
    return initial conditions of the variables
'''
class s_FitzHughNagumo_inh:
    
    def __init__(self, cond):
        self.g_syn = cond
        return
        
    def eqs(self):
        syn = '''
                g_syn: 1
                I_syn_post = g_syn/(1.0+exp(-1000*V_pre/mV)): 1 (summed)
                '''
        return syn

    def onpre(self):
        return None

    def onpost(self):
        return None
        
    def namespace(self):
        return None

    def init_cond(self):
        return {'g_syn': self.g_syn}


class s_lif_ex:

    def __init__(self, conduct):
        self.g_syn = conduct
        return

    def eqs(self):
        return '''w_syn : 1'''

    def onpre(self):
        return '''ge += w_syn'''

    def onpost(self):
        return None

    def namespace(self):
        return None

    def init_cond(self):
        return dict(w_syn = self.g_syn)

class s_lif_in:

    def __init__(self, conduct):
        self.g_syn = conduct
        return

    def eqs(self):
        return '''w_syn : 1'''

    def onpre(self):
        return '''gi += w_syn'''

    def onpost(self):
        return None

    def namespace(self):
        return None

    def init_cond(self):
        return dict(w_syn = self.g_syn)


class s_gapjunc_in:

    def __init__(self, conduct):
        self.g_syn = conduct
        return


    def eqs(self):
        S = '''
             w : 1
             I_synI_post = w*(v_pre - v_post)*nS : amp (summed)
             '''
        return S

    def onpre(self):
        return None

    def onpost(self):
        return None

    def namespace(self):
        return None

    def init_cond(self):
        return dict(w = self.g_syn)


#Empirical STDP
class s_lifSTDP_ex:

    '''
    conduct: max conductance
    eta: learning rate
    taupre/taupost: time constant for STDP decay
    '''
    def __init__(self, conduct, eta, taupre, taupost):
        self.g_syn = conduct
        self.dApre = eta*conduct
        self.dApost = -eta*conduct * taupre / taupost * 1.05
        self.taupre = taupre
        self.taupost = taupost
        return

    def eqs(self):
        S = '''w_syn : 1
            dApre/dt = -Apre / taupre : 1 (event-driven)
            dApost/dt = -Apost / taupost : 1 (event-driven)'''

        return S

    def onpre(self):
        on_pre='''ge += w_syn
                Apre += dApre
                w_syn = clip(w_syn + Apost, 0, g_syn)'''
        return on_pre

    def onpost(self):
        on_post='''Apost += dApost
                w_syn = clip(w_syn + Apre, 0, g_syn)'''

        return on_post

    def namespace(self):
        return dict(g_syn = self.g_syn,
                    dApre = self.dApre,
                    dApost = self.dApost,
                    taupre = self.taupre,
                    taupost = self.taupost)

    def init_cond(self):
        return dict(w_syn = 'rand()*g_syn')
