from pathlib import Path
from thor_scsi.factory import accelerator_from_config
from thor_scsi.pyflame import Config
import thor_scsi.lib as tslib

import numpy as np
import matplotlib.pyplot as plt

import gtpsa
import os
import time
import copy
import random

from multiprocessing import Process, Queue, Pipe
from queue import Empty

from scipy.stats import truncnorm, norm



# t_file = Path(os.environ["HOME"]) / "Devel"/ "gitlab" / "dt4cc"/"lattices" / "b2_stduser_beamports_blm_tracy_corr.lat"
prefix = Path(os.environ["HOME"])
prefix = Path("/home/schnizer")
t_dir =  prefix / "Devel" / "gitlab" / "dt4acc" / "lattices"
t_file = t_dir / "b2_stduser_beamports_blm_tracy_corr_with_nlk.lat"

x_, px_ = 0, 1


emittance_start = 70e-9
nv = 6
mo = 1
default_desc = gtpsa.desc(nv, mo)


def create_nlk_interpolation(nlk_name):
    def compute_mirror_position_plate(ref_pos, mirror_pos, *, y_plane=True):
        assert y_plane
        dy = ref_pos.imag - mirror_pos.imag
        return ref_pos - 2 * dy * 1j

    # fmt: off
    ref_pos1 =  8e-3 +  7e-3j
    ref_pos2 = 17e-3 + 15e-3j
    # fmt: on
    t_current = -7e2

    # fmt: off
    t_current *= 1 - 1 * 0.14 / 2
    ref_pos1  *= 1 - 0.14
    ref_pos2  *= 1 - 0.14

    plate_position1 = 5e-3j
    mirror_pos1 = compute_mirror_position_plate(ref_pos1, plate_position1)

    inner = tslib.aircoil_filament(ref_pos1.real, ref_pos1.imag,  t_current)
    outer = tslib.aircoil_filament(ref_pos2.real, ref_pos2.imag, -t_current)
    mirror = tslib.aircoil_filament(mirror_pos1.real, mirror_pos1.imag, -t_current * 0.14)
    nlkf_intp = tslib.NonLinearKickerInterpolation([inner, outer, mirror])

    c = Config()
    c.setAny("L", 0e0)
    c.setAny("name", nlk_name)
    c.setAny("N", 1)
    nlk = tslib.FieldKick(c)
    nlk.set_field_interpolator(nlkf_intp)
    return nlk, nlkf_intp



def calulate_sigma_px(sigma_x, *, emittance=emittance_start):
    sigma_px = np.sqrt(emittance ** 2 - sigma_x ** 2)
    return sigma_px

def create_state_space_vector(*, mu_x=0e0, mu_px=0e0, desc=default_desc):
    ps = gtpsa.ss_vect_double(desc, mo, nv)
    #ps.set_identity()
    ps.set_zero()
    ps[x_]+=mu_x
    ps[px_]+=mu_px
    return ps
    
    t_x = ps[x_]
    t_x.set(0, mu_x)
    t_px = ps[px_]
    t_px.set(0, mu_px)
    ps[x_] = t_x
    ps[px_] = t_px
    return ps


def particle_propagation(index,acc_info,remaining_info):
    x_list, px_list, when_activate_NLK, rounds_to_save, kicker_strength = remaining_info 

        
    acc,calc_config,nlkfk,nlkf_in = acc_info #get accelerator information 
    mu_x = x_list[index]        #mu_x,mu_px are global variables
    mu_px = px_list[index]


    tp = acc_info #get accelerator information
    emittance_start = 70e-9


    mu_x_process = []      #save how the position changes during the rounds
    for runde in range(1000):
        if runde in rounds_to_save:      #due to memory problems we can't save every round
            mu_x_process.append(mu_x)

        if runde != when_activate_NLK:
            nlkf_intp.set_scale(0.0)

        elif runde == when_activate_NLK:
            nlkf_intp.set_scale(kicker_strength)
        else:
            raise ValueError("should not end up here")

        ps = create_state_space_vector(mu_x=mu_x,mu_px=mu_px)

        result = acc.propagate(calc_config, ps)
        assert result==len(acc) 

        #update mu_x and mu_px
        n_mu_x = ps.cst()[x_]
        n_mu_px = ps.cst()[px_]
        mu_x=n_mu_x
        mu_px=n_mu_px

        if n_mu_x>0.015:    #septum splint
            return False, mu_x_process
        elif n_mu_x<=0.015:
            continue
        else:
            return False,mu_x_process   #n_mu_x might be NAN, if the settings are extreme

    if 1000 in rounds_to_save:      #saving result of last round
            mu_x_process.append(mu_x)
    return True,mu_x_process

def run(idx, args, worker_acc_info,remaining_info):
    """
    Inputs:  -idx: Worker index
             -args: Job index
    """
    return particle_propagation(args, worker_acc_info)






class Particle_Propagator():
    """
    Particle Propagator


    Parameters
    ----------
    
    Attributes
    ----------
   
    ----------
    """

    def __init__(self,when_activate_NLK = 1):
        self.when_activate_NLK = when_activate_NLK
        
            
       
    def run_1000(self,x_list,px_list,when_activate_NLK=1,kicker_strength=1.0,rounds_to_save=[i for i in range(0,1000,10)]):
        assert len(x_list)==len(px_list)
        assert len(x_list)%25==0   #else we will get problems
        
        self.x_list = x_list
        self.px_list = px_list
        self.kicker_strength = kicker_strength
        self.when_activate_NLK = when_activate_NLK
        self.rounds_to_save = rounds_to_save
        
        a=time.time()
        pipe_list = self.run_jobs([('job', i) for i in range(0,len(x_list),25)], workers=20)
        print("Needed time:" time.time()-a)
        
        return self._analyze_pipe_list(pipe_list)

    
    
    def _analyze_pipe_list(self,pipe_list):
        idx2 = 0         #running index
        number_True = 0  #how many electrons made the 1000 circles
        number_Total = 0
        position_array = np.zeros((len(self.x_list),len(self.rounds_to_save)).astype("float64")  #array to save movement of the electrons

        for x in range(len(pipe_list)):
            for i in range(10000):
                try:
                    _,result,process = pipe_list[x].recv()

                    number_Total+=1
                    if result == True:
                        number_True+=1

                    position_array[idx2,:len(process)]=process
                    idx2+=1
                except EOFError:
                    #print(x,idx2)
                    pipe_list[x].close()
                    break
        return number_True,number_Total,position_array
    
    

    def _run_jobs(self,jobs, workers=1):

        q = Queue()  #Multiprocessing Queue to remember which jobs need to be done

        def worker(idx,q,send_end,remaining_info):
            """
            Each worker gets their individuall accelerator
            """
            acc = accelerator_from_config(t_file)
            calc_config = tslib.ConfigType()

            #Beschreibung von NLK
            nlkfk = acc.find("pkdnl1kr", 0)
            nlk_name = nlkfk.name
            _, nlkf_intp = create_nlk_interpolation(nlk_name)
            nlkfk.set_field_interpolator(nlkf_intp)
            assert(nlkfk.name == nlk_name)

            worker_acc_info = (acc,calc_config,nlkfk,nlkf_intp)   #compress all information


            try:
                while not q.empty():
                    args = q.get(timeout=1)    #has form ("Job", job_index)
                    print(args[1])

                    for running_idx in range(25): #let each worker run multiple runs at once, 
                                                  #to decrease time consumption

                        result, process = run(idx,args[1]+running_idx, worker_acc_info,remaining_info)

                        send_end.send((idx,result, np.array(process)*1000))    #send information to output pipe

            except Empty:    #just for safety reasons
                return True

        for job in jobs:
            q.put(job)

        pipe_list=[]    #list of input/output pipes
        processes = []  #list of all workers

        for i in range(0, workers):
            recv_end, send_end = Pipe(False)
            pipe_list.append(recv_end)
               
            remaining_info = (self.x_list,self.px_list,self.when_activate_NLK,self.rounds_to_save,self.kicker_strength)
            p = Process(target=worker, args=[i,q,send_end,remaining_info])
            p.daemon = True
            p.start()
            processes.append(p)

        for p in processes: 
            p.join()

        #need to return the pipe_list. analyzing it here gives errors
        return pipe_list    
    
    



__all__=["Particle_Propagator"]