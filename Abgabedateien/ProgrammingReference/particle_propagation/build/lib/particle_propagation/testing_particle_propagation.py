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

from multiprocessing import Process, Queue, Array
from queue import Empty

from scipy.stats import truncnorm, norm



# t_file = Path(os.environ["HOME"]) / "Devel"/ "gitlab" / "dt4cc"/"lattices" / "b2_stduser_beamports_blm_tracy_corr.lat"
prefix = Path(os.environ["HOME"])
prefix = Path("/home/schnizer")
t_dir =  prefix / "Devel" / "gitlab" / "dt4acc" / "lattices"
t_file = t_dir / "b2_stduser_beamports_blm_tracy_corr_with_nlk.lat"

x_, px_ = 0, 1
y_, py_ = 2, 3

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
    assert sigma_x <=emittance_start

    
    sigma_px = np.sqrt(emittance ** 2 - sigma_x ** 2)
    return sigma_px

def emmitance_propagation(position_array):
    return np.mean(position_array,0),np.std(position_array,0)

def create_state_space_vector(*, mu_x=0e0, mu_px=0e0, mu_y=0e0, mu_py=0e0, desc=default_desc):
    ps = gtpsa.ss_vect_double(desc, mo, nv)
    #ps.set_identity()
    ps.set_zero()
    ps[x_]+=mu_x
    ps[px_]+=mu_px
    ps[y_]+=mu_y
    ps[py_]+=mu_py
    return ps
    
    t_x = ps[x_]
    t_x.set(0, mu_x)
    t_px = ps[px_]
    t_px.set(0, mu_px)
    ps[x_] = t_x
    ps[px_] = t_px
    
    t_y = ps[y_]
    t_y.set(0, mu_y)
    t_py = ps[py_]
    t_py.set(0, mu_py)
    ps[y_] = t_y
    ps[py_] = t_py
    
    return ps


def particle_propagation(index,acc_info,remaining_info):
    x_list, px_list, y_list, py_list,when_activate_NLK, rounds_to_save, y_rounds_to_save, kicker_strength = remaining_info 
    
        
    acc,calc_config,nlkfk,nlkf_intp = acc_info #get accelerator information 
    mu_x = x_list[index]        #mu_x,mu_px are global variables
    mu_px = px_list[index]
    mu_y = y_list[index]        #mu_x,mu_px are global variables
    mu_py = py_list[index]
    
    
    #tp = acc_info #get accelerator information
    emittance_start = 70e-9

    
    mu_x_process = []      #save how the position changes during the rounds
    mu_y_process = []
    for runde in range(1000):
        if runde in rounds_to_save:      #due to memory problems we can't save every round
            mu_x_process.append(mu_x)
            
        if runde in y_rounds_to_save:
            mu_y_process.append(mu_px)          #MU Y is MU PX!!!!!
            
        if runde != when_activate_NLK:
            nlkf_intp.set_scale(0.0)

        elif runde == when_activate_NLK:
            nlkf_intp.set_scale(kicker_strength)
        else:
            raise ValueError("should not end up here")

        ps = create_state_space_vector(mu_x=mu_x,mu_px=mu_px,mu_y=mu_y,mu_py=mu_py)    #x px y py   #TO PLOT!
        
        if runde == 0:                                                   #CHANGES!!!!!!!!!!!!!!!!!!!!!
            acc.propagate(calc_config, ps,169,len(acc)-169)

        else:
            
            result = acc.propagate(calc_config, ps)
            assert result==len(acc) 

        #update mu_x and mu_px
        n_mu_x = ps.cst()[x_]
        n_mu_px = ps.cst()[px_]
        mu_x=n_mu_x
        mu_px=n_mu_px
        
        n_mu_y = ps.cst()[y_]
        n_mu_py = ps.cst()[py_]
        mu_y=n_mu_y
        mu_py=n_mu_py
        #print(ps.cst())
        if n_mu_x>0.015:    #septum splint
            
            return False, mu_x_process, mu_y_process, (x_list[index],px_list[index])
        elif n_mu_x<=0.015:
            continue
        else:
            #print("NAN PROBLEM")
            return False,mu_x_process, mu_y_process, (x_list[index],px_list[index])   #n_mu_x might be NAN, if the settings are extreme

    if 1000 in rounds_to_save:      #saving result of last round
            mu_x_process.append(mu_x)
    return True, mu_x_process, mu_y_process, (x_list[index],px_list[index])





def single_particle(x,px,y=0,py=0,when_activate_NLK=0,kicker_strength=1.0,rounds_to_save=range(0,1000,10),y_rounds_to_save=[]):
    remaining_info = [x], [px], [y], [py], when_activate_NLK, rounds_to_save, y_rounds_to_save, kicker_strength

     
    acc = accelerator_from_config(t_file)
    calc_config = tslib.ConfigType()

    #Beschreibung von NLK
    nlkfk = acc.find("pkdnl1kr", 0)
    nlk_name = nlkfk.name
    _, nlkf_intp = create_nlk_interpolation(nlk_name)
    nlkfk.set_field_interpolator(nlkf_intp)
    assert(nlkfk.name == nlk_name)

    acc_info = (acc,calc_config,nlkfk,nlkf_intp)   #compress all information
    
    return particle_propagation(0,acc_info,remaining_info)
    




def run(idx, args, worker_acc_info,remaining_info):
    """
    Inputs:  -idx: Worker index
             -args: Job index
    """
    return particle_propagation(args, worker_acc_info,remaining_info)






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
        
            
       
    def run_1000(self,x_list,px_list,y_list=None,py_list=None,when_activate_NLK=1,kicker_strength=1.0,
                 rounds_to_save=[i for i in range(0,1000,10)],y_rounds_to_save=[]):
        assert len(x_list)==len(px_list)
        assert len(x_list)%25==0   #else errors occur
        
        self.x_list = x_list
        self.px_list = px_list
        self.kicker_strength = kicker_strength
        self.when_activate_NLK = when_activate_NLK
        self.rounds_to_save = rounds_to_save
        self.y_rounds_to_save = y_rounds_to_save
        
        if y_list is None: self.y_list = np.zeros(len(x_list))
        else: self.y_list = y_list
        if py_list is None: self.py_list = np.zeros(len(x_list))
        else: self.py_list = py_list
            
        a=time.time()
        result = self._run_jobs([('job', i) for i in range(0,len(x_list),25)], workers=20)
        print("Needed time:", time.time()-a)
        
        return result


    def _run_jobs(self,jobs, workers=1):

        q = Queue()  #Multiprocessing Queue to remember which jobs need to be done

        def worker(idx,q,queue_worker,queue_finish,remaining_info):
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
                while True:
                    
                    args = q.get(timeout=1)    #has form ("Job", job_index)

                    for running_idx in range(25): #let each worker run multiple runs at once, 
                                                  #to decrease time consumption
                        #print(f"w {idx:02d} start {args[1]+running_idx}")
                        result, process, y_process, start = run(idx,args[1]+running_idx, worker_acc_info,remaining_info)
                        #print(f"w {idx:02d} ended run {args[1]+running_idx}")
                        queue_worker.put((idx,result, np.array(process)*1000,
                                          np.array(y_process)*1000,start),timeout = 15)#send information to output pipe
                        #print(f"w {idx:02d} end {args[1]+running_idx}")
            except Empty:    #if q is empty
                queue_finish.put((idx),timeout = 1)
                queue_worker.close()
                return 
            
            print(f"worker {idx:02d}, WHY END UP HERE??-------------------------------")
        
        def information_extractor_worker(queue_finish, worker_queue_list,position_array,
                                         y_position_array,result_list,start_list, args):
            finished_pipes_counter = 0            
            idx = 0         #running index

            while finished_pipes_counter < len(worker_queue_list) or idx<args[0]: #args[0] = number of jobs to be done
                #print(f"amount finished: {idx}")
                #Check if workers are still working
                while finished_pipes_counter < len(worker_queue_list):
                    try:                                              # while not empty, get all finished workers
                        queue_finish.get(block=True, timeout=.01)     #if not empty -> worker finished
                        finished_pipes_counter+=1
                    except Empty:
                        break
                
                #empty worker queues
                for x in range(len(worker_queue_list)):
                    while True:
                        try:
                            _,result,process,y_process,start = worker_queue_list[x].get(block=True, timeout=.1)

                            result_list[idx] = float(result)
                            start_list[idx*2:(idx+1)*2] = start      #start is 2 dimensional

                            position_array[idx * args[1]:(idx * args[1]) + len(process)]=process #processes are len(process) dim
                            y_position_array[idx * args[2]:(idx * args[2]) + len(y_process)]=y_process
                            idx+=1
                        except Empty:
                            break
            
            queue_finish.close()
            return 
        
        
        
        for job in jobs:
            q.put(job)

        worker_queue_list=[]    #list of input/output pipes
        processes = []          #list of all workers
        
        queue_finish = Queue()

        for i in range(0, workers):
            queue_worker = Queue()
            worker_queue_list.append(queue_worker)
            
            remaining_info = (self.x_list,self.px_list,self.y_list,self.py_list,self.when_activate_NLK,self.rounds_to_save,
                              self.y_rounds_to_save,self.kicker_strength)
            p = Process(target=worker, args=[i,q,queue_worker,queue_finish,remaining_info])
            p.daemon = True
            p.start()
            processes.append(p)
        
        #Create variables
        unshared_arr = np.zeros(len(self.x_list)*len(self.rounds_to_save))
        position_array = Array('d', unshared_arr)
        
        unshared_arr2 = np.zeros(len(self.x_list))
        result_list = Array('d', unshared_arr2)
        
        unshared_arr3 = np.zeros(len(self.x_list)*2)
        start_list = Array('d', unshared_arr3)
        
        unshared_arr4 = np.zeros(len(self.x_list)*len(self.y_rounds_to_save))
        y_position_array = Array('d', unshared_arr4)
        
        p = Process(target=information_extractor_worker, args=[queue_finish, worker_queue_list,position_array,
                                                               y_position_array,result_list,
                                                               start_list, (len(self.x_list),len(self.rounds_to_save),
                                                                            len(self.y_rounds_to_save))])
        p.daemon = True
        p.start()
        p.join()
        
        '''
        PIDs = {p.pid:count for count,p in enumerate(processes)}
        while len(PIDs)!=0:
            for p in processes: 
                p.join(timeout = 1)
                if p.exitcode is not None:
                    try:
                        del PIDs[p.pid]
                    except KeyError:
                        pass
                print("PIDs LIST: ",PIDs)    
        ''' 

        #Loading all variables
        position_arr = np.frombuffer(position_array.get_obj())
        position_arr = position_arr.reshape((len(self.x_list),len(self.rounds_to_save)))
        
        results = np.frombuffer(result_list.get_obj())
        
        starts = np.frombuffer(start_list.get_obj())
        starts = starts.reshape((len(self.x_list),2))
        
        y_position_arr = np.frombuffer(y_position_array.get_obj())
        y_position_arr = y_position_arr.reshape((len(self.x_list),len(self.y_rounds_to_save)))

        return results,position_arr,y_position_arr,starts
    
    



__all__=["Particle_Propagator","calulate_sigma_px","emmitance_propagation","single_particle"]