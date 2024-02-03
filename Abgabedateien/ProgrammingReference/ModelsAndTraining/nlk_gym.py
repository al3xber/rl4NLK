import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.stats import norm

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#sampling class
class NLK_Sampler:
    #this class is used to sample new electrons
    def __init__(self):
        self.lower = np.array([0.01242802014328473,-2.3510967845422677,    #lower polynom weights, for injection area
                                 135.64889234592374,-2621.6674111929387,8374.6738385004])   
        self.upper = np.array([-0.026705410090359233,6.412451482988342,  #upper polynom weights, for injection area
                                 -549.8383734808201,20891.368558370945,-298723.030031348])
        self.diff = self.upper-self.lower
        self.transform = np.array([-3.2911361375977704,102.00328087158005,
                                   15593.102791660853,-518062.7698926044])
        self.poly = PolynomialFeatures(degree=4)
        self.poly_transform = PolynomialFeatures(degree=3)
    def sample(self):
        x = np.random.uniform(size=(1,1))
        x = x**1.8       #fixing term, not known why to use
        x = x*(22.85e-3-15e-3)+15e-3

        x_poly = self.poly_transform.fit_transform(x)
        x = x_poly@self.transform
        x = x*(22.85e-3-15e-3)+15e-3
        x = np.array([[x[0]]])
        
        height = np.random.uniform()
        x_poly = self.poly.fit_transform(x)   #to x^0,x^1,...,x^4
        y = x_poly@self.lower + height*x_poly@self.diff
        return x[0,0],y[0]    #not normalized 



class NLK_Env(gym.Env):
    def __init__(self,reward_design_1 = 5, reward_design_2 = 5, deterministic=False):
        super(NLK_Env, self).__init__()
        self.reward_design_1 = reward_design_1
        self.reward_design_2 = reward_design_2
        
        #information on electrons. We initialise randomly
        self.x = 0.018    
        self.px = 0.0
        self.round = 0
        
        self.sampler = NLK_Sampler()
        
        #information for kicker usage
        self.activated_NLK = False
        self.round_survived_array = np.load("Zone_array5.npy")[0]
        self.optimal_NLK_strength_array = np.load("Ztwo_array5.npy")[0]
        
        self.x_list = np.linspace(-100.0e-3,90e-3,1280)
        self.px_list = np.linspace(-12e-3,7e-3,640)
        
        self.x_min,self.x_max, self.x_len     = self.x_list[0], self.x_list[-1], len(self.x_list)
        self.px_min, self.px_max, self.px_len = self.px_list[0], self.px_list[-1], len(self.px_list)
        
                
        
        #information for round to round behaviour
        self.roundX = np.load("ZroundX.npy").T
        self.roundPX = np.load("ZroundPX.npy").T
                
        self.x_list2 = np.linspace(-45e-3,24.5e-3,1280)
        self.px_list2 = np.linspace(-0.003,0.003,1280)
        
        self.x_min2,self.x_max2, self.x_len2     = self.x_list2[0], self.x_list2[-1], len(self.x_list2)
        self.px_min2, self.px_max2, self.px_len2 = self.px_list2[0], self.px_list2[-1], len(self.px_list2)
        
        
        
        
        #noises
        self.noise_first_round = 65e-6
        self.noise_x = 6.5e-6
        self.noise_px = .8*6.5e-6
        self.noise_NLK = 0.0125
        
        if deterministic:
            self.noise_first_round = 0
            self.noise_x = 0
            self.noise_px = 0
            self.noise_NLK = 0
        
        #normalizations
        self.reward_normalization = 1000
        self.x_normalization = 100
        self.px_normalization = 1000
        self.round_normalization = 100
        
        # 1-tupel for possible action: kicker strength
        self.action_space = spaces.Box(low = -1.0, high = 1.0, dtype = np.float64)
        
        # Observation space contains round, x/px values normalized
        self.observation_space = spaces.Box(low=np.array([0,-5.0, -5.0]), 
                                            high=np.array([10.0,5.0, 5.0]), dtype=np.float64)

    def reset(self, seed = None, options = None):
        self.round = 0   #set round to 0
        self.activated_NLK = False   #kicker not activated
        if options == None:
            self.x, self.px = self.sampler.sample()
            
        else:
            assert type(options)==dict
            if "x,px" in options.keys():
                self.x, self.px = options["x,px"]
            else:
                raise Exception("Options wrong")
        self.x *= self.x_normalization     #normalise values
        self.px *= self.px_normalization

        return np.array([0,self.x  ,self.px]),{}

    def step(self, action): 
        NLK_activated = (abs(action[0])>0.16)              #see strength_NLK
        NLK_strength = np.sign(action[0])*(action[0]**4)   #NLK strength 
        
        # Move the agent based on the selected action
        if self.round == 1000:  #terminal condition
            done = True
            return np.array([self.round, self.x,self.px]), 1000/self.reward_normalization, done,False, {} 
        
        elif self.round == 0:  #add noise
            noise_x_sample = np.random.normal(0,self.noise_first_round)
            noise_px_sample = np.random.normal(0,self.noise_first_round)
        else:   
            noise_x_sample = np.random.normal(0,self.noise_x)
            noise_px_sample = np.random.normal(0,self.noise_px)
        
        x_normalized = self.x/self.x_normalization     #normalise
        px_normalized = self.px/self.px_normalization 
        
        x_normalized += noise_x_sample       #add noise
        px_normalized += noise_px_sample

        self.round += 0.1    #added round with normalisation
        
        if NLK_activated==False:   #if kicker was not activated
            
            idx_x  = np.floor(self.x_len2*((x_normalized-self.x_min2)/(self.x_max2-self.x_min2))).astype("int")  #get index of x
            idx_px = np.floor(self.px_len2*((px_normalized-self.px_min2)/(self.px_max2-self.px_min2))).astype("int") #get index of px
            
            
            if (idx_x >= 0) and (idx_x < self.x_len2 - 1) and (idx_px >= 0) and (idx_px < self.px_len2 - 1):  #check if in data
                
                where_nan = np.vstack([self.roundX[idx_px,  idx_x],
                                         self.roundX[idx_px,  idx_x+1],
                                         self.roundX[idx_px+1,idx_x],
                                         self.roundX[idx_px+1,idx_x+1],
                                         self.roundPX[idx_px,  idx_x],
                                         self.roundPX[idx_px,  idx_x+1],
                                         self.roundPX[idx_px+1,idx_x],
                                         self.roundPX[idx_px+1,idx_x+1]]).T
                where_nan = np.sum(where_nan) 
                if np.isnan(where_nan):  #check if it somewhere none. If so consider electron lost
                    done = True
                    return np.array([self.round, self.x,self.px]), 0, done, False, {} 
                
                
                
                
                
                point = np.array([x_normalized, px_normalized])
                   
                #calculate distances
                distance_matrix = np.array([np.sum((point-
                                             np.array([self.x_list2[idx_x],self.px_list2[idx_px]]))**2,axis=0),
                                     np.sum((point-
                                             np.array([self.x_list2[idx_x],self.px_list2[idx_px+1]]))**2,axis=0),
                                     np.sum((point-
                                             np.array([self.x_list2[idx_x+1],self.px_list2[idx_px]]))**2,axis=0),
                                     np.sum((point-
                                             np.array([self.x_list2[idx_x+1],self.px_list2[idx_px+1]]))**2,axis=0)]).T


                distance_matrix = np.abs(distance_matrix)
                distance_matrix = distance_matrix/np.sum(distance_matrix)
                #points with smaller distance are more important
                distance_matrix = 1-distance_matrix
                distance_matrix = distance_matrix/np.sum(distance_matrix)

                
                predicted_x = np.hstack([self.roundX[idx_px,  idx_x],
                                         self.roundX[idx_px+1,  idx_x],
                                         self.roundX[idx_px,idx_x+1],
                                         self.roundX[idx_px+1,idx_x+1]])
                predicted_px = np.hstack([self.roundPX[idx_px,  idx_x],
                                         self.roundPX[idx_px+1,  idx_x],
                                         self.roundPX[idx_px,idx_x+1],
                                         self.roundPX[idx_px+1,idx_x+1]])
                
                assert distance_matrix.shape == predicted_x.shape
                
                
                
                

                predicted_x = np.sum(distance_matrix * predicted_x)
                predicted_px = np.sum(distance_matrix * predicted_px)
            
            
                
                self.x = predicted_x * self.x_normalization
                self.px = predicted_px * self.px_normalization
            
            
                if self.x > 0.015 * self.x_normalization: #at 15mm septum. If outside consider lost 
                    done = True
                    return np.array([self.round, self.x,self.px]), 0, done, False, {} 
            else:   #außerhalb der region
                done = True
                return np.array([self.round, self.x,self.px]), 0, done, False, {} 
            
            done = False
            return np.array([self.round, self.x,self.px]), 0, done, False, {} 
        else:   #if kicker is activated
            assert self.activated_NLK==False  #NLK can only be activated once
            self.activated_NLK = True
            
            idx_x = np.floor((x_normalized-self.x_min)*(self.x_len/(-self.x_min+self.x_max))).astype("int")
            idx_px = np.floor((px_normalized-self.px_min)*(self.px_len/(-self.px_min+self.px_max))).astype("int")

            if idx_x>= 0 and idx_x<self.x_len-1 and idx_px>= 0 and idx_px<self.px_len-1:
                if np.sum((self.round_survived_array[int(idx_px):int(idx_px+2),int(idx_x):int(idx_x+2)])==1000)>1:
                    #optimal kicker strength calculation 
                    point = np.array([x_normalized, px_normalized])
                    
                    distance_matrix = np.array([np.sum((point-
                                                 np.array([self.x_list[idx_x],self.px_list[idx_px]]))**2,axis=0),
                                         np.sum((point-
                                                 np.array([self.x_list[idx_x],self.px_list[idx_px+1]]))**2,axis=0),
                                         np.sum((point-
                                                 np.array([self.x_list[idx_x+1],self.px_list[idx_px]]))**2,axis=0),
                                         np.sum((point-
                                                 np.array([self.x_list[idx_x+1],self.px_list[idx_px+1]]))**2,axis=0)]).T
                    
                    
                    distance_matrix = np.abs(distance_matrix)
                    distance_matrix = distance_matrix/np.sum(distance_matrix)
                    #points with smaller distance are more important
                    distance_matrix = 1-distance_matrix
                    distance_matrix = distance_matrix/np.sum(distance_matrix)
                    
                    
                    
                    
                    optimal_NLK_strength_matrix = np.array([self.optimal_NLK_strength_array[idx_px,  idx_x],
                                         self.optimal_NLK_strength_array[idx_px+1,  idx_x],
                                         self.optimal_NLK_strength_array[idx_px,idx_x+1],
                                         self.optimal_NLK_strength_array[idx_px+1,idx_x+1]]).T   
                    
                    assert distance_matrix.shape==optimal_NLK_strength_matrix.shape
                    
                    optimal_kicker_strength = np.sum(distance_matrix * optimal_NLK_strength_matrix) 
                    
                    #add noise to kicker action
                    noise_NLK_sample = np.random.normal(0,self.noise_NLK)  #eine null mehr 
                    
                    #approximation of how many electrons out of 1000 would survive the injection
                    reward = 985*np.exp(-(14.5*(NLK_strength+noise_NLK_sample-optimal_kicker_strength))**4)
                                                           
                    reward = reward/self.reward_normalization
                    done = True

                    return np.array([self.round, self.x,self.px]), reward, done, False, {}   
            
            #if it is outside the kickable area, but the kicker was activated, consider electron lost
            done = True
            
            reward = (self.reward_design_1-self.reward_design_2*np.abs(NLK_strength)) /self.reward_normalization     #to get Kickstrength closer to 0.   See reward design section 
            
            return np.array([self.round, self.x,self.px]), reward, done, False, {}      
        
        
    def render(self):
        return self.round, self.x, self.px
