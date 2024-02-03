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


def encoder(points, x_normalization, px_normalization, normalized = True):
    #assert points are normalized!
    if not normalized:
        points[0,:]*=x_normalization
        points[1,:]*=px_normalization
        
    if (points.shape)[1] == 0:
        #check if points are contained
        return np.float32(np.zeros((2,))), np.uint8(np.zeros((1,51,51)))
    
    means = points.mean(axis = 1, dtype=np.float32)
    assert means.shape == (2,)
    
    range_xpx = np.array([0.006*x_normalization,0.0005*px_normalization])
    
    #create a histogram 
    H,_,_ = np.histogram2d(points[0,:],points[1,:], bins=51, range = np.vstack([means-range_xpx,means+range_xpx]).T)
    
    
    if np.sum(H)!= 0:  #assert there is data
        H = ((H/np.sum(H))*255).astype(np.uint8).T
    
    return means, H[None]   #[None] for (1,51,51) shape
    
class NLK_Env(gym.Env):
    def __init__(self, deterministic = False):
        super(NLK_Env, self).__init__()

        self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float64) #1st index equals round, 2nd kicker_strength
        
        # Observation space contains  x/px values in cm and histogram plot 
        self.observation_space = spaces.Dict(
                                    spaces = {
                                        "x,px mean": gym.spaces.Box(-3, 3, (2,), dtype=np.float32),
                                        "image": gym.spaces.Box(0, 255, (1,51,51), dtype=np.uint8)
                                        }
                                    )
        #round information
        self.round = 0
        #distribution information
        self.x_mean = 0.018
        self.px_mean = 0.0
        self.sigma_x = 0.00112
        self.sigma_px = 6.29e-05
        
        
        x_points = norm.rvs(loc=self.x_mean, scale=self.sigma_x, size=1000)  #generate random x points
        px_points = norm.rvs(loc=self.px_mean, scale=self.sigma_px, size=1000)  #generate px points 
        
        #normalizations
        self.reward_normalization = 1000
        self.x_normalization = 100
        self.px_normalization = 1000
        
        #normalize points
        x_points = x_points * self.x_normalization
        px_points = px_points * self.px_normalization
        
        self.x = x_points
        self.px = px_points
        self.points = np.vstack([x_points,px_points])   #no correlation is used
        self.points = self.points[:,self.points[0,:]>0.015 * self.x_normalization]   # remove points that are inside of septum
          
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
        if deterministic:
            self.noise_first_round, self.noise_x, self.noise_px, self.noise_NLK = 0, 0, 0, 0
        else:
            self.noise_first_round = 65e-6
            self.noise_x = 6.5e-6
            self.noise_px = .8*6.5e-6
            self.noise_NLK = 0.0125
        
        
        #sampler
        self.sampler = NLK_Sampler()
        
        
    def reset(self, seed = None, options = None):
        self.activated_NLK = False   
        
        points_are_given = False  
        if options is None:
            self.x, self.px = self.sampler.sample()
            x_points = norm.rvs(loc=self.x, scale = self.sigma_x, size = 1000) * self.x_normalization  
            px_points = norm.rvs(loc=self.px, scale=self.sigma_px, size = 1000) * self.px_normalization  

        else:
            assert type(options)==dict
            if "x,px" in options.keys():
                self.x, self.px = options["x,px"]
                #mean values are given. generate new x/px values and normalize them
                x_points = norm.rvs(loc=self.x, scale = self.sigma_x, size = 1000) * self.x_normalization  
                px_points = norm.rvs(loc=self.px, scale=self.sigma_px, size = 1000) * self.px_normalization  
            elif "points" in options.keys():   #points are given
                points_are_given = True
                x_points = options["points"][0,:] * self.x_normalization  
                px_points = options["points"][1,:] * self.px_normalization  
            else:
                raise Exception("Options wrong")
        self.x = x_points
        self.px = px_points
        self.points = np.vstack([x_points,px_points])   #no correlation
        if not points_are_given:
            self.points = self.points[:,self.points[0,:] > 0.015 * self.x_normalization]   # remove points that are inside of septum
        
        #encode the points
        xpx_mean, image = encoder(self.points, self.x_normalization, self.px_normalization)
        
        out_dict = {"x,px mean": xpx_mean,
                    "image": image}
        out_info = {}
        
        return out_dict, out_info
    
    def check_if_empty(self, reward = 0, xpx_mean = np.array([0.0,0.0])):
        #function that checks if self.points or self.x contains points and if no points are left, returns gym output
        if (self.points.shape)[1] == 0 or len(self.x) == 0:
            out_dict={"x,px mean" : np.float32(xpx_mean),
                      "image" : np.uint8(np.zeros((1,51,51)))}
            reward = reward
            truncated = False
            terminated = True
            info = {}
        
            return out_dict, reward, terminated, truncated, info 
        return False
    
    def step(self, action):
        activation_round = (action[0]+1)*5 # -> (-1,-.8)->0, (-.8,-.6)->1, (-.6,-.4)->2, (-.4,-.2)->3, (-.2,0)->4
        probability_for_next = activation_round-np.floor(activation_round)
        activation_round = int(np.floor(activation_round))+np.random.binomial(1, probability_for_next)

        terminated = False

        for runde in range(activation_round):
            out_dict, reward, terminated, truncated, info = self._step(0.0)
            if terminated:
                return out_dict, reward, terminated, truncated, info
        
        out_dict, reward, terminated, truncated, info = self._step(action[1])
        terminated = True   #in case action strength is 0.0
        
        return out_dict, reward, terminated, truncated, info
           

    
    def _step(self, action): 
        check = self.check_if_empty()   #for safety reasons. Should not be necessary
        if check != False: 
            return check
        
        NLK_activated = (abs(action)>0.0)              #see strength_NLK
        NLK_strength = action                          #np.sign(action[0])*(action[0]**4)   #NLK strength 
        
        
        if self.round == 0:
            #in round 0 we add extra noise
            noise_x_sample = np.random.normal(0,self.noise_first_round)
            noise_px_sample = np.random.normal(0,self.noise_first_round)
        else:   
            noise_x_sample = np.random.normal(0,self.noise_x)
            noise_px_sample = np.random.normal(0,self.noise_px)
        
        #normalize and add noise 
        self.x = self.points[0,:]/self.x_normalization
        self.px = self.points[1,:]/self.px_normalization
        
        self.x += noise_x_sample     
        self.px += noise_px_sample
        
        #add round to round counter
        self.round += 1
        
        if NLK_activated==False:
            #get indices within round to round behaviour files
            idx_x  = np.floor(self.x_len2*((self.x-self.x_min2)/(self.x_max2-self.x_min2))).astype("int")
            idx_px = np.floor(self.px_len2*((self.px-self.px_min2)/(self.px_max2-self.px_min2))).astype("int")
            
            #remove points that are out of the valid area
            valid_idx = (idx_x >= 0)*(idx_x < self.x_len2 - 1)*(idx_px >= 0)*(idx_px < self.px_len2 - 1)
            
            self.x = self.x[valid_idx]
            self.px = self.px[valid_idx]
            idx_x = idx_x[valid_idx]
            idx_px = idx_px[valid_idx]
            
            
            #remove points that contain nan
            where_nan = np.vstack([self.roundX[idx_px,  idx_x],
                                         self.roundX[idx_px,  idx_x+1],
                                         self.roundX[idx_px+1,idx_x],
                                         self.roundX[idx_px+1,idx_x+1],
                                         self.roundPX[idx_px,  idx_x],
                                         self.roundPX[idx_px,  idx_x+1],
                                         self.roundPX[idx_px+1,idx_x],
                                         self.roundPX[idx_px+1,idx_x+1]]).T
            where_nan = np.sum(where_nan, axis=1) 
            valid_idx = (1 - np.isnan(where_nan)).astype(bool)   
            
            self.x = self.x[valid_idx]
            self.px = self.px[valid_idx]
            idx_x = idx_x[valid_idx]
            idx_px = idx_px[valid_idx]
            self.points = np.vstack([self.x,self.px])
            
            #check if there are still points left. If no electrons are left, the points are out of the valid area
            check = self.check_if_empty()
            if check != False: 
                return check
            
            #calculate for each point the distance to the individual corner points
            distance_matrix = np.vstack([np.sum(np.abs(self.points-
                                                 np.vstack([self.x_list2[idx_x],self.px_list2[idx_px]])),axis=0),
                                         np.sum(np.abs(self.points-
                                                 np.vstack([self.x_list2[idx_x],self.px_list2[idx_px+1]])),axis=0),
                                         np.sum(np.abs(self.points-
                                                 np.vstack([self.x_list2[idx_x+1],self.px_list2[idx_px]])),axis=0),
                                         np.sum(np.abs(self.points-
                                                 np.vstack([self.x_list2[idx_x+1],self.px_list2[idx_px+1]])),axis=0)]).T
                                         
            #normalize
            distance_matrix = distance_matrix/np.sum(distance_matrix,axis=1)[:,None]
            #points with smaller distance are more important
            distance_matrix = 1-distance_matrix
            distance_matrix = distance_matrix/np.sum(distance_matrix,axis=1)[:,None]
            
            assert np.isclose(np.sum(distance_matrix[0]),1.0)
            assert distance_matrix.shape == (len(self.x),4)
        
            predicted_x = np.vstack([self.roundX[idx_px,  idx_x],
                                         self.roundX[idx_px+1,  idx_x],
                                         self.roundX[idx_px,idx_x+1],
                                         self.roundX[idx_px+1,idx_x+1]]).T
            predicted_px = np.vstack([self.roundPX[idx_px,  idx_x],
                                         self.roundPX[idx_px+1,  idx_x],
                                         self.roundPX[idx_px,idx_x+1],
                                         self.roundPX[idx_px+1,idx_x+1]]).T
            
            assert distance_matrix.shape == predicted_x.shape
            
            predicted_x = np.sum(distance_matrix * predicted_x, axis = 1)
            predicted_px = np.sum(distance_matrix * predicted_px, axis = 1)
            
            #normalize points
            self.points = np.vstack([predicted_x * self.x_normalization, predicted_px * self.px_normalization])
            
            #remove points that are outside of the septum
            self.points = self.points[:, self.points[0,:] < 0.015 * self.x_normalization]
            
            check = self.check_if_empty()    #if no electrons survived, all electrons crashed against the septum
            if check != False: 
                return check
            
            #if this point is reached, some electrons have survived being not kicked
            #therefore we return the encoded points and terminated = False.
            xpx_mean, image = encoder(self.points, self.x_normalization, self.px_normalization)
            
            out_dict={"x,px mean": xpx_mean,
                      "image": image}
            reward = 0/self.reward_normalization
            truncated = False
            terminated = False
            info = {}
            return out_dict, reward, terminated, truncated, info
        
        else:   #NLK is activated
            assert self.activated_NLK==False  #NLK can only be activated once
            self.activated_NLK = True
            
            #get indices within NLK usage files
            idx_x  = np.floor(self.x_len*((self.x-self.x_min)/(self.x_max-self.x_min))).astype("int")
            idx_px = np.floor(self.px_len*((self.px-self.px_min)/(self.px_max-self.px_min))).astype("int")
            
            #remove points that are out of the valid area
            valid_idx = (idx_x >= 0)*(idx_x < self.x_len - 1)*(idx_px >= 0)*(idx_px < self.px_len - 1)  
            
            self.x = self.x[valid_idx]
            self.px = self.px[valid_idx]
            idx_x = idx_x[valid_idx]
            idx_px = idx_px[valid_idx]
            
            #check if there are still points left. If no electrons are left, the points are out of the valid area
            check = self.check_if_empty()
            if check != False: 
                return check
            
            rounds_survived = np.vstack([self.round_survived_array[idx_px,  idx_x],
                                         self.round_survived_array[idx_px,  idx_x+1],
                                         self.round_survived_array[idx_px+1,idx_x],
                                         self.round_survived_array[idx_px+1,idx_x+1]]).T
            rounds_survived = np.sum(rounds_survived == 1000,axis=1) 
            valid_idx = rounds_survived > 1  
            #remove points where no neighbor point has survived 1000 rounds
            x_mean = np.mean(self.x)
            px_mean = np.mean(self.px)
            
            self.x = self.x[valid_idx]
            self.px = self.px[valid_idx]
            idx_x = idx_x[valid_idx]
            idx_px = idx_px[valid_idx]

            self.points = np.vstack([self.x,self.px])
            
            #check if there are still points left. If no electrons are left, NLK has been activated outside of Kicker area
            if self.check_if_empty() != False:                 
                reward = 0
                return self.check_if_empty(reward = reward, xpx_mean = np.array([x_mean*self.x_normalization, 
                                                                                 px_mean*self.px_normalization]))
            
            #optimal kicker strength calculation
            assert self.points.shape == np.vstack([self.x_list[idx_x],self.px_list[idx_px]]).shape
            #calulate distance to each neighbour in the grid
            distance_matrix = np.vstack([np.sum((self.points-
                                                 np.vstack([self.x_list[idx_x],self.px_list[idx_px]]))**2,axis=0),
                                         np.sum((self.points-
                                                 np.vstack([self.x_list[idx_x],self.px_list[idx_px+1]]))**2,axis=0),
                                         np.sum((self.points-
                                                 np.vstack([self.x_list[idx_x+1],self.px_list[idx_px]]))**2,axis=0),
                                         np.sum((self.points-
                                                 np.vstack([self.x_list[idx_x+1],self.px_list[idx_px+1]]))**2,axis=0)]).T
                                         
            #normalize
            distance_matrix = distance_matrix/np.sum(distance_matrix,axis=1)[:,None]
            #points with smaller distance are more important
            distance_matrix = 1-distance_matrix
            distance_matrix = distance_matrix/np.sum(distance_matrix,axis=1)[:,None]
            
            assert np.isclose(np.sum(distance_matrix[0]),1.0)
            assert distance_matrix.shape == (len(self.x),4)
            
            #get for each electron the optimal Kicker strength
            optimal_NLK_strength_matrix = np.vstack([self.optimal_NLK_strength_array[idx_px,  idx_x],
                                         self.optimal_NLK_strength_array[idx_px+1,  idx_x],
                                         self.optimal_NLK_strength_array[idx_px,idx_x+1],
                                         self.optimal_NLK_strength_array[idx_px+1,idx_x+1]]).T   #transposed wichtig!
            
            assert distance_matrix.shape == optimal_NLK_strength_matrix.shape
            
            optimal_kicker_strength = np.sum(distance_matrix * optimal_NLK_strength_matrix,axis = 1)  
            
            #add noise
            noise_NLK_sample = np.random.normal(0,self.noise_NLK)  
            
            #calculate reward
            reward = ((self.points.shape[1])/1000)*(985/1000)*np.sum(np.exp(-(14.5*(NLK_strength+noise_NLK_sample-optimal_kicker_strength))**4)) / self.reward_normalization

            #normalize points
            self.points[0,:] *= self.x_normalization
            self.points[1,:] *= self.px_normalization
                
            #if this point is reached, some electrons can be successfully kicked
            #therefore we return the encoded points, the reward and terminated = True.
            xpx_mean, image = encoder(self.points, self.x_normalization, self.px_normalization)
            
            out_dict={"x,px mean": xpx_mean,
                      "image": image}
            truncated = False
            terminated = True
            info = {}
            return out_dict, reward, terminated, truncated, info
                
    def render(self):
        
        return True
