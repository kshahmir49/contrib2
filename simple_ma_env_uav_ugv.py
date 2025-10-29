import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import PIL.Image as Image
import random
import math
from Elements import UAV, CAR, Person
from gym import Env, spaces
import time
import gym
import itertools

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

class Simple_MA_Env_Centralized_UAV_UGV(Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, size=100):
        super(Simple_MA_Env_Centralized_UAV_UGV, self).__init__()

        self.size = size

        self.step_count = 0

        # self.reward_dimish_rate = 0.01
        self.reward_dimish_rate = 0.1
        
        ## Observation space is a RGB image of size 640x640
        self.observation_shape = (size, size, 3)
        self.observation_space = spaces.Box(low = 0, 
                                            high = 255,
                                            shape = self.observation_shape,
                                            dtype = np.uint8)
        
        self.action_space_uav = gym.spaces.Discrete(8,)
        self.action_space_ugv = gym.spaces.Discrete(4,)

        # Canvas - used to render the environment
        self.canvas = np.zeros(self.observation_shape) * 1
        
        # A list of elements to be included in the observation space
        self.elements = []

        # Permissible area 
        self.y_min = 0
        self.x_min = 0
        self.y_max = self.observation_shape[1]
        self.x_max = self.observation_shape[0]

    def reset(self):
        ## Reset the reward
        self.ep_return  = 0
        self.people_count = 0
        self.total_people = 1
        self.people = []
        self.prev_near_target_uav = 99999
        self.prev_near_target_ugv = 99999
        self.lap_region_uav = np.zeros((100,100))
        self.lap_region_ugv = np.zeros((100,100))

        self.agents = {"uav": UAV("uav", self.x_max, self.x_min, self.y_max, self.y_min), 
                       "ugv": CAR("ugv", self.x_max, self.x_min, self.y_max, self.y_min)}

        ## Generate random coordinates to place the agent
        for key in self.agents.keys():
            x = random.randrange(int(self.observation_shape[0] * 0.2), int(self.observation_shape[0] * 0.8))
            y = random.randrange(int(self.observation_shape[1] * 0.2), int(self.observation_shape[1] * 0.8))
            self.agents[key].set_position(x,y)
            if key=="uav":
                for i in range(x-1,x+3):
                    for j in range(y-1,y+3): pass
                        #self.lap_region_uav[i][j] = 1
            if key=="ugv":
                for i in range(x-1,x+3):
                    for j in range(y-1,y+3): pass
                        #self.lap_region_ugv[i][j] = 1

            # Reset canvas 
            self.canvas = np.zeros(self.observation_shape) * 1
        

        # Initialize people
        for i in range(self.total_people):
            a = random.randrange(int(self.observation_shape[0] * 0.2), int(self.observation_shape[0] * 0.8))
            b = random.randrange(int(self.observation_shape[1] * 0.2), int(self.observation_shape[1] * 0.8))
            person = Person(f"person_{i}", self.x_max, self.x_min, self.y_max, self.y_min)
            person.set_position(a,b)
            self.people.append(person)
        
        self.elements = [self.agents["uav"], self.agents["ugv"]]

        near_target_uav = 99999
        for i in self.people:
            if math.dist((i.x, i.y),self.agents["uav"].get_position()) < near_target_uav:
                near_target_uav = math.dist((i.x, i.y),self.agents["uav"].get_position())

        near_target_ugv = 99999
        for i in self.people:
            if math.dist((i.x, i.y),self.agents["ugv"].get_position()) < near_target_ugv:
                near_target_ugv = math.dist((i.x, i.y),self.agents["ugv"].get_position())
        
        self.prev_near_target_uav = near_target_uav
        self.prev_near_target_ugv = near_target_ugv

        self.canvas = np.zeros(self.observation_shape) * 1
        self.draw_elements_on_canvas()
        return self.canvas
            
    def step(self, uav_action, ugv_action):
        done = False
        self.step_count += 1
        self.reward_uav = -1
        self.reward_ugv = -1

        ## Apply the action on the uav
        if uav_action == 0:
            ## If action leads to collision with wall, don't move
            self.agents["uav"].move(0, (self.size/20))
        elif uav_action == 1:
            ## If action leads to collision with wall, don't move
            self.agents["uav"].move(0,-(self.size/20))
        elif uav_action == 2:
            ## If action leads to collision with wall, don't move
            self.agents["uav"].move((self.size/20),0)
        elif uav_action == 3:
            ## If action leads to collision with wall, don't move
            self.agents["uav"].move(-(self.size/20),0)
        elif uav_action == 4:
            ## If action leads to collision with wall, don't move
            self.agents["uav"].move((self.size/20),(self.size/20))
        elif uav_action == 5:
            ## If action leads to collision with wall, don't move
            self.agents["uav"].move((self.size/20),-(self.size/20))
        elif uav_action == 6:
            ## If action leads to collision with wall, don't move
            self.agents["uav"].move(-(self.size/20),(self.size/20))
        elif uav_action == 7:
            ## If action leads to collision with wall, don't move
            self.agents["uav"].move(-(self.size/20),-(self.size/20))

        ## Apply the action on the ugv
        if ugv_action == 0:
            ## If action leads to collision with wall, don't move
            self.agents["ugv"].move(0, (self.size/50))
        elif ugv_action == 1:
            ## If action leads to collision with wall, don't move
            self.agents["ugv"].move(0,-(self.size/50))
        elif ugv_action == 2:
            ## If action leads to collision with wall, don't move
            self.agents["ugv"].move((self.size/50),0)
        elif ugv_action == 3:
            ## If action leads to collision with wall, don't move
            self.agents["ugv"].move(-(self.size/50),0)

        # if self.step_count % 10 == 0:
        #     for people in self.people:
        #         people.move(random.uniform(-1, 1), random.uniform(-1, 1))

        uav_x, uav_y = self.agents["uav"].get_position()
        uav_x, uav_y = int(uav_x), int(uav_y)

        ugv_x, ugv_y = self.agents["ugv"].get_position()
        ugv_x, ugv_y = int(ugv_x), int(ugv_y)

        for i in range(uav_x-1,uav_x+3):
            for j in range(uav_y-1,uav_y+3):
                self.lap_region_uav[i][j] += 1
        for i in range(ugv_x-1,ugv_x+3):
            for j in range(ugv_y-1,ugv_y+3):
                self.lap_region_ugv[i][j] += 1

        temp = self.people.copy()
        for i in range(len(self.people)):
            if math.dist((self.people[i].x, self.people[i].y),(uav_x,uav_y)) < 5:
                self.reward_uav = 3
                self.people_count += 1
                temp.remove(self.people[i])
        
        self.people = temp

        temp = self.people.copy()
        for i in range(len(self.people)):
            if math.dist((self.people[i].x, self.people[i].y),(ugv_x,ugv_y)) < 10:
                self.reward_ugv = 3
                self.people_count += 1
                temp.remove(self.people[i])
        
        self.people = temp

        near_target_uav = 99999
        for i in self.people:
            if math.dist((i.x, i.y),self.agents["uav"].get_position()) < near_target_uav:
                near_target_uav = math.dist((i.x, i.y),self.agents["uav"].get_position())
        
        if near_target_uav < self.prev_near_target_uav:
            self.reward_uav = 0.1
        self.prev_near_target_uav = near_target_uav

        near_target_ugv = 99999
        for i in self.people:
            if math.dist((i.x, i.y),self.agents["ugv"].get_position()) < near_target_ugv:
                near_target_ugv = math.dist((i.x, i.y),self.agents["ugv"].get_position())
        
        if near_target_ugv < self.prev_near_target_ugv:
            self.reward_ugv = 0.1
        self.prev_near_target_ugv = near_target_ugv

        self.reward_uav
        self.ep_return += self.reward_uav + self.reward_ugv

        if self.people_count == self.total_people:
            done = True
            # print("All people found in",self.step_count,"steps")
        self.canvas = np.zeros(self.observation_shape) * 1
        self.draw_elements_on_canvas()
        if self.step_count % 20 == 0:
            pass
            # print("People Found:",self.people_count,"in",self.step_count,"steps")
        return self.canvas, self.reward_uav, self.reward_ugv, done, self.people_count, self.lap_region_uav, self.lap_region_ugv, self.step_count
    
   
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    
    def draw_elements_on_canvas(self):
        '''
        This function projects all the elements on canvas
        '''

        ## Place each element at their respective postions
        for elem in self.elements:
            elem_shape = elem.icon.shape
            x,y = int(elem.x), int(elem.y)
            self.canvas[x: x + elem_shape[0], y : y + elem_shape[1]] = elem.icon
        
        for p in self.people:
            p_shape = p.icon.shape
            x,y = int(p.x), int(p.y)
            self.canvas[x: x + p_shape[0], y : y + p_shape[1]] = p.icon

    def render(self, mode = "human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            # cv2.imshow("Game", self.canvas)
            # Naming a window
            cv2.namedWindow("Simple MA Env", cv2.WINDOW_NORMAL)
            # Using resizeWindow()
            cv2.resizeWindow("Simple MA Env", 700, 700)
            # Displaying the image
            cv2.imshow("Simple MA Env", self.canvas)
            cv2.waitKey(10)

        elif mode == "rgb_array":
            return self.canvas

    def close(self):
        cv2.destroyAllWindows()

    def get_return(self):
        return self.ep_return