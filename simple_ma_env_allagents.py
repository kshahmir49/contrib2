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

class Simple_MA_Env_Centralized_AllAgents(Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, size=100):
        super(Simple_MA_Env_Centralized_AllAgents, self).__init__()

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
        
        self.action_space_agent1 = gym.spaces.Discrete(4,)
        self.action_space_agent2 = gym.spaces.Discrete(4,)
        self.action_space_agent3 = gym.spaces.Discrete(12,)

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
        self.prev_near_target_agent1 = 99999
        self.prev_near_target_agent2 = 99999
        self.prev_near_target_agent3 = 99999
        self.lap_region_agent1 = np.zeros((100,100))
        self.lap_region_agent2 = np.zeros((100,100))
        self.lap_region_agent3 = np.zeros((100,100))

        self.agents = {"agent1": CAR("agent1", self.x_max, self.x_min, self.y_max, self.y_min), 
                       "agent2": CAR("agent2", self.x_max, self.x_min, self.y_max, self.y_min),
                       "agent3": CAR("agent3", self.x_max, self.x_min, self.y_max, self.y_min)}

        ## Generate random coordinates to place the agent
        x = random.randrange(int(self.observation_shape[0] * 0.2), int(self.observation_shape[0] * 0.8))
        y = random.randrange(int(self.observation_shape[1] * 0.2), int(self.observation_shape[1] * 0.8))
        for key in self.agents.keys():
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
        
        self.elements = [self.agents["agent1"], self.agents["agent2"], self.agents["agent3"]]

        near_target_agent1 = 99999
        for i in self.people:
            if math.dist((i.x, i.y),self.agents["agent1"].get_position()) < near_target_agent1:
                near_target_agent1 = math.dist((i.x, i.y),self.agents["agent1"].get_position())

        near_target_agent2 = 99999
        for i in self.people:
            if math.dist((i.x, i.y),self.agents["agent2"].get_position()) < near_target_agent2:
                near_target_agent2 = math.dist((i.x, i.y),self.agents["agent2"].get_position())
            
        near_target_agent3 = 99999
        for i in self.people:
            if math.dist((i.x, i.y),self.agents["agent3"].get_position()) < near_target_agent3:
                near_target_agent3 = math.dist((i.x, i.y),self.agents["agent3"].get_position())
        
        self.prev_near_target_agent1 = near_target_agent1
        self.prev_near_target_agent2 = near_target_agent2
        self.prev_near_target_agent3 = near_target_agent3

        self.canvas = np.zeros(self.observation_shape) * 1
        self.draw_elements_on_canvas()
        return self.canvas
            
    def step(self, agent1_action, agent2_action, agent3_action):
        done = False
        self.step_count += 1
        self.reward_agent1 = -1
        self.reward_agent2 = -1
        self.reward_agent3 = -1

        ## Apply the action on the uav
        if agent1_action == 0:
            ## If action leads to collision with wall, don't move
            self.agents["agent1"].move(0, 2)
        elif agent1_action == 1:
            ## If action leads to collision with wall, don't move
            self.agents["agent1"].move(0,-2)
        elif agent1_action == 2:
            ## If action leads to collision with wall, don't move
            self.agents["agent1"].move(2,0)
        elif agent1_action == 3:
            ## If action leads to collision with wall, don't move
            self.agents["agent1"].move(-2,0)

        if agent2_action == 0:
            ## If action leads to collision with wall, don't move
            self.agents["agent2"].move(2, 2)
        elif agent2_action == 1:
            ## If action leads to collision with wall, don't move
            self.agents["agent2"].move(2,-2)
        elif agent2_action == 2:
            ## If action leads to collision with wall, don't move
            self.agents["agent2"].move(-2,2)
        elif agent2_action == 3:
            ## If action leads to collision with wall, don't move
            self.agents["agent2"].move(-2,-2)

        if agent3_action == 0:
            self.agents["agent3"].move(2, 2)
        elif agent3_action == 1:
            self.agents["agent3"].move(2,-2)
        elif agent3_action == 2:
            self.agents["agent3"].move(-2,2)
        elif agent3_action == 3:
            self.agents["agent3"].move(-2,-2)
        elif agent3_action == 4:
            self.agents["agent3"].move(2,1)
        elif agent3_action == 5:
            self.agents["agent3"].move(1,2)
        elif agent3_action == 6:
            self.agents["agent3"].move(2,-1)
        elif agent3_action == 7:
            self.agents["agent3"].move(1,-2)
        elif agent3_action == 8:
            self.agents["agent3"].move(-1,-2)
        elif agent3_action == 9:
            self.agents["agent3"].move(-2,-1)
        elif agent3_action == 10:
            self.agents["agent3"].move(-1,2)
        elif agent3_action == 11:
            self.agents["agent3"].move(-2,1)

        # if self.step_count % 10 == 0:
        #     for people in self.people:
        #         people.move(random.uniform(-1, 1), random.uniform(-1, 1))

        agent1_x, agent1_y = self.agents["agent1"].get_position()
        agent1_x, agent1_y = int(agent1_x), int(agent1_y)

        agent2_x, agent2_y = self.agents["agent2"].get_position()
        agent2_x, agent2_y = int(agent2_x), int(agent2_y)

        agent3_x, agent3_y = self.agents["agent3"].get_position()
        agent3_x, agent3_y = int(agent3_x), int(agent3_y)

        for i in range(agent1_x-1,agent1_x+3):
            for j in range(agent1_y-1,agent1_y+3):
                self.lap_region_agent1[i][j] += 1
        for i in range(agent2_x-1,agent2_x+3):
            for j in range(agent2_y-1,agent2_y+3):
                self.lap_region_agent2[i][j] += 1
        for i in range(agent3_x-1,agent3_x+3):
            for j in range(agent3_y-1,agent3_y+3):
                self.lap_region_agent3[i][j] += 1

        temp = self.people.copy()
        for i in range(len(self.people)):
            if math.dist((self.people[i].x, self.people[i].y),(agent1_x,agent1_y)) < 15:
                self.reward_agent1 = 3
                self.people_count += 1
                temp.remove(self.people[i])
        
        self.people = temp

        temp = self.people.copy()
        for i in range(len(self.people)):
            if math.dist((self.people[i].x, self.people[i].y),(agent2_x,agent2_y)) < 15:
                self.reward_agent2 = 3
                self.people_count += 1
                temp.remove(self.people[i])
        
        self.people = temp

        temp = self.people.copy()
        for i in range(len(self.people)):
            if math.dist((self.people[i].x, self.people[i].y),(agent3_x,agent3_y)) < 15:
                self.reward_agent3 = 3
                self.people_count += 1
                temp.remove(self.people[i])
        
        self.people = temp

        near_target_agent1 = 99999
        for i in self.people:
            if math.dist((i.x, i.y),self.agents["agent1"].get_position()) < near_target_agent1:
                near_target_agent1 = math.dist((i.x, i.y),self.agents["agent1"].get_position())
        
        if near_target_agent1 < self.prev_near_target_agent1:
            self.reward_agent1 = 0.1
        self.prev_near_target_agent1 = near_target_agent1

        near_target_agent2 = 99999
        for i in self.people:
            if math.dist((i.x, i.y),self.agents["agent2"].get_position()) < near_target_agent2:
                near_target_agent2 = math.dist((i.x, i.y),self.agents["agent2"].get_position())
        
        if near_target_agent2 < self.prev_near_target_agent2:
            self.reward_agent2 = 0.1
        self.prev_near_target_agent2 = near_target_agent2

        near_target_agent3 = 99999
        for i in self.people:
            if math.dist((i.x, i.y),self.agents["agent3"].get_position()) < near_target_agent3:
                near_target_agent3 = math.dist((i.x, i.y),self.agents["agent3"].get_position())
        
        if near_target_agent3 < self.prev_near_target_agent3:
            self.reward_agent3 = 0.1
        self.prev_near_target_agent3 = near_target_agent3

        self.ep_return += self.reward_agent1 + self.reward_agent2 + self.reward_agent3

        if self.people_count == self.total_people:
            done = True
            # print("All people found in",self.step_count,"steps")
        self.canvas = np.zeros(self.observation_shape) * 1
        self.draw_elements_on_canvas()
        if self.step_count % 20 == 0:
            pass
            # print("People Found:",self.people_count,"in",self.step_count,"steps")
        return self.canvas, self.reward_agent1, self.reward_agent2, self.reward_agent3, done, self.people_count, self.lap_region_agent1, self.lap_region_agent2, self.lap_region_agent3, self.step_count
    
   
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