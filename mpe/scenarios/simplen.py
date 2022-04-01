from tkinter.tix import DirTree
import numpy as np

from .._mpe_utils.core import Agent, Landmark, World
from .._mpe_utils.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self,n=2):
        world = World()
        num_landmarks = n
        # add agents
        world.agents = [Agent() for i in range(1)]
        #enumerate将一个可遍历的数据对象，组合为一个索引序列，同时列出数据和数据下标
        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.collide = False
            agent.silent = True
            
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size=0.2
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.75, 0.75])
        # world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
        # set goal landmark
        goal = world.landmarks[0]
        goal.color = np.array([0.25, 0.25, 0.25])
        for agent in world.agents:
            agent.goal_a = goal
        # set random initial states
        for agent in world.agents:
            #NOTE 每次初始化的位置，生成一个维度是world.dim_p，范围[-1,1]区间内均匀分布随机数
            agent.state.p_pos = np_random.uniform(-10, +10, world.dim_p)
            #初始速度为0
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-10, +10, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        #xhx修改
        #STAR 修改了reward，加入了根号
        # print(100*agent.state.p_pos , 100*world.landmarks[0].state.p_pos)
        distlist= []

        dist2 = np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        dist2 = np.sqrt(dist2)
        distlist.append(dist2)
        for i, landmark in enumerate(world.landmarks):
            if(landmark!=agent.goal_a):
                distlist.append(np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos))))
        return dist2

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)
