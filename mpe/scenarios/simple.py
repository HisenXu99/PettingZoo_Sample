from re import S
from tkinter.tix import DirTree
import numpy as np

from .._mpe_utils.core import Agent, Landmark, World
from .._mpe_utils.scenario import BaseScenario


#加入的求夹角的函数
def clockwise_angle(v1, v2):
    x1,y1 = v1
    x2,y2 = v2
    dot = x1*x2+y1*y2
    det = x1*y2-y1*x2
    theta = np.arctan2(det, dot)
    # theta = theta if theta>0 else 2*np.pi+theta
    return theta

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # add agents
        world.agents = [Agent() for i in range(1)]
        #enumerate将一个可遍历的数据对象，组合为一个索引序列，同时列出数据和数据下标
        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.collide = False
            agent.silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(1)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.75, 0.75])
        world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
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

        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        dist2 = np.sqrt(dist2)
        return -dist2




    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # return np.concatenate([agent.state.p_vel] + entity_pos)

        state = []
        for entity in world.landmarks:
            # dist = np.sum(np.square(agent.state.p_pos - entity.state.p_pos))
            # dist = np.sqrt(dist)
            # theta=clockwise_angle(agent.state.p_pos,entity.state.p_pos)
            # # print(theta)
            # state.append([dist,theta])
            state.append(entity.state.p_pos - agent.state.p_pos)
        # if(state[0][1]<0):
        #     print(np.concatenate([agent.state.p_vel] + state))
        return np.concatenate([agent.state.p_vel] + state)
        




