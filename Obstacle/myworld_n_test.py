# from importlib_metadata import re
import tensorflow.compat.v1 as tf
import numpy as np
import random
import os
import sys 
sys.path.append(os.path.dirname(sys.path[0]))
from mpe_obstacle import simple_n
import A2C_n as A2C
import datetime
import time


LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning r'ate for critic
MAX_EPISODE = 100
DISPLAY_REWARD_THRESHOLD = 0.1  # renders environment if total episode reward is greater then this threshold
RENDER = False
Num_obstacle=2
Num_state=(Num_obstacle+2)*2


model_path = 'save_model/2022-03-31-12_03_32/filename.ckpt'
# model_path = 'save/filename.ckpt'
env = simple_n.env(max_cycles=120, continuous_actions=False,n=Num_obstacle+1)
# parallel_env = parallel_wrapper_fn(env)
sess = tf.Session()
actor = A2C.Actor(sess, n_features=Num_state, n_actions=5, lr=LR_A)
critic = A2C.Critic(sess, n_features=Num_state, lr=LR_C)
saver = tf.train.Saver()  #声明ta.train.Saver()类用于保存
load_path = saver.restore(sess, model_path)

# reward_list = []



for i_episode in range(MAX_EPISODE):
    env.reset()
    observation = np.ones(Num_state)
    observation=observation.reshape(-1)
    t = 0
    track_r = []
    ep_rs_sum=0
    # reward_real=0
    o_last=np.zeros((Num_obstacle+2)*2)
    d_last=-5
    # obstacle_flag=0
    for agent in env.agent_iter(120):
        reward_real=0
        obstacle_flag=0
        if RENDER: env.render()
        action = actor.choose_action(observation)
        # print(action)
        d=env.step(action)
        # reward=reward_dict["agent_0"]
        observation_, d, done, info = env.last()

        ########################正奖励########################
        # if d>d_last:
        #     reward_real=reward_real+1

        # for i in range(Num_obstacle):
        #     # print(observation_[2*(i+1):2*(i+1)+2])
        #     delta=np.sum(np.square(observation_[2*(i+1):2*(i+1)+2]))-2
        #     if delta<0:
        #         obstacle_flag=1
        #         break
        # if obstacle_flag==0:
        #     reward_real=reward_real+2

        # if d>-1:
        #     reward_real=reward_real+100
        #     done=1

        ########################负奖励########################
        # if d>d_last:
        #     reward_real=reward_real+1

        # for i in range(Num_obstacle):
        #     # print(observation_[2*(i+1):2*(i+1)+2])
        #     delta=np.sum(np.square(observation_[2*(i+1):2*(i+1)+2]))-2
        #     if delta<0:
        #         obstacle_flag=1
        #         break
        # if obstacle_flag==1:
        #     reward_real=0

        # if d>-1:
        #     reward_real=reward_real+500
        #     done=1

        ########################距离负奖励########################
        # if d>d_last:
        #     reward_real=reward_real+1
        reward_real=(d-d_last)*20
        if reward_real<0:
            reward_real=0
        if d>-1:
            reward_real=reward_real+10
            done=1

        for i in range(Num_obstacle):
            # print(observation_[2*(i+1):2*(i+1)+2])
            delta=(np.sum(np.square(observation_[2*(i+2):2*(i+2)+2]))-8)/10
            if delta<0:
                # # print(delta)
                # obstacle_flag=1
                reward_real=reward_real+delta
                break
        # if obstacle_flag==1:
        #     reward_real=0



        d_last=d

        observation_=observation_/10     #NOTE:不除10就一直执行一个动作，除10之后效果很明显

        track_r.append(reward_real)
        # td_error = critic.learn(observation, reward_real, observation_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        # actor.learn(observation, action, td_error)     # true_gradient = grad[logPi(s,a) * td_error]


        observation=observation_
        
        t=t+1


        if done:
            ep_rs_sum=sum(track_r)
            print("episode:", i_episode, "  reward:", ep_rs_sum)
            # reward_list.append(ep_rs_sum)
            # reward_array=np.array(reward_list)  
            if i_episode > 0: RENDER = True  # rendering
            break

        time.sleep(0.005)
