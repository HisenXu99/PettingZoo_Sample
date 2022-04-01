# from importlib_metadata import re
import tensorflow.compat.v1 as tf
import numpy as np
import random

from mpe import simple_v2
import A2C as A2C



LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning r'ate for critic
MAX_EPISODE = 80
DISPLAY_REWARD_THRESHOLD = 0.1  # renders environment if total episode reward is greater then this threshold
RENDER = False


env = simple_v2.env(max_cycles=120, continuous_actions=False)
# parallel_env = parallel_wrapper_fn(env)
sess = tf.Session()
actor = A2C.Actor(sess, n_features=4, n_actions=5, lr=LR_A)
critic = A2C.Critic(sess, n_features=4, lr=LR_C)
saver = tf.train.Saver()  #声明ta.train.Saver()类用于保存
sess.run(tf.global_variables_initializer())




for i_episode in range(MAX_EPISODE):
    env.reset()
    observation = np.ones(4)
    observation=observation.reshape(-1)
    t = 0
    track_r = []
    ep_rs_sum=0
    reward_real=0
    reward_last=-2
    for agent in env.agent_iter(120):
        if RENDER: env.render()
        action = actor.choose_action(observation)
        reward_dict=env.step(action)
        # reward=reward_dict["agent_0"]
        observation_, reward, done, info = env.last()


        reward_real=(reward-reward_last)
        if reward_real<0:
            reward_real=0
        if reward>-0.3:
            reward_real=reward_real+10
            # done=1


        reward_last=reward
        observation_=observation_/10
        track_r.append(reward_real)
        td_error = critic.learn(observation, reward_real, observation_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(observation, action, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        observation=observation_
        
        t=t+1



        if done:
            ep_rs_sum=sum(track_r)
            print("episode:", i_episode, "  reward:", ep_rs_sum)
            if i_episode > 0: RENDER = True  # rendering
            break

print("123")
save_path = saver.save(sess,'save/filename.ckpt')#保存路径为相对路径的save文件夹,保存名为filename.ckpt
print ("[+] Model saved in file: %s" % save_path)