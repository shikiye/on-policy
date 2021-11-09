import gfootball.env as football_env
import gym
import numpy as np

class GFootballEnv(gym.Env):
  def __init__(self, args, is_render=False):
    self.num_agents = 4
    self.env = football_env.create_environment(
        env_name=args.scenario_name,#'5_vs_5',
        stacked=False,
        #logdir=os.path.join(tempfile.gettempdir(), 'rllib_test'),
        write_goal_dumps=False, write_full_episode_dumps=False, render=is_render,
        dump_frequency=0,
        number_of_left_players_agent_controls=self.num_agents,
        channel_dimensions=(42, 42)
    )
    self.num_actions = 19
    self.available_actions = []
    for i in range(self.num_agents):
        self.available_actions.append([1] * self.num_actions)
    self.action_space = [gym.spaces.Discrete(self.env.action_space.nvec[1])for _ in range(self.num_agents)]
    self.observation_space = [gym.spaces.Box(
        low=self.env.observation_space.low[0],
        high=self.env.observation_space.high[0],
        dtype=self.env.observation_space.dtype)for _ in range(self.num_agents)]
    self.share_observation_space = [gym.spaces.Box(
        low=self.env.observation_space.low[0],
        high=self.env.observation_space.high[0],
        dtype=self.env.observation_space.dtype) for _ in range(self.num_agents)]
    #print(self.env.observation_space.low[0].shape) #42 42 4, value=255
    #print(self.env.observation_space.high[0].shape)

  def seed(self, seed=None):
      if seed is None:
          np.random.seed(1)
      else:
          np.random.seed(seed)

  def reset(self):
    obs = self.env.reset()
    #obs = np.array([obs * self.num_agents])
    share_obs = obs
    return obs, share_obs, self.available_actions

  def step(self, action):
    obs, rew, done, info = self.env.step(action)
    rews = []
    for i in range(len(rew)):
        rews.append([rew[0]])
    rews = np.array(rews)
    #obs = np.array([obs * self.num_agents])
    share_obs = obs
    dones = np.array([[done] * self.num_agents])
    infos = np.array([[info] * self.num_agents])
    #return (self.observation(), np.array(reward, dtype=np.float32), done, info)
    return obs, share_obs, rews, dones, infos, self.available_actions
    #return self.env.step(action)

  def render(self, mode='human'):
    return self.env.render(mode)