import torch
import numpy as np
from active_feature_extractor.experiments.linear_q_learner import get_avg_discounted_value,LinearTDLearner
import gym

def test_get_avg_next_value_simple_rewards():
    minibatch_size = 2
    gamma = 1.
    td_lambda = 5
    dones = torch.zeros((minibatch_size,td_lambda))
    mask = torch.ones((minibatch_size,td_lambda))
    rewards = 3 * torch.ones((minibatch_size,td_lambda))
    values = torch.zeros((minibatch_size,td_lambda))
    actual_avg_value = get_avg_discounted_value(dones, mask, rewards, values, gamma)
    expected_avg_value = 3*torch.sum(torch.arange(1,td_lambda+1).float())/5*torch.ones(minibatch_size)
    np.testing.assert_almost_equal(actual_avg_value.detach().numpy(), expected_avg_value.detach().numpy())


def test_get_avg_next_value_simple_rewards_dones():
    minibatch_size = 2
    gamma = 1.
    td_lambda = 5
    dones = torch.zeros((minibatch_size,td_lambda))
    mask = torch.ones((minibatch_size,td_lambda))
    dones[:,3] = True
    rewards = 3 * torch.ones((minibatch_size,td_lambda))
    values = torch.zeros((minibatch_size,td_lambda))
    actual_avg_value = get_avg_discounted_value(dones, mask, rewards, values, gamma)
    expected_avg_value = 3*torch.sum(torch.arange(1,4+1).float())/4*torch.ones(minibatch_size)
    np.testing.assert_almost_equal(actual_avg_value.detach().numpy(), expected_avg_value.detach().numpy())


def test_get_avg_next_value_simple_rewards_mask():
    minibatch_size = 2
    gamma = 1.
    td_lambda = 5
    dones = torch.zeros((minibatch_size,td_lambda))
    mask = torch.ones((minibatch_size,td_lambda))
    mask[:,3] = False
    rewards = 3 * torch.ones((minibatch_size,td_lambda))
    values = torch.zeros((minibatch_size,td_lambda))
    actual_avg_value = get_avg_discounted_value(dones, mask, rewards, values, gamma)
    expected_avg_value = 3*torch.sum(torch.arange(1,3+1).float())/3*torch.ones(minibatch_size)
    np.testing.assert_almost_equal(actual_avg_value.detach().numpy(), expected_avg_value.detach().numpy())


def test_get_avg_next_value_values_done():
    minibatch_size = 2
    gamma = 1.
    td_lambda = 5
    dones = torch.zeros((minibatch_size,td_lambda))
    mask = torch.ones((minibatch_size,td_lambda))
    dones[:,3] = True
    rewards = torch.zeros((minibatch_size,td_lambda))
    values = 3*4/3*torch.ones((minibatch_size,td_lambda))
    actual_avg_value = get_avg_discounted_value(dones, mask, rewards, values, gamma)
    expected_avg_value = 3*torch.ones(minibatch_size)
    np.testing.assert_almost_equal(actual_avg_value.detach().numpy(), expected_avg_value.detach().numpy())


def test_get_avg_next_value_values_mask():
    minibatch_size = 2
    gamma = 1.
    td_lambda = 5
    dones = torch.zeros((minibatch_size,td_lambda))
    mask = torch.ones((minibatch_size,td_lambda))
    mask[:,3] = True
    rewards = torch.zeros((minibatch_size,td_lambda))
    values = 3*torch.ones((minibatch_size,td_lambda))
    actual_avg_value = get_avg_discounted_value(dones, mask, rewards, values, gamma)
    expected_avg_value = 3*torch.ones(minibatch_size)
    np.testing.assert_almost_equal(actual_avg_value.detach().numpy(), expected_avg_value.detach().numpy())


class LinearlySolvableEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(shape=(3,),high=10,low=0)

    def _obs(self):
        return np.array([0.2*(self.counter//5),2*(self.counter%5),1],dtype="float32")

    def reset(self):
        self.counter = 50
        return self._obs()

    def step(self, action):
        self.counter -= 1
        done = self.counter == 0
        rew = 1 #+ random.normal()*0.3
        return self._obs(), rew, done, {}


def test_linear_q_learner_solvable():
    env = LinearlySolvableEnv()
    observations = []
    rewards = []
    dones = []
    num_rollout_steps = 40000
    train_set_size = 20000
    n_epocs = 1000
    minibatch_size = 128
    td_lambda = 1
    gamma = 0.99
    test_set_size = num_rollout_steps - train_set_size
    env.reset()
    for i in range(num_rollout_steps):
        obs, rew, done, info = env.step(env.action_space.sample())
        observations.append(obs)
        rewards.append(rew)
        dones.append(done)
        if done:
            env.reset()

    observations = torch.tensor(np.stack(observations))
    rewards = torch.tensor(rewards)
    dones = torch.tensor(dones)
    mask = torch.ones(num_rollout_steps)

    train_obs = observations[:train_set_size]
    train_rews = rewards[:train_set_size]
    train_dones = dones[:train_set_size]
    train_mask = mask[:train_set_size]

    test_obs = observations[train_set_size:]
    test_rews = rewards[test_set_size:]
    test_dones = dones[test_set_size:]
    test_mask = mask[test_set_size:]

    def transform_obs(obs_batch):
        return obs_batch

    transformed_obs_size = transform_obs(observations[:1]).shape[1]
    learner = LinearTDLearner(transformed_obs_size, 'cpu', transform_obs)
    for i in range(n_epocs):
        train_td_er = learner.update_epoc(train_obs, train_dones, train_mask, train_rews, minibatch_size, td_lambda, gamma)
        true_value_err = learner.evaluate(test_obs, minibatch_size, test_dones, test_mask, test_rews, gamma)
        print(true_value_err, "\t", train_td_er)


def test_linear_q_learner_cartpole():
    env = gym.make("CartPole-v0")
    observations = []
    rewards = []
    dones = []
    num_rollout_steps = 40000
    train_set_size = 20000
    n_epocs = 100
    minibatch_size = 128
    td_lambda = 3
    gamma = 0.99
    test_set_size = num_rollout_steps - train_set_size
    env.reset()
    for i in range(num_rollout_steps):
        obs, rew, done, info = env.step(env.action_space.sample())
        observations.append(obs)
        rewards.append(rew)
        dones.append(done)
        if done:
            env.reset()

    observations = torch.tensor(np.stack(observations))
    rewards = torch.tensor(rewards)
    dones = torch.tensor(dones)
    mask = torch.ones(num_rollout_steps)

    train_obs = observations[:train_set_size]
    train_rews = rewards[:train_set_size]
    train_dones = dones[:train_set_size]
    train_mask = mask[:train_set_size]

    test_obs = observations[train_set_size:]
    test_rews = rewards[test_set_size:]
    test_dones = dones[test_set_size:]
    test_mask = mask[test_set_size:]

    obs_max = torch.max(observations, dim=0).values
    obs_min = torch.min(observations, dim=0).values

    def fourier_transform(obs_batch, num_transforms, step_scale_change):
        flat_data = obs_batch.view(-1)
        scales = torch.cumprod(torch.ones(num_transforms)*step_scale_change, 0)/step_scale_change
        scaled_data = flat_data.view(-1,1) + scales.view(1,-1)
        transformed = torch.cat([torch.sin(scaled_data),torch.cos(scaled_data)],axis=1)
        return transformed.view(obs_batch.shape[0],-1)


    def transform_obs(obs_batch):
        # return obs_batch
        normalized_obs = 4 * obs_batch / (obs_max - obs_min) + obs_min
        return fourier_transform(normalized_obs, 8, 0.6)

    transformed_obs_size = transform_obs(observations[:1]).shape[1]
    learner = MLPTDLearner(transformed_obs_size, 'cpu', transform_obs)
    for i in range(n_epocs):
        train_td_er = learner.update_epoc(train_obs, train_dones, train_mask, train_rews, minibatch_size, td_lambda, gamma)
        true_value_err = learner.evaluate(test_obs, minibatch_size, test_dones, test_mask, test_rews, gamma)
        print(true_value_err, "\t", train_td_er)


if __name__ == "__main__":
    test_linear_q_learner_solvable()












#
