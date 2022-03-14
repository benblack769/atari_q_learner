import torch
import numpy as np
from torch import nn


def get_avg_discounted_value(next_dones, next_mask, next_rewards, next_value_preds, gamma):
    minibatch_size, td_lambda = next_dones.shape

    lambda_gamma = torch.cumprod(torch.ones(td_lambda)*gamma,dim=0) * (1/gamma)

    # all dones also must be masks, by definition
    next_value_mask = next_mask * (1 - next_dones)
    # mask to apply to values
    lambda_mask = torch.cumprod(next_value_mask, dim=1)
    # still reward agents if they are done that step
    lambda_rew_mask = lambda_mask + next_dones
    lambda_rews = lambda_rew_mask*torch.cumsum(next_rewards,dim=1)
    lambda_values = lambda_mask * next_value_preds

    num_rews = torch.sum(lambda_rew_mask, dim=1)

    discounted_values = lambda_gamma * (lambda_rews + gamma * lambda_values)

    avg_discounted_value = torch.sum(discounted_values, dim=1) / (num_rews + 1e-10)
    return avg_discounted_value



class LinearTDsLearner:
    def __init__(self, feature_size, device, feature_preproc, learn_rate=0.001):
        self.value_function = nn.Linear(feature_size, 1, bias=True).to(device)
        self.optim = torch.optim.Adam(self.value_function.parameters(), lr=learn_rate)
        self.feature_size = feature_size
        self.device = device
        self.feature_preproc = feature_preproc

    def values(self, features):
        values = self.value_function.forward(features)
        return values

    def update_epoc(self, feature_sequence, dones, mask, rewards, minibatch_size, td_lambda, gamma):
        num_samples = len(feature_sequence)-td_lambda-1
        tot_td_err = 0
        num_steps = 0
        order = torch.randperm(num_samples, device=self.device)
        for i in range(0, num_samples-minibatch_size+1, minibatch_size):
            with torch.no_grad():
                idxs = order[i:i+minibatch_size]
                all_idx_block = idxs.view(-1,1) + torch.arange(0, td_lambda+1, device=self.device).view(1,-1)
                all_idxs = all_idx_block.flatten()
                next_idxs = all_idx_block[:,1:].flatten()
                all_features = feature_sequence[all_idxs]
                processed_features = self.feature_preproc(all_features)

            value_preds = self.value_function.forward(processed_features).view(minibatch_size, -1)

            with torch.no_grad():
                next_value_preds = value_preds[:,1:]
                next_dones = dones[next_idxs].view(minibatch_size,-1).float()
                next_mask = mask[next_idxs].view(minibatch_size,-1).float()
                next_rewards =  rewards[next_idxs].view(minibatch_size,-1)
                avg_discounted_value = get_avg_discounted_value(next_dones, next_mask, next_rewards, next_value_preds, gamma)

            cur_values = value_preds[:,0]
            self.optim.zero_grad()
            td_err = torch.mean((avg_discounted_value.detach() - cur_values)**2)
            td_err.backward()
            self.optim.step()
            tot_td_err += float(td_err.detach().numpy())
            num_steps += 1
        return tot_td_err / (num_steps*minibatch_size)

    @torch.no_grad()
    def _predict_values(self, feature_sequence, minibatch_size):
        num_samples = len(feature_sequence)
        value_batch_outs = []
        for i in range(0, num_samples, minibatch_size):
            maxi = min(num_samples, i+minibatch_size)
            feature_batch = feature_sequence[i: maxi]
            proced_features = self.feature_preproc(feature_batch)
            pred_vals = self.value_function.forward(proced_features).flatten()
            value_batch_outs.append(pred_vals)
        predicted_values = torch.cat(value_batch_outs, axis=0)
        print(predicted_values.shape)
        return predicted_values

    def _get_actual_values(self, dones, masks, rewards, gamma):
        value = 0
        actual_values = []
        rewards = rewards.cpu().detach().numpy().tolist()
        dones = dones.cpu().detach().numpy().tolist()
        masks = masks.cpu().detach().numpy().tolist()
        for rew, done, mask in zip(rewards, dones, masks):
            # TODO: This does not implement mask logic!!!
            if done:
                value = 0
            value += rew
            actual_values.append(value)
            value *= gamma
        actual_values.reverse()
        return torch.tensor(actual_values)

    def evaluate(self, feature_sequence, minibatch_size, dones, masks, rewards, gamma):
        actual_values = self._get_actual_values(dones, masks, rewards, gamma)
        pred_values = self._predict_values(feature_sequence, minibatch_size)

        return torch.mean(torch.abs(actual_values - pred_values))











#
