#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 08:48:36 2023

@author: jaehooncha
"""
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.metrics import mutual_info_score

def _encodings(target):
  discretized = np.zeros_like(target)
  for i in range(target.shape[0]):
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(target[i, :])
    encoded = label_encoder.transform(target[i, :])
    discretized[i, :] = encoded
  return discretized

def _histogram_discretize(target, num_bins=20):
  """Discretization based on histograms."""
  discretized = np.zeros_like(target)
  for i in range(target.shape[0]):
    discretized[i, :] = np.digitize(target[i, :], np.histogram(
        target[i, :], num_bins)[1][:-1])
  return discretized



def discrete_mutual_info(mus, ys):
  """Compute discrete mutual information."""
  num_codes = mus.shape[0]
  num_factors = ys.shape[0]
  m = np.zeros([num_codes, num_factors])
  for i in range(num_codes):
    for j in range(num_factors):
      m[i, j] = mutual_info_score(ys[j, :], mus[i, :])
  return m

def discrete_entropy(ys):
  """Compute discrete mutual information."""
  num_factors = ys.shape[0]
  h = np.zeros(num_factors)
  for j in range(num_factors):
    h[j] = mutual_info_score(ys[j, :], ys[j, :])
  return h


def repres_from_obs(observations, transform, model, valid_idx):
  observations = transform(observations)
  if len(observations.shape) == 3:
    observations = np.expand_dims(observations, axis = 1)

  if observations.shape[1]>3:
    observations = observations.transpose((0, 3, 1, 2))

  device = next(iter(model.parameters()))[0].device
  representations = model.latent(torch.Tensor(observations).view(observations.shape[0], -1).to(device))[:, valid_idx]
  representations = representations.detach().cpu().numpy()
  return representations

def ground_truth_data_sample(ground_truth_factors, ground_truth_data, transform, model,
                              batch_size, valid_idx, random_state, repre = False):
  # Sample two mini batches of latent variables.
  rnd_idx = random_state.randint(0, len(ground_truth_factors), batch_size)
  factors = ground_truth_factors[rnd_idx]
  observations = ground_truth_data[rnd_idx]
  if repre:
      representations = repres_from_obs(observations, transform, model, valid_idx)
      return representations, factors, observations
  else:
      return factors, observations


def generate_batch_factor_code(ground_truth_factors, ground_truth_data, transform, model,
                               num_points, batch_size, valid_idx, random_state):
  """Sample a single training sample based on a mini-batch of ground-truth data.
  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    num_points: Number of points to sample.
    random_state: Numpy random state used for randomness.
    batch_size: Batchsize to sample points.
  Returns:
    representations: Codes (num_codes, num_points)-np array.
    factors: Factors generating the codes (num_factors, num_points)-np array.
  """
  representations = None
  factors = None
  i = 0
  for i in range(int(num_points/batch_size)):
    current_factors, current_observations = \
        ground_truth_data_sample(ground_truth_factors, ground_truth_data, transform, model,
                                      batch_size, valid_idx, random_state)
    if i == 0:
      factors = current_factors
      representations = repres_from_obs(current_observations, transform, model, valid_idx)
    else:
      factors = np.vstack((factors, current_factors))
      representations = np.vstack((representations,
                                   repres_from_obs(current_observations, transform, model, valid_idx)))
  return np.transpose(representations), np.transpose(factors)

def compute_jemmig(ground_truth_factors,
                ground_truth_data,
                transform,
                model,
                random_state,
                num_train,
                valid_idx,
                categorical_continuous_factors = False,
                continuous_factors = False,
                batch_size=16,
                num_bins = 20,
                artifact_dir=None):
  """Computes the mutual information gap.
  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    batch_size: Batch size for sampling.
  Returns:
    Dict with average mutual information gap.
  """
  del artifact_dir

  mus_train, ys_train = generate_batch_factor_code(ground_truth_factors, ground_truth_data, transform, model,
                                 num_train, batch_size, valid_idx, random_state)


  assert mus_train.shape[1] == num_train
  return _compute_jemmig(mus_train, ys_train, categorical_continuous_factors, continuous_factors, num_bins)


def _compute_jemmig(mus_train, ys_train, categorical_continuous_factors, continuous_factors, num_bins):
  """Computes score based on both training and testing codes and factors."""
  score_dict = {}

  if categorical_continuous_factors:
    discretized_ys = _encodings(ys_train)
  elif continuous_factors:
    discretized_ys = _histogram_discretize(ys_train, num_bins)
  else:
    discretized_ys = ys_train


  discretized_mus = _histogram_discretize(mus_train, num_bins)
  m = discrete_mutual_info(discretized_mus, discretized_ys)
  assert m.shape[0] == mus_train.shape[0]
  assert m.shape[1] == ys_train.shape[0]
  # m is [num_latents, num_factors]
  entropx = discrete_entropy(discretized_mus)
  entropy = discrete_entropy(discretized_ys)

  entropxx = np.repeat(entropx.reshape(-1, 1), len(entropy), axis = 1)
  entropyy = np.repeat(entropy.reshape(1, -1), len(entropx), axis = 0)

  sorted_m = np.sort(m, axis=0)[::-1]
  mi_idx = np.argsort(m, axis=0)[::-1]
  migap = sorted_m[0, :] - sorted_m[1, :]
  joint_entropy = entropxx + entropyy - m
  je_top = joint_entropy[mi_idx[0], np.arange(len(entropy))]

  score_dict["discrete_jemmig"] = 1 - np.mean(np.divide(np.mean(je_top - migap), entropy + np.log2(num_bins)))
  return score_dict