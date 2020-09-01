# Author: Mikita Sazanovich

import collections
from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
  @abstractmethod
  def update_state(self, y_true, y_pred):
    pass

  @abstractmethod
  def result(self):
    pass


class MAEMetric(Metric):
  def __init__(self):
    super().__init__()
    self.errors_sum = 0.0
    self.errors_count = 0

  def update_state(self, y_true, y_pred):
    losses = np.mean(np.abs(y_true - y_pred), axis=1)
    self.errors_sum += np.sum(losses)
    self.errors_count += len(losses)

  def result(self):
    return self.errors_sum / self.errors_count


class MSEMetric(Metric):
  def __init__(self):
    super().__init__()
    self.errors_sum = 0.0
    self.errors_count = 0

  def update_state(self, y_true, y_pred):
    losses = np.mean(np.square(y_true - y_pred), axis=1)
    self.errors_sum += np.sum(losses)
    self.errors_count += len(losses)

  def result(self):
    return self.errors_sum / self.errors_count


class BalancedMetric(Metric):
  INF = 1000
  BINS = [(-INF, 0.5, -INF, 0.5), (-INF, 0.5, 0.5, INF), (0.5, INF, -INF, 0.5), (0.5, INF, 0.5, INF)]

  def __init__(self, metric_cls):
    super().__init__()
    self.bins = collections.defaultdict(metric_cls)

  def update_state(self, y_true, y_pred):
    for yt, yp in zip(y_true, y_pred):
      bin_id = BalancedMetric.get_bin_id_for(yt)
      self.bins[bin_id].update_state(np.array([yt]), np.array([yp]))

  def result(self):
    errors = []
    for _, bin_mae_metric in self.bins.items():
      errors.append(bin_mae_metric.result())
    return np.mean(errors)

  @staticmethod
  def get_bin_id_for(point):
    bin_id = None
    for i, (l0, u0, l1, u1) in enumerate(BalancedMetric.BINS):
      if l0 <= point[0] <= u0 and l1 <= point[1] <= u1:
        bin_id = i
        break
    if bin_id is None:
      raise ValueError(f'Bin has not been found for action {point}')
    else:
      return bin_id


class BMAEMetric(BalancedMetric):
  def __init__(self):
    super().__init__(MAEMetric)


class BMSEMetric(BalancedMetric):
  def __init__(self):
    super().__init__(MSEMetric)
