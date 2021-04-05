# Author: Mikita Sazanovich

import glob
import os
import fnmatch

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def get_ckpt_name(checkpoints_dir):
  pattern = "ckpt-*"
  ckpt_files = glob.glob(os.path.join(checkpoints_dir, pattern))
  if not ckpt_files:
    raise ValueError(f"Could not find any {pattern} files in {checkpoints_dir}.")
  ckpt_file = os.path.basename(ckpt_files[0])
  ckpt_name = ckpt_file.split('.')[0]
  return ckpt_name


def get_metric_from_experiment_logs(experiment_path, metric_tag):
  paths_to_event_files = glob.glob(os.path.join(experiment_path, 'summaries', 'events*'))
  assert len(paths_to_event_files) == 1
  path_to_event_files = paths_to_event_files[0]

  metric_values = []
  for event in tf.compat.v1.train.summary_iterator(path_to_event_files):
    for v in event.summary.value:
      if v.tag == metric_tag:
        metric_values.append(tf.make_ndarray(v.tensor))
  return np.array(metric_values)


def build_plot_for_metric(metric_tag, exp_patterns, exp_names):
  exps_dir = os.path.join('output', 'unit')
  pattern_to_exp_metric_measurements = []
  for exp_pattern in exp_patterns:
    matched_exp_names = fnmatch.filter(exp_names, exp_pattern)
    exp_paths = [os.path.join(exps_dir, matched_exp_name) for matched_exp_name in matched_exp_names]
    exp_metric_measurements = []
    for exp_path in exp_paths:
      metric_values = get_metric_from_experiment_logs(exp_path, metric_tag=metric_tag)
      exp_metric_measurements.append(metric_values)
    pattern_to_exp_metric_measurements.append(exp_metric_measurements)

  exp_to_mean_per_measurement = []
  exp_to_std_per_measurement = []
  for exp_metric_measurements in pattern_to_exp_metric_measurements:
    exp_metric_measurements = np.array(exp_metric_measurements)
    mean_per_measurement = np.mean(exp_metric_measurements, axis=0, keepdims=False)
    std_per_measurement = np.std(exp_metric_measurements, axis=0, keepdims=False)
    exp_to_mean_per_measurement.append(mean_per_measurement)
    exp_to_std_per_measurement.append(std_per_measurement)

  n = len(exp_patterns)
  exp_ids = list(range(n))
  exp_ids.sort(key=lambda exp_id: exp_to_mean_per_measurement[exp_id][-1], reverse=True)
  for exp_id in exp_ids:
    print(f'{exp_patterns[exp_id]} with {len(pattern_to_exp_metric_measurements[exp_id])} matches')
    print(exp_to_mean_per_measurement[exp_id][-1], '+-', exp_to_std_per_measurement[exp_id][-1])

  colors = plt.cm.get_cmap('viridis', n)
  plt.figure(figsize=(10, 8))
  for exp_id in exp_ids:
    mean_per_measurement = exp_to_mean_per_measurement[exp_id]
    std_per_measurement = exp_to_std_per_measurement[exp_id]
    steps = list(range(1, len(mean_per_measurement) + 1))
    color = colors(exp_id)
    plt.plot(steps, mean_per_measurement, c=color)
    plt.fill_between(
        steps,
        mean_per_measurement - std_per_measurement,
        mean_per_measurement + std_per_measurement,
        color=color,
        alpha=.2)
  plt.title(metric_tag)
  plt.legend([exp_patterns[exp_id] for exp_id in exp_ids])
  plt.show()


def build_plots_for_metrics(metric_tags, exp_pattern, exp_names):
  for metric_tag in metric_tags:
    build_plot_for_metric(metric_tag, exp_pattern, exp_names)
