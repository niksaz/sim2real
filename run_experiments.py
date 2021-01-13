# Author: Mikita Sazanovich

import itertools
import os
import configuration


def run_experiment(config, update_params, tag):
  generated_config = config.copy()
  for field_path, field_value in update_params.items():
    field_parts = field_path.split('/')
    config_scope = generated_config
    for index, field in enumerate(field_parts):
      assert field in config_scope, f'Updating {field_path} which is unspecified in the initial config.'
      if index + 1 == len(field_parts):
        config_scope[field] = field_value
      else:
        config_scope = config_scope[field]

  generated_config_path = 'configs/unit/generated_duckietown_unit.yaml'
  configuration.dump_config(generated_config, generated_config_path)
  os.system(f'python train_unit.py {tag} --config_path {generated_config_path}')


def iterations_to_desc(iterations: int) -> str:
  return f'{iterations // 1000}K'


def main():
  original_config_path = 'configs/unit/duckietown_unit.yaml'
  config = configuration.load_config(original_config_path)
  iterations = 25000
  tag_to_update_params = {
      f'SIM2REAL-SHUFFLE-{iterations_to_desc(iterations)}': {
          'hyperparameters/shuffle_episode_indexes': True,
          'hyperparameters/loss/triplet_w': 0.0,
          'hyperparameters/loss/triplet_margin': 0.0,
          'hyperparameters/loss/tcc_w': 0.0,
          'hyperparameters/iterations': iterations,
      },
      f'SIM2REAL-TRIPLET-0.1-MAR-1.0-SHUFFLE-{iterations_to_desc(iterations)}': {
          'hyperparameters/shuffle_episode_indexes': True,
          'hyperparameters/loss/triplet_w': 0.1,
          'hyperparameters/loss/triplet_margin': 1.0,
          'hyperparameters/loss/tcc_w': 0.0,
          'hyperparameters/iterations': iterations,
      },
      f'SIM2REAL-TCC-0.1-SHUFFLE-{iterations_to_desc(iterations)}': {
          'hyperparameters/shuffle_episode_indexes': True,
          'hyperparameters/loss/triplet_w': 0.0,
          'hyperparameters/loss/triplet_margin': 0.0,
          'hyperparameters/loss/tcc_w': 0.1,
          'hyperparameters/iterations': iterations,
      },
      f'SIM2REAL-TCC-0.1-TRIPLET-0.1-MAR-1.0-SHUFFLE-{iterations_to_desc(iterations)}': {
          'hyperparameters/shuffle_episode_indexes': True,
          'hyperparameters/loss/triplet_w': 0.1,
          'hyperparameters/loss/triplet_margin': 1.0,
          'hyperparameters/loss/tcc_w': 0.1,
          'hyperparameters/iterations': iterations,
      },
  }
  for _ in itertools.count():
    for tag, update_params in tag_to_update_params.items():
      run_experiment(config, update_params, tag)


if __name__ == '__main__':
  main()
