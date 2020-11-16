# Author: Mikita Sazanovich

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


def main():
  original_config_path = 'configs/unit/duckietown_unit.yaml'
  config = configuration.load_config(original_config_path)
  iterations = 10000
  for repeat in range(3):
    for tcc_w in [0.0, 0.1, 1.0, 10.0, 100.0]:
      tag = 'TCC'
      tag += f'-W-{tcc_w}'
      run_experiment(
          config, {
              'hyperparameters/iterations': iterations,
              'hyperparameters/loss/tcc_w': tcc_w,
          },
          tag)


if __name__ == '__main__':
  main()
