# Author: Mikita Sazanovich

import os
import configuration


def run_experiment(config, update_params, tag):
  generated_config = dict(config)
  for field_path, field_value in update_params.items():
    field_parts = field_path.split('/')
    config_scope = generated_config
    for field in field_parts[:-1]:
      config_scope = config_scope[field]
    config_scope[field_parts[-1]] = field_value

  generated_config_path = 'exps/unit/generated_duckietown_unit.yaml'
  configuration.dump_config(generated_config, generated_config_path)
  os.system(f'python train_unit.py {tag} --config_path {generated_config_path}')


def main():
  original_config_path = 'exps/unit/duckietown_unit.yaml'
  config = configuration.load_config(original_config_path)
  run_experiment(
      config, {
        'hyperparameters/iterations': 1000,
        'hyperparameters/control/loss': 'mae',
      },
      'ADAM_SGD_ABA_MAE_1k')
  run_experiment(
      config, {
        'hyperparameters/iterations': 1000,
        'hyperparameters/control/loss': 'mse',
      },
      'ADAM_SGD_ABA_MSE_1k')


if __name__ == '__main__':
  main()
