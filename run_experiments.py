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

  generated_config_path = 'configs/unit/generated_duckietown_unit.yaml'
  configuration.dump_config(generated_config, generated_config_path)
  os.system(f'python train_unit.py {tag} --config_path {generated_config_path}')


def main():
  original_config_path = 'configs/unit/duckietown_unit.yaml'
  config = configuration.load_config(original_config_path)
  iterations = 10000
  for z_recon_w in [0.0, 0.1]:
    for z_temporal_w in [0.0, 0.1, 1.0, 10.0]:
      run_experiment(
          config, {
              'hyperparameters/iterations': iterations,
              'hyperparameters/z_recon_w': z_recon_w,
              'hyperparameters/z_temporal_w': z_temporal_w,
          },
          f'SEP02_RECON_{z_recon_w}_TEMPO_{z_temporal_w}_RUN1_10k')


if __name__ == '__main__':
  main()
