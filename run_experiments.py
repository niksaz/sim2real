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
  os.system(f'python train.py {tag} --config_path {generated_config_path} --summarize')


def iterations_to_desc(iterations: int) -> str:
  return f'{iterations // 1000}K'


def main():
  original_config_path = 'configs/unit/duckietown_unit.yaml'
  config = configuration.load_config(original_config_path)
  iterations = 100000
  channels = 16
  tag_to_update_params_items = []
  for map_label, map_path in [
      ('DALP', 'home/zerogerc/msazanovich/aido3/data/daffy_loop_empty'),
      ('DAIS', 'home/zerogerc/msazanovich/aido3/data/daffy_udem1')]:
    for use_tcc_loss in [True, False]:
      for use_triplet_loss in [True, False]:
        if use_tcc_loss:
          tcc_w = 0.1
        else:
          tcc_w = 0.0
        if use_triplet_loss:
          triplet_w = 0.1
          triplet_margin = 1.0
        else:
          triplet_w = 0.0
          triplet_margin = 0.0
        tag = (f'T46-{map_label}2DUCK-{channels}'
               f'-TCC-{tcc_w}-TRIPLET-{triplet_w}-MAR-{triplet_margin}-{iterations_to_desc(iterations)}')
        update_params = {
          'hyperparameters/loss/tcc_w': tcc_w,
          'hyperparameters/loss/triplet_w': triplet_w,
          'hyperparameters/loss/triplet_margin': triplet_margin,
          'hyperparameters/iterations': iterations,
          'hyperparameters/gen/ch': channels,
          'hyperparameters/dis/ch': channels,
          'datasets/general/datasets_dir': '/',
          'datasets/domain_a/dataset_path': map_path,
          'datasets/domain_b/dataset_path': 'home/zerogerc/msazanovich/duckietown-data/aido3/duckietown',
        }
        tag_to_update_params_items.append((tag, update_params))
  for index in itertools.count():
    item_index = index % len(tag_to_update_params_items)
    tag, update_params = tag_to_update_params_items[item_index]
    run_experiment(config, update_params, tag)


if __name__ == '__main__':
  main()
