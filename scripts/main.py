import os, sys, hydra
from omegaconf import DictConfig, OmegaConf

os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import jax
jax.config.update('jax_enable_x64', True)
jax.default_backend()

from engine import Trainer

@hydra.main(
    config_path = '../confs', config_name = 'default', version_base = '1.3'
)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir 
    trainer = Trainer(cfg, output_dir)
    trainer.fit()
    trainer.evaluate(jax.random.PRNGKey(20010830))

if __name__ == '__main__':
    main()