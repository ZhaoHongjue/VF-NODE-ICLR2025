# VF-NODE-ICLR2025
Code for VF-NODE: "[Accelerating Neural ODEs: A Variational Formulation-based Approach](https://openreview.net/forum?id=trV41CpAK4&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FConference%2FAuthors%23your-submissions))"

## Experiments

```bash
python scripts/main.py project=base ode=Toggle model=NeuralODE loss=spline_vf_loss use_wandb=True \
scheduler=cosine_onecycle scheduler.kwargs.transition_steps=5000 \
data.point_num=100 data.noise_level=0.01 data.ratio=0.8 
```
- Please refer to `./confs/ode/` to see the available options for `ode`.
- Please refer to `./confs/model/` to see the available options for `model`.
- Please refer to `./confs/loss/` to see the available options for `loss`.

Other hyperparameters are set in `./confs/default.yaml`. Please refer to the paper for more details about the hyperparameters for each experiment.

## Citation
If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{zhao2025accelerating,
    title={Accelerating Neural {ODE}s: A Variational Formulation-based Approach},
    author={Hongjue Zhao and Yuchen Wang and Hairong Qi and Zijie Huang and Han Zhao and Lui Sha and Huajie Shao},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=trV41CpAK4}
}
```

