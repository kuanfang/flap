
# Fine-Tuning with Lossy Affordance Planner (FLAP)

This repo contains the code for:

[Generalization with Lossy Affordances: Leveraging Broad Offline Data for Learning Visuomotor Tasks](https://arxiv.org/abs/2210.06601)

[Kuan Fang](https://kuanfang.github.io/), [Patrick Yin](https://patrickyin.me/), [Ashvin Nair](https://ashvin.me/), [Homer Walke](https://homerwalke.com/), [Gengchen Yan](https://www.linkedin.com/in/gengchen-matt-yan/), [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/). 

Conference on Robot Learning (CoRL), 2022. Oral Presentation

[[Project Page](https://sites.google.com/view/project-flap)]

BibTex:
```
@article{fang2022flap,
      title={Generalization with Lossy Affordances: Leveraging Broad Offline Data for Learning Visuomotor Tasks}, 
      author={Kuan Fang and Patrick Yin and Ashvin Nair and Homer Walke and Gengchen Yan and Sergey Levine},
      journal={Conference on Robot Learning (CoRL)}, 
      year={2022},
}
```

## Installation

### Create Conda Env

Install and use the included anaconda environment.
```
$ conda env create -f flap.yml
$ source activate flap
(flap) $
```

### Dependencies
Download the dependency repos.
- [bullet-manipulation](https://github.com/patrickhaoy/bullet-manipulation) (contains environments): ```git clone https://github.com/patrickhaoy/bullet-manipulation.git```
- [multiworld](https://github.com/vitchyr/multiworld) (contains environments): ```git clone https://github.com/vitchyr/multiworld```
- [rllab](https://github.com/rll/rllab) (contains visualization code):  ```git clone https://github.com/rll/rllab```

Add paths.
```
export PYTHONPATH=$PYTHONPATH:/path/to/multiworld
export PYTHONPATH=$PYTHONPATH:/path/to/doodad
export PYTHONPATH=$PYTHONPATH:/path/to/bullet-manipulation
export PYTHONPATH=$PYTHONPATH:/path/to/bullet-manipulation/bullet-manipulation/roboverse/envs/assets/bullet-objects
export PYTHONPATH=$PYTHONPATH:/path/to/railrl-private
```

## Offline Dataset and Goals

Below we assume the data is stored at `DATA_PATH`.

Download the simulation data and goals from [here](https://drive.google.com/file/d/1stYc26P2OMT3SmDPXzuhj8_5YbUBYdH9/view?usp=share_link). Alternatively, you can recollect a new dataset by running in the root directory of `bullet-manipulation`:
```
python shapenet_scripts/4dof_rotate_td_pnp_push_color_angle_vary_demo_collector_parallel_mixedtask.py --save_path DATA_PATH/ --name drawer_pnp_push_vary_color_angle_mixed_tasks --num_threads 4
```
and resample new goals by running:
```
python shapenet_scripts/presample_goal_with_plan.py --output_dir DATA_PATH/drawer_pnp_push_vary_color_angle_mixed_tasks/ --downsample --test_env_seeds 0 1 2 --timeout_k_steps_after_done 5 --mix_timeout_k
```

## Experiments
Below we assume the trained models are stored at `DATA_CKPT`.

### Offline Pre-Training
To pretrain a VQ-VAE on the simulation dataset, run:
```
python experiments/train_eval_vqvae.py --data_dir DATA_PATH/drawer_pnp_push_vary_color_angle_mixed_tasks --root_dir DATA_CKPT/flap/vqvae
```
To visualize loss curves and reconstructions of images with VQ-VAE, open the tensorboard log file with `tensorboard --logdir DATA_CKPT/flap/vqvae`.

To train offline RL, run:
```
python experiments/train_eval_flap.py 
--data_dir DATA_PATH --local --gpu --save_pretrained 
--name flap/offline_rl
```

To train affordance, run:
```
python experiments/train_eval_affordance.py 
--data_dir DATA_PATH --local --gpu --save_pretrained 
--name flap/affordance
--pretrained_rl_path DATA_CKPT/flap/offline_rl/run0/id0/itr_-1.pt
```

### Online Fine-Tuning
To conduct online fine-tuning with FLAP, run:
```
python experiments/train_eval_flap.py
--data_dir DATA_PATH --local --gpu --save_pretrained
--name flap/flap
--pretrained_rl_path DATA_CKPT/flap/affordance/run0/id0/itr_-1.pt
--arg_binding algo_kwargs.start_epoch=0
--arg_binding algo_kwargs.num_epochs=150
--arg_binding reward_kwargs.obs_type='image' 
--arg_binding reward_kwargs.reward_type='cosine'
--arg_binding eval_contextual_env_kwargs.fraction_planning=0.0
--arg_binding finetune_with_obs_encoder=True
--arg_binding eval_seeds=0
```

### Visualization
During training, the results will be saved to a file called under
```
LOCAL_LOG_DIR/<exp_prefix>/<foldername>
```
 - `LOCAL_LOG_DIR` is the directory set by `railrl.launchers.config.LOCAL_LOG_DIR`
 - `<exp_prefix>` is given either to `setup_logger`.
 - `<foldername>` is auto-generated and based off of `exp_prefix`.
 - inside this folder, you should see a file called `params.pkl`. To visualize a policy, run

You can visualize the results by running `jupyter notebook`, opening `flap_reproduce.ipynb`, and setting `dirs = [LOCAL_LOG_DIR/<exp_prefix>/<foldername>]`.
