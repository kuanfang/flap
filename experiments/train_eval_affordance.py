import os
import glob
from absl import app
from absl import flags

from roboverse.envs.sawyer_drawer_pnp_push import SawyerDrawerPnpPush

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

from rlkit.envs.drawer_pnp_push_commands import drawer_pnp_push_commands  # NOQA
from rlkit.learning.affordance import affordance_experiment
from rlkit.learning.affordance import process_args
from rlkit.utils import arg_util
from rlkit.utils.logging import logger as logging

flags.DEFINE_string('data_dir', './data', '')
flags.DEFINE_string('dataset', 'drawer_pnp_push_mixed_exclude_task', '')
flags.DEFINE_string('name', None, '')
flags.DEFINE_string('base_log_dir', None, '')
flags.DEFINE_string('pretrained_rl_path', None, '')
flags.DEFINE_bool('local', True, '')
flags.DEFINE_bool('gpu', True, '')
flags.DEFINE_bool('save_pretrained', True, '')
flags.DEFINE_bool('debug', False, '')
flags.DEFINE_bool('script', False, '')
flags.DEFINE_integer('run_id', 0, '')
flags.DEFINE_multi_string(
    'arg_binding', None, 'Variant binding to pass through.')

FLAGS = flags.FLAGS


def get_paths(data_dir, dataset):  # NOQA
    # dataset = 'env6_vary_exclusive'

    # VAL Data
    if dataset is None:  # NOQA
        raise ValueError

    elif dataset == 'drawer_pnp_push_mixed_exclude_task':
        data_path = 'drawer_pnp_push_vary_color_angle_mixed_tasks/'  # NOQA
        data_path = os.path.join(data_dir, data_path)
        paths = glob.glob(os.path.join(data_path, '*demos.pkl'))
        exclude_paths = glob.glob(
            os.path.join(data_path, 'scene0_*drawer*demos.pkl'))
        demo_paths = [
            dict(path=path,
                 obs_dict=True,
                 is_demo=True,
                 use_latents=True)
            for path in paths if path not in exclude_paths]
        logging.info('Number of demonstration files: %d' % len(demo_paths))

    else:
        assert False

    logging.info('data_path: %s', data_path)

    return data_path, demo_paths


def get_default_variant(data_path, demo_paths, pretrained_rl_path):
    vqvae = os.path.join(data_path, 'pretrained')

    default_variant = dict(
        imsize=48,
        env_type='td_pnp_push',
        policy_kwargs=dict(
            hidden_sizes=[256, 256],
            std=None,
            max_log_std=-1,
            min_log_std=-2,
            std_architecture='values',
            output_activation=None,
        ),
        env_kwargs=dict(
            test_env=True,
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        vf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        obs_encoder_kwargs=dict(
            hidden_dim=128,
            use_normalization=False,
        ),
        affordance_kwargs=dict(
            hidden_dim=256,
        ),
        network_type=None,

        trainer_kwargs=dict(
            discount=0.995,
            lr=3E-4,
            reward_scale=1,

            kld_weight=0.1,

            reward_transform_kwargs=dict(m=1, b=0),
            terminal_transform_kwargs=None,

            affordance_weight=10000.,
            affordance_beta=200,

            augment_params={
                'RandomResizedCrop': dict(
                    scale=(0.9, 1.0),
                    ratio=(0.9, 1.1),
                ),
                'ColorJitter': dict(
                    brightness=(0.75, 1.25),
                    contrast=(0.9, 1.1),
                    saturation=(0.9, 1.1),
                    hue=(-0.1, 0.1),
                ),
                'RandomCrop': dict(
                    padding=4,
                    padding_mode='edge'
                ),
            },
            augment_order=['RandomResizedCrop', 'ColorJitter'],
            augment_probability=0.0,

            use_estimated_logvar=False,
            use_sampled_encodings=False,
            noise_level=0.0,
        ),

        max_path_length=400,
        algo_kwargs=dict(
            # batch_size=256,
            batch_size=64,
            # start_epoch=-100,  # offline epochs
            start_epoch=-1000,  # offline epochs
            num_epochs=0,  # online epochs

            num_eval_steps_per_epoch=0,
            num_expl_steps_per_train_loop=0,
            num_trains_per_train_loop=1000,
            num_online_trains_per_train_loop=None,
            min_num_steps_before_training=0,

            eval_epoch_freq=float('inf'),
            offline_expl_epoch_freq=float('inf'),
        ),
        replay_buffer_kwargs=dict(
            fraction_future_context=1.0,
            fraction_next_context=0.0,
            fraction_last_context=0.,
            fraction_foresight_context=0.0,
            fraction_perturbed_context=0.0,
            fraction_distribution_context=0.0,
            min_future_dt=10,
            max_future_dt=20,
            max_previous_dt=None,
            max_last_dt=None,
            max_size=int(1E6),
        ),
        reward_kwargs=dict(
            obs_type='latent',
            reward_type='sparse',
            epsilon=3.0,
            terminate_episode=False,
        ),

        reset_keys_map=dict(
            image_observation='initial_latent_state'
        ),
        pretrained_vae_path=vqvae,

        path_loader_kwargs=dict(
            delete_after_loading=True,
            recompute_reward=True,
            demo_paths=demo_paths,
            split_max_steps=None,
            demo_train_split=0.95,
            min_path_length=15,
        ),

        renderer_kwargs=dict(
            create_image_format='HWC',
            output_image_format='CWH',
            flatten_image=True,
            width=48,
            height=48,
        ),

        add_env_demos=False,

        evaluation_goal_sampling_mode='presampled_images',
        exploration_goal_sampling_mode='presampled_images',
        training_goal_sampling_mode='presample_latents',

        presampled_goal_kwargs=dict(
            eval_goals='',  # HERE
            eval_goals_kwargs={},
            expl_goals='',
            expl_goals_kwargs={},
            training_goals='',
            training_goals_kwargs={},
        ),

        launcher_config=dict(
            unpack_variant=True,
            region='us-west-1',  # HERE
        ),
        logger_config=dict(
            snapshot_mode='gap',
            snapshot_gap=50,
        ),

        use_image=False,
        finetune_with_obs_encoder=True,
        pretrained_rl_path=pretrained_rl_path,
        eval_seeds=0,
        num_demos=20,

        # VIB
        obs_encoding_dim=64,
        affordance_encoding_dim=8,

    )

    return default_variant


def process_variant(dataset, variant, data_path):  # NOQA
    # Error checking
    if not variant['use_image']:
        assert variant['trainer_kwargs']['augment_probability'] == 0.0
    env_type = variant['env_type']

    ########################################
    # Set the eval_goals.
    ########################################
    if 'eval_seeds' in variant.keys():
        eval_seed_str = f"_seed{variant['eval_seeds']}"
    else:
        eval_seed_str = ''

    eval_goals = os.path.join(data_path, f'{env_type}_goals{eval_seed_str}.pkl')  # NOQA
    ########################################
    # Goal sampling modes.
    ########################################
    variant['presampled_goal_kwargs']['eval_goals'] = eval_goals

    variant['exploration_goal_sampling_mode'] = 'presampled_images'
    variant['training_goal_sampling_mode'] = 'presampled_images'
    ########################################
    # Environments.
    ########################################
    variant['env_class'] = SawyerDrawerPnpPush
    variant['env_kwargs']['downsample'] = True
    variant['env_kwargs']['env_obs_img_dim'] = 196
    variant['env_kwargs']['test_env_command'] = (
        drawer_pnp_push_commands[variant['eval_seeds']])

    ########################################
    # Image.
    ########################################
    if variant['use_image']:
        variant['obs_encoder_kwargs'] = dict()

        for demo_path in variant['path_loader_kwargs']['demo_paths']:
            demo_path['use_latents'] = False

        variant['replay_buffer_kwargs']['max_size'] = int(5E5)

def main(_):
    data_path, demo_paths = get_paths(data_dir=FLAGS.data_dir,
                                      dataset=FLAGS.dataset)
    default_variant = get_default_variant(
        data_path,
        demo_paths,
        FLAGS.pretrained_rl_path,
    )

    sweeper = hyp.DeterministicHyperparameterSweeper(
        {},
        default_parameters=default_variant,
    )

    logging.info('arg_binding: ')
    logging.info(FLAGS.arg_binding)

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variant = arg_util.update_bindings(variant,
                                           FLAGS.arg_binding,
                                           check_exist=True)
        process_variant(FLAGS.dataset, variant, data_path)
        variants.append(variant)

    run_variants(affordance_experiment,
                 variants,
                 run_id=FLAGS.run_id,
                 process_args_fn=process_args)


if __name__ == '__main__':
    app.run(main)