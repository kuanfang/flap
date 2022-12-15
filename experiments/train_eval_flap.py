import os
import glob
from absl import app
from absl import flags

from roboverse.envs.sawyer_drawer_pnp_push import SawyerDrawerPnpPush

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants
from rlkit.torch.sac.policies import GaussianPolicy
from rlkit.torch.sac.policies import GaussianTwoChannelCNNPolicy

from rlkit.envs.drawer_pnp_push_commands import drawer_pnp_push_commands  # NOQA
from rlkit.learning.flap import flap_experiment
from rlkit.learning.flap import process_args
from rlkit.utils import arg_util
from rlkit.utils.logging import logger as logging

from rlkit.networks.encoding_networks import EncodingGaussianPolicy  # NOQA
from rlkit.networks.encoding_networks import EncodingGaussianPolicyV2  # NOQA


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
        env_kwargs=dict(
            test_env=True,
        ),
        policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256, 256],
            std=None,
            max_log_std=-1,
            min_log_std=-2,
            std_architecture='values',
            output_activation=None,
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        vf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        obs_encoder_kwargs=dict(),
        network_type='vqvae',

        trainer_kwargs=dict(
            discount=0.995,
            lr=3E-4,
            reward_scale=1,

            soft_target_tau=5e-2,

            kld_weight=0.1,

            reward_transform_kwargs=dict(m=1, b=0),
            terminal_transform_kwargs=None,

            beta=0.01,
            quantile=0.8,
            clip_score=100,

            fraction_generated_goals=0.0,

            min_value=None,
            max_value=None,

            end_to_end=False,
            affordance_weight=100.,

            use_encoding_reward_online=True,
            encoding_reward_thresh=.95,

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

            bc=False,
        ),

        max_path_length=400,
        algo_kwargs=dict(
            batch_size=256,
            start_epoch=-100,  # offline epochs
            num_epochs=0,  # online epochs

            num_eval_steps_per_epoch=2000,
            num_expl_steps_per_train_loop=2000,
            num_trains_per_train_loop=1000,
            num_online_trains_per_train_loop=2000,
            min_num_steps_before_training=4000,

            eval_epoch_freq=5,
            offline_expl_epoch_freq=5,
        ),
        replay_buffer_kwargs=dict(
            fraction_next_context=0.1,
            fraction_future_context=0.6,
            fraction_foresight_context=0.0,
            fraction_perturbed_context=0.0,
            fraction_distribution_context=0.0,
            max_size=int(1E6),
        ),
        online_offline_split=True,
        reward_kwargs=dict(
            obs_type='latent',
            reward_type='sparse',
            epsilon=3.0,
            terminate_episode=False,
        ),
        online_offline_split_replay_buffer_kwargs=dict(
            offline_replay_buffer_kwargs=dict(
                fraction_next_context=0.1,
                fraction_future_context=0.9,  # For offline data only.
                fraction_foresight_context=0.0,
                fraction_perturbed_context=0.0,
                fraction_distribution_context=0.0,
                max_size=int(6E5),
            ),
            online_replay_buffer_kwargs=dict(
                fraction_next_context=0.1,
                fraction_future_context=0.6,
                fraction_foresight_context=0.0,
                fraction_perturbed_context=0.0,
                fraction_distribution_context=0.0,
                max_size=int(4E5),
            ),
            sample_online_fraction=0.6
        ),

        save_video=True,
        expl_save_video_kwargs=dict(
            save_video_period=25,
            pad_color=0,
        ),
        eval_save_video_kwargs=dict(
            save_video_period=25,
            pad_color=0,
        ),

        pretrained_vae_path=vqvae,

        path_loader_kwargs=dict(
            delete_after_loading=True,
            recompute_reward=True,
            demo_paths=demo_paths,
            split_max_steps=None,
            demo_train_split=0.95,
        ),

        renderer_kwargs=dict(
            create_image_format='HWC',
            output_image_format='CWH',
            flatten_image=True,
            width=48,
            height=48,
        ),

        add_env_demos=False,
        add_env_offpolicy_data=False,

        load_demos=True,

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

        use_expl_planner=True,
        expl_planner_type='mppi',
        expl_planner_kwargs=dict(
            cost_mode='l2_vf',
            buffer_size=100,
            num_levels=1,
            min_dt=30,
        ),
        expl_planner_scripted_goals=None,
        expl_contextual_env_kwargs=dict(
            num_planning_steps=4,
            fraction_planning=1.0,
            subgoal_timeout=60,
            subgoal_reaching_thresh=None,
            mode='o',
        ),

        use_eval_planner=True,
        eval_planner_type='mppi',
        eval_planner_kwargs=dict(
            cost_mode='l2_vf',
            buffer_size=100,
            num_levels=1,
            min_dt=30,
        ),
        eval_planner_scripted_goals=None,
        eval_contextual_env_kwargs=dict(
            num_planning_steps=4,
            fraction_planning=1.0,
            subgoal_timeout=60,
            subgoal_reaching_thresh=None,
            mode='o',
        ),

        scripted_goals=None,

        reset_interval=1,
        expl_reset_interval=0,

        launcher_config=dict(
            unpack_variant=True,
            region='us-west-1',  # HERE
        ),
        logger_config=dict(
            snapshot_mode='gap',
            snapshot_gap=50,
        ),

        trainer_type='vib',
        network_version=0,

        use_image=False,
        finetune_with_obs_encoder=False,
        pretrained_rl_path=pretrained_rl_path,
        eval_seeds=0,

        # VIB
        obs_encoding_dim=64,
        affordance_encoding_dim=8,

        policy_class_name='v1',
        use_encoder_in_policy=True,
        fix_encoder_online=True,

        # Video
        num_video_columns=5,
        save_paths=False,

    )

    return default_variant


def process_variant(dataset, variant, data_path):  # NOQA
    # Error checking
    assert variant['algo_kwargs']['start_epoch'] % variant['algo_kwargs']['eval_epoch_freq'] == 0  # NOQA
    if variant['algo_kwargs']['start_epoch'] < 0:
        assert variant['algo_kwargs']['start_epoch'] % variant['algo_kwargs']['offline_expl_epoch_freq'] == 0  # NOQA
    if variant['pretrained_rl_path'] is not None:
        assert variant['algo_kwargs']['start_epoch'] == 0
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
    variant['presampled_goal_kwargs']['expl_goals'] = eval_goals
    variant['presampled_goal_kwargs']['training_goals'] = eval_goals

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
        variant['policy_class'] = GaussianTwoChannelCNNPolicy

        variant['obs_encoder_kwargs'] = dict()

        for demo_path in variant['path_loader_kwargs']['demo_paths']:
            demo_path['use_latents'] = False

        variant['replay_buffer_kwargs']['max_size'] = int(5E5)
        variant['online_offline_split_replay_buffer_kwargs']['offline_replay_buffer_kwargs']['max_size'] = int(2E5)  # NOQA
        variant['online_offline_split_replay_buffer_kwargs']['online_replay_buffer_kwargs']['max_size'] = int(3E5)  # NOQA

    ########################################
    # Misc.
    ########################################
    if variant['reward_kwargs']['reward_type'] in [
            'sparse', 'onion', 'highlevel']:
        variant['trainer_kwargs']['max_value'] = 0.0
        variant['trainer_kwargs']['min_value'] = -1. / (
            1. - variant['trainer_kwargs']['discount'])

    if variant['use_encoder_in_policy']:
        if variant['policy_class_name'] == 'v1':
            variant['policy_class'] = EncodingGaussianPolicy
        elif variant['policy_class_name'] == 'v2':
            variant['policy_class'] = EncodingGaussianPolicyV2
        else:
            raise ValueError


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

    run_variants(flap_experiment,
                 variants,
                 run_id=FLAGS.run_id,
                 process_args_fn=process_args)


if __name__ == '__main__':
    app.run(main)
