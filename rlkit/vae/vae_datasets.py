import os  # NOQA
from absl import logging  # NOQA

import numpy as np
import joblib

from torch.utils.data import Dataset

from rlkit.utils import image_util


def transform_uint8_to_float32(data):
    return data.astype(np.float32) / 255 - 0.5


class VaeDataset(Dataset):

    def __init__(self,  # NOQA
                 data_path,
                 encoding_path=None,
                 train=None,
                 vqvae=None,
                 crop=None,
                 transform=transform_uint8_to_float32,
                 preprocess_image=False,
                 augment_image=False,
                 channel_first=True,
                 is_val_format=True,
                 num_skills=6,
                 dt=1,
                 dt_tolerance=0,
                 ):

        # Load the data and the encoding.
        logging.info('Loading the dataset from %s...', data_path)
        try:
            data_dict = np.load(data_path, allow_pickle=True)
            data_dict = data_dict.item()
        except Exception:
            data_dict = joblib.load(data_path)

        if is_val_format:
            self.data = data_dict['observations']
            self.data = np.reshape(
                self.data,
                list(self.data.shape[:2]) + [3, 48, 48])
            self.data = np.transpose(self.data, [0, 1, 4, 3, 2])

        assert self.data.shape[-1] == 3, 'The shape of the data: %r' % (
            self.data.shape)

        if encoding_path is None or augment_image:
            self.encoding = None
        else:
            logging.info(
                'Loading the vqvae encoding from %s...', encoding_path)
            try:
                self.encoding = np.load(encoding_path, allow_pickle=True)
            except Exception:
                self.encoding = joblib.load(encoding_path)

        if 'num_steps' in data_dict:
            self.num_steps_per_traj = data_dict['num_steps']
            assert self.num_steps_per_traj.min() > 0
        else:
            self.num_steps_per_traj = None

        # Preprocess the data.
        if crop is not None:
            self.data = self.data[..., :crop[0], :crop[1], :]

        if channel_first:
            if self.data.ndim == 4:
                self.data = np.transpose(self.data, (0, 3, 1, 2))
            elif self.data.ndim == 5:
                self.data = np.transpose(self.data, (0, 1, 4, 2, 3))

        if preprocess_image:
            self.data = transform_uint8_to_float32(self.data)

        self.transform = transform

        self.num_samples = self.data.shape[0]
        self.num_steps = self.data.shape[1]

        self.dt = dt

        if dt_tolerance is None:
            self.dt_tolerance = int(dt / 2)
        else:
            self.dt_tolerance = dt_tolerance

    def __len__(self):
        return self.data.shape[0]

    def _get_num_steps(self, idx):
        if self.num_steps_per_traj is not None:
            num_steps = self.num_steps_per_traj[idx]
        else:
            num_steps = self.num_steps

        return num_steps

    def __getitem__(self, idx):
        ret = {}

        num_steps = self._get_num_steps(idx)
        assert num_steps > 0

        t = int(np.random.uniform(0, num_steps))

        data_i = self.data[idx, t]

        if self.transform is not None:
            data_i = self.transform(data_i)

        ret['s'] = data_i

        if self.encoding is not None:
            encoding_i = self.encoding[idx, t]
            ret['h'] = encoding_i

        return ret


class VaeMultistepDataset(VaeDataset):

    def __init__(self,
                 data_path,
                 encoding_path=None,
                 train=None,
                 crop=None,
                 transform=transform_uint8_to_float32,
                 preprocess_image=False,
                 augment_image=False,
                 dt=1,
                 dt_tolerance=1,
                 num_goals=4,
                 channel_first=True,
                 is_val_format=True,
                 random_reverse=False,
                 ):
        super(VaeMultistepDataset, self).__init__(
            data_path=data_path,
            encoding_path=encoding_path,
            train=train,
            transform=transform,
            crop=crop,
            preprocess_image=preprocess_image,
            augment_image=augment_image,
            channel_first=channel_first,
            is_val_format=is_val_format,
            dt=dt,
            dt_tolerance=dt_tolerance,
        )

        assert len(self.data.shape) == 5

        self.num_goals = num_goals

        self.random_reverse = random_reverse

        assert self.num_steps >= self.num_goals * self.dt, (
            'data_shape: %r, num_steps: %d, num_goals: %d, dt: %d'
            % (self.data.shape, self.num_steps, self.num_goals, self.dt))

        if self.num_steps_per_traj is not None:
            min_steps = self.num_steps_per_traj.min()
            assert (
                min_steps > 1 + self.num_goals * (self.dt - self.dt_tolerance)
            ), (
                ('data_shape: %r, min_steps: %d, '
                 'num_goals: %d, dt: %d, dt_tolerance: %d')
                % (self.data.shape, min_steps,
                   self.num_goals, self.dt, self.dt_tolerance))

    def __getitem__(self, idx):
        ret = {}

        num_steps = self._get_num_steps(idx)
        assert (num_steps > 1 + self.num_goals * (self.dt - self.dt_tolerance))

        # Random reverse.
        reverse = False
        if self.random_reverse:
            if np.random.rand() > 0.5:
                reverse = True

        # Choose the time steps.
        valid_sample = False
        num_attempts = 0
        t = None
        while not valid_sample:
            if num_attempts > 10000:
                raise ValueError(
                    'Invalid sequence of shape %r of index %d, '
                    'with num_goals: %d, dt: %d, dt_tolerance: %d'
                    % (self.data[idx].shape, idx,
                       self.num_goals, self.dt, self.dt_tolerance))

            t_list = [0]
            t = 0
            for i in range(self.num_goals):
                last_t = t
                high = min(last_t + self.dt, num_steps - 1) + 1
                low = max(last_t + self.dt - self.dt_tolerance, last_t + 1)
                if low >= high:
                    break
                t = int(np.random.uniform(low, high))
                t_list.append(t)

            if len(t_list) == 1 + self.num_goals:
                valid_sample = True

            num_attempts += 1

        t0 = int(np.random.uniform(0, num_steps - t))
        t_list = [t0 + t for t in t_list]

        if reverse:
            t_list.reverse()

        # Select the states.
        data_i = self.data[idx, t_list]

        if self.transform is not None:
            data_i = self.transform(data_i)

        # if reverse:
        #     data_i = np.flip(data_i, axis=0).copy()

        ret['s'] = data_i

        # Select the encodings.
        if self.encoding is not None:
            encoding_i = self.encoding[idx, t_list]

            # if reverse:
            #     encoding_i = np.flip(encoding_i, axis=0).copy()

            ret['h'] = encoding_i

        ret['t'] = np.array(t_list)

        return ret


class VaeAnyStepDataset(VaeDataset):

    def __init__(self,
                 data_path,
                 encoding_path=None,
                 train=None,
                 crop=None,
                 transform=transform_uint8_to_float32,
                 preprocess_image=False,
                 augment_image=False,
                 dt=None,
                 dt_tolerance=None,
                 num_goals=1,
                 channel_first=True,
                 is_val_format=True,
                 ):
        super(VaeAnyStepDataset, self).__init__(
            data_path=data_path,
            encoding_path=encoding_path,
            train=train,
            transform=transform,
            crop=crop,
            preprocess_image=preprocess_image,
            augment_image=augment_image,
            channel_first=channel_first,
            is_val_format=is_val_format,
            dt=None,
            dt_tolerance=0,
        )

        assert len(self.data.shape) == 5

    def __getitem__(self, idx):
        ret = {}

        num_steps = self._get_num_steps(idx)

        t0 = int(np.random.uniform(0, num_steps))
        t1 = int(np.random.uniform(0, num_steps))
        t_list = [t0, t1]

        # Select the states.
        data_i = self.data[idx, t_list]

        if self.transform is not None:
            data_i = self.transform(data_i)

        # if reverse:
        #     data_i = np.flip(data_i, axis=0).copy()

        ret['s'] = data_i

        # Select the encodings.
        if self.encoding is not None:
            encoding_i = self.encoding[idx, t_list]

            # if reverse:
            #     encoding_i = np.flip(encoding_i, axis=0).copy()

            ret['h'] = encoding_i

        ret['t'] = np.array(t_list)

        return ret


class VaeFinalGoalDataset(VaeDataset):

    def __init__(self,
                 data_path,
                 encoding_path=None,
                 train=None,
                 vqvae=None,
                 crop=None,
                 transform=transform_uint8_to_float32,
                 preprocess_image=False,
                 augment_image=False,
                 dt=1,
                 dt_tolerance=10,
                 num_goals=1,
                 channel_first=True,
                 is_val_format=True,
                 random_reverse=False,
                 ):
        super(VaeFinalGoalDataset, self).__init__(
            data_path=data_path,
            encoding_path=encoding_path,
            train=train,
            vqvae=vqvae,
            transform=transform,
            crop=crop,
            preprocess_image=preprocess_image,
            augment_image=augment_image,
            channel_first=channel_first,
            is_val_format=is_val_format,
            dt=dt,
            dt_tolerance=dt_tolerance,
        )

        assert len(self.data.shape) == 5
        assert num_goals == 1

        self.random_reverse = random_reverse

    def __getitem__(self, idx):
        ret = {}

        num_steps = self._get_num_steps(idx)
        assert num_steps > 1 + self.dt_tolerance

        reverse = False
        if self.random_reverse:
            if np.random.rand() > 0.5:
                reverse = True

        t0 = int(np.random.uniform(0, self.dt_tolerance))
        t1 = int(np.random.uniform(max(0, num_steps - self.dt_tolerance),
                                   num_steps - 1))
        t_list = [t0, t1]

        if reverse:
            t_list.reverse()

        # Select the states.
        data_i = self.data[idx, t_list]

        if self.transform is not None:
            data_i = self.transform(data_i)

        ret['s'] = data_i

        # Select the encodings.
        if self.encoding is not None:
            encoding_i = self.encoding[idx, t_list]
            ret['h'] = encoding_i

        ret['t'] = np.array(t_list)

        return ret


# class VaeClassifierDataset(VaeDataset):
#
#     def __init__(self,
#                  data_path,
#                  encoding_path=None,
#                  train=None,
#                  crop=None,
#                  transform=transform_uint8_to_float32,
#                  preprocess_image=False,
#                  augment_image=False,
#                  dt=1,
#                  dt_tolerance=0,
#                  channel_first=True,
#                  is_val_format=True,
#                  ):
#         super(VaeClassifierDataset, self).__init__(
#             data_path=data_path,
#             encoding_path=encoding_path,
#             train=train,
#             transform=transform,
#             crop=crop,
#             preprocess_image=preprocess_image,
#             augment_image=augment_image,
#             channel_first=channel_first,
#             is_val_format=is_val_format,
#             dt=dt,
#             dt_tolerance=dt_tolerance,
#         )
#
#         assert len(self.data.shape) == 5
#
#     def __getitem__(self, idx):
#         t0 = int(np.random.uniform(0, self.num_steps - 1))
#
#         t1_candidates = [
#             t for t in range(0, self.num_steps - 1)
#             if np.abs(t - t0) <= self.dt]
#         t2_candidates = [
#             t for t in range(0, self.num_steps - 1)
#             if np.abs(t - t0) > self.dt]
#
#         t1 = np.random.choice(t1_candidates)
#         t2 = np.random.choice(t2_candidates)
#
#         data_i = np.stack(
#             [self.data[idx, t0, ...],
#              self.data[idx, t1, ...],
#              self.data[idx, t2, ...]],
#             axis=0)
#
#         if self.transform is not None:
#             data_i = self.transform(data_i)
#
#         if self.encoding is None:
#             return data_i
#         else:
#             encoding_i = np.stack(
#                 [self.encoding[idx, t0, ...],
#                  self.encoding[idx, t1, ...],
#                  self.encoding[idx, t2, ...]],
#                 axis=0)
#             return {
#                 's': data_i,
#                 'h': encoding_i,
#                 'dt1': np.abs(t1 - t0),
#                 'dt2': np.abs(t2 - t0),
#             }

class VaeGCPDataset(VaeDataset):

    def __init__(self,
                 data_path,
                 encoding_path=None,
                 train=None,
                 crop=None,
                 transform=transform_uint8_to_float32,
                 preprocess_image=False,
                 dt=1,
                 dt_tolerance=1,
                 num_goals=4,
                 channel_first=True,
                 is_val_format=True,
                 ):
        super(VaeGCPDataset, self).__init__(
            data_path=data_path,
            encoding_path=encoding_path,
            train=train,
            transform=transform,
            crop=crop,
            preprocess_image=preprocess_image,
            channel_first=channel_first,
            is_val_format=is_val_format,
            dt=dt,
            dt_tolerance=dt_tolerance,
        )

        assert len(self.data.shape) == 5
        assert np.log2(num_goals).is_integer()

        self.num_steps = self.data.shape[1]
        self.num_goals = num_goals
        self.num_levels = int(np.log2(num_goals))
        assert self.num_steps >= self.num_goals * (
            self.dt + self.dt_tolerance)

    def __getitem__(self, idx):
        ret = {}

        # Choose the time steps.

        # Random
        # t_list = [0, self.num_steps - 1]
        # for level in range(self.num_levels):
        #     offset = self.num_levels - level
        #     t_list_copy = t_list.copy()
        #     for t_init, t_goal in pairwise(t_list_copy):
        #         t = np.random.randint(t_init+offset, t_goal-offset+1)
        #         t_list.append(t)
        #     t_list.sort()

        # Naive
        # t_list = range(0, self.num_steps, self.num_steps // self.num_goals)

        # Dt_tolerance
        t_list = [0]
        t = 0
        for i in range(self.num_goals):
            dt = self.num_steps // self.num_goals + int(
                np.random.uniform(-self.dt_tolerance, self.dt_tolerance))
            t = t + dt
            t_list.append(t)

        t0 = int(np.random.uniform(0, self.num_steps - t))
        t_list = [t0 + t for t in t_list]

        # Select the states.
        data_i = self.data[idx, t_list]

        if self.transform is not None:
            data_i = self.transform(data_i)

        ret['s'] = data_i

        # Select the encodings.
        if self.encoding is not None:
            encoding_i = self.encoding[idx, t_list]

            ret['h'] = encoding_i

        return ret


# def pairwise(iterable):
#     "s -> (s0,s1), (s1,s2), (s2, s3), ..."
#     a, b = itertools.tee(iterable)
#     next(b, None)
#     return zip(a, b)
