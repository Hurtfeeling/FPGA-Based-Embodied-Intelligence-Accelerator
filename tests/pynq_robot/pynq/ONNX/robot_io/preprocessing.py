"""NumPy-only model normalization and pre-tokenized language inputs."""
import json
import numpy as np


class StateActionNormalizer(object):
    """LeRobot MEAN_STD: state=(x-mean)/(std+eps), action=y*std+mean."""
    def __init__(self, stats_json):
        with open(str(stats_json), 'r', encoding='utf-8') as stream:
            stats = json.load(stream)
        self.eps = float(stats.get('eps', 1e-8))
        if not np.isfinite(self.eps) or self.eps <= 0:
            raise ValueError('Normalizer eps must be positive and finite')
        for name in ('state_mean', 'state_std', 'action_mean', 'action_std'):
            value = np.asarray(stats[name], dtype=np.float32)
            if value.shape != (6,) or not np.isfinite(value).all():
                raise ValueError(name + ' must contain six finite values')
            if name.endswith('_std') and (value < 0).any():
                raise ValueError(name + ' cannot be negative')
            setattr(self, name, value)

    def normalize_state(self, state):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (6,) or not np.isfinite(state).all():
            raise ValueError('Raw state must be a finite six-dimensional vector')
        output = ((state-self.state_mean) / (self.state_std+np.float32(self.eps))).reshape(1,6)
        if not np.isfinite(output).all():
            raise ValueError('Normalized state is not finite')
        return np.ascontiguousarray(output, dtype=np.float32)

    def denormalize_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (1,50,6) or not np.isfinite(actions).all():
            raise ValueError('Normalized actions must be finite FP32 [1,50,6]')
        output = actions*self.action_std+self.action_mean
        if not np.isfinite(output).all():
            raise ValueError('Denormalized actions are not finite')
        return np.ascontiguousarray(output, dtype=np.float32)


def read_language_assets(path):
    with np.load(str(path), allow_pickle=False) as archive:
        tokens = archive['language_tokens']
        mask = archive['language_mask']
    if tokens.shape != (1,48) or tokens.dtype != np.int64 or (tokens < 0).any():
        raise ValueError('Language tokens require nonnegative int64 [1,48]')
    if mask.shape != (1,48) or mask.dtype != np.bool_ or not mask.any():
        raise ValueError('Language mask requires nonempty bool [1,48]')
    return dict(language_tokens=np.ascontiguousarray(tokens), language_mask=np.ascontiguousarray(mask))
