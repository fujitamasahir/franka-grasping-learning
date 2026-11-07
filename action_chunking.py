"""Shared data, policy, and chunk-controller code for the preregistered study."""
from pathlib import Path
import json
import hashlib
import time

import numpy as np
import torch
from torch import nn

PROTOCOL_PATH = Path('bc_results/action_chunking_protocol.json')
PROTOCOL_SHA_PATH = Path('bc_results/action_chunking_protocol.sha256')
AUDIT_PATH = Path('bc_results/action_chunking_data_audit.json')
SOURCE_FILES = (Path('demonstrations/franka_expert_train.npz'), Path('demonstrations/franka_expert_wide.npz'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_rollout_seed_set(rows, expected_seeds, label='evaluation'):
    """Check rollout completeness and uniqueness before calculating metrics."""
    def index(rows, label):
        result = {}
        for row in rows:
            seed = int(row['seed'])
            if seed in result:
                raise ValueError(f'Duplicate {label} evaluation seed {seed}')
            result[seed] = row
        return result
    result = index(rows, label)
    if set(result) != set(map(int, expected_seeds)):
        raise ValueError(f'{label} evaluations do not match the preregistered seed set')
    return result


def pair_rollout_rows(baseline_rows, policy_rows, expected_seeds=None):
    """Validate shared deterministic positions before paired success analysis."""
    baseline = {}
    policy = {}
    for row in baseline_rows:
        seed=int(row['seed'])
        if seed in baseline:raise ValueError(f'Duplicate baseline evaluation seed {seed}')
        baseline[seed]=row
    for row in policy_rows:
        seed=int(row['seed'])
        if seed in policy:raise ValueError(f'Duplicate policy evaluation seed {seed}')
        policy[seed]=row
    if set(baseline) != set(policy):
        raise ValueError('Paired baseline and policy evaluations use different positions')
    if expected_seeds is not None and set(baseline) != set(map(int, expected_seeds)):
        raise ValueError('Paired evaluations do not match the preregistered seed set')
    return [(baseline[s], policy[s]) for s in sorted(baseline)]


def verify_protocol():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))
    declared = PROTOCOL_SHA_PATH.read_text(encoding='utf-8').split()[0]
    actual = sha256(PROTOCOL_PATH)
    if actual != declared:
        raise RuntimeError(f'Action chunking protocol SHA mismatch: {actual} != {declared}')
    audit = json.loads(AUDIT_PATH.read_text(encoding='utf-8'))
    if sha256(AUDIT_PATH) != protocol['dataset']['audit_sha256']:
        raise RuntimeError('Dataset audit SHA mismatch')
    for path in SOURCE_FILES:
        expected = protocol['dataset']['sha256'][str(path)]
        if sha256(path) != expected:
            raise RuntimeError(f'Demonstration source changed after preregistration: {path}')
    return protocol, audit


class ActionChunkBC(nn.Module):
    """State-conditioned MLP predicting H XYZ actions and H gripper logits."""
    def __init__(self, horizon):
        super().__init__()
        if horizon not in (1, 4):
            raise ValueError('Only preregistered horizons H=1 and H=4 are supported')
        self.horizon = int(horizon)
        self.backbone = nn.Sequential(
            nn.Linear(12, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.xyz_head = nn.Linear(128, 3 * self.horizon)
        self.grip_head = nn.Linear(128, self.horizon)

    def forward(self, obs):
        feat = self.backbone(obs)
        xyz = torch.tanh(self.xyz_head(feat)).reshape(-1, self.horizon, 3)
        grip_logits = self.grip_head(feat).reshape(-1, self.horizon)
        return xyz, grip_logits


def compute_chunk_loss(model, obs, actions):
    xyz, grip_logits = model(obs)
    xyz_loss = nn.functional.smooth_l1_loss(xyz, actions[:, :, :3])
    grip_target = (actions[:, :, 3] > 0).float()
    grip_loss = nn.functional.binary_cross_entropy_with_logits(grip_logits, grip_target)
    return xyz_loss + 0.25 * grip_loss, xyz_loss, grip_loss


def load_demonstrations():
    """Load and validate the two unsampled, episode-grouped expert sources."""
    protocol, audit = verify_protocol()
    obs_parts, act_parts, id_parts, time_parts = [], [], [], []
    offset = 0
    for source_no, path in enumerate(SOURCE_FILES):
        z = np.load(path)
        obs = z['observations'].astype(np.float32, copy=True)
        actions = z['actions'].astype(np.float32, copy=True)
        old_ids = z['episode_ids'].astype(np.int64, copy=True)
        dt, max_time = float(z['control_dt']), float(z['max_time'])
        if obs.shape != (len(actions), 12) or actions.shape != (len(old_ids), 4):
            raise AssertionError(f'Observation/action alignment invalid: {path}')
        if not np.isfinite(obs).all() or not np.isfinite(actions).all():
            raise AssertionError(f'Non-finite demonstration data: {path}')
        if np.max(np.abs(actions)) > 1.00001:
            raise AssertionError(f'Action values outside normalized range: {path}')
        if not np.all(np.isin(actions[:, 3], (-1.0, 1.0))):
            raise AssertionError(f'Unexpected gripper convention in {path}')
        if not np.isclose(float(z['max_move']), 0.15) or not np.isclose(max_time, 16.0) or not np.isclose(dt, .1):
            raise AssertionError(f'Control configuration mismatch: {path}')
        if not bool(np.all(z['successes'])):
            raise AssertionError(f'Unsuccessful source trajectory present: {path}')
        remap = {int(e): int(e) + offset for e in np.unique(old_ids)}
        ids = np.asarray([remap[int(e)] for e in old_ids], dtype=np.int64)
        expected_episode_ids = set(audit['split']['training_episode_ids'] + audit['split']['validation_episode_ids'])
        if not set(map(int, ids)) <= expected_episode_ids:
            raise AssertionError(f'Remapped episode IDs not in preregistered split: {path}')
        for eid in np.unique(ids):
            ix = np.flatnonzero(ids == eid)
            if ix[-1] - ix[0] + 1 != len(ix):
                raise AssertionError(f'Episode rows are not contiguous: {path} episode {eid}')
            times = obs[ix, 9] * max_time
            if len(times) > 1 and (np.any(np.diff(times) <= 0) or np.any(np.abs(np.diff(times) - dt) > 1e-4)):
                raise AssertionError(f'Temporally discontinuous episode: {path} episode {eid}')
        obs_parts.append(obs); act_parts.append(actions); id_parts.append(ids)
        offset = 200
    observations = np.concatenate(obs_parts)
    actions = np.concatenate(act_parts)
    episode_ids = np.concatenate(id_parts)
    return observations, actions, episode_ids, protocol, audit


def build_windows(observations, actions, episode_ids, split_ids, horizon):
    """Return same temporal start states for H1/H4; reject all boundary/gap crossings."""
    xs, ys, owners = [], [], []
    max_time = 16.0
    dt = 0.1
    split_ids = set(map(int, split_ids))
    for eid in sorted(split_ids):
        ix = np.flatnonzero(episode_ids == eid)
        if not len(ix):
            raise AssertionError(f'Missing split episode {eid}')
        if ix[-1] - ix[0] + 1 != len(ix):
            raise AssertionError(f'Interleaved episode {eid}')
        for start in range(len(ix) - 3):
            window = ix[start:start + 4]
            if not np.all(episode_ids[window] == eid):
                raise AssertionError('Action chunk crosses episode boundary')
            times = observations[window, 9] * max_time
            if np.any(np.abs(np.diff(times) - dt) > 1e-4):
                raise AssertionError('Action chunk spans a temporal gap')
            xs.append(observations[window[0]])
            ys.append(actions[window[:horizon]])
            owners.append(eid)
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(owners, np.int64)


def load_bc_policy(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    horizon = int(checkpoint['horizon'])
    model = ActionChunkBC(horizon)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def predict_chunk(model, obs, obs_mean, obs_std):
    normalized = (np.asarray(obs, dtype=np.float32) - obs_mean) / obs_std
    tensor = torch.from_numpy(normalized).unsqueeze(0)
    with torch.inference_mode():
        xyz, grip_logits = model(tensor)
    xyz = xyz[0].cpu().numpy()
    grip = np.where(grip_logits[0].cpu().numpy() >= 0, 1.0, -1.0).astype(np.float32)
    return np.concatenate([xyz, grip[:, None]], axis=1).astype(np.float32)


class BCActionChunkController:
    """Caches H actions with a separately configurable runtime replan interval."""
    def __init__(self, model, obs_mean, obs_std, horizon, replan_interval=None):
        self.model = model
        self.obs_mean = np.asarray(obs_mean, np.float32)
        self.obs_std = np.asarray(obs_std, np.float32)
        self.horizon = int(horizon)
        # Preserve legacy behavior: an omitted interval consumes the full chunk.
        self.replan_interval = self.horizon if replan_interval is None else int(replan_interval)
        if self.replan_interval not in (1, 2, 4) or self.replan_interval > self.horizon:
            raise ValueError('replan_interval must be one of 1, 2, or 4 and no greater than horizon')
        self.cached_actions = None
        self.chunk_index = 0
        self.inference_calls = 0
        self.inference_latencies = []

    def plan(self, obs):
        start = time.perf_counter()
        chunk = predict_chunk(self.model, obs, self.obs_mean, self.obs_std)
        self.inference_latencies.append(time.perf_counter() - start)
        if chunk.shape != (self.horizon, 4):
            raise AssertionError(f'Expected planned action shape {(self.horizon,4)}, got {chunk.shape}')
        self.cached_actions = chunk.copy()
        self.chunk_index = 0
        self.inference_calls += 1
        return self.cached_actions[0].copy()

    @property
    def planned_action(self):
        if self.cached_actions is None:
            raise RuntimeError('Controller must be reset before requesting an action')
        return self.cached_actions[self.chunk_index].copy()

    def reset(self, obs):
        self.inference_calls = 0
        self.inference_latencies = []
        return self.plan(obs)

    def advance(self, next_obs, terminated=False, truncated=False):
        if terminated or truncated:
            return False
        if self.chunk_index + 1 >= self.replan_interval:
            self.plan(next_obs)
            return True
        self.chunk_index += 1
        return False

    def state_dict(self):
        return {
            'horizon': self.horizon,
            'replan_interval': self.replan_interval,
            'cached_actions': None if self.cached_actions is None else self.cached_actions.copy(),
            'chunk_index': int(self.chunk_index),
            'inference_calls': int(self.inference_calls),
            'inference_latencies': list(self.inference_latencies),
        }

    def load_state_dict(self, state):
        if int(state['horizon']) != self.horizon:
            raise ValueError('Chunk controller horizon differs from saved state')
        saved_interval = int(state.get('replan_interval', self.horizon))
        if saved_interval != self.replan_interval:
            raise ValueError('Chunk controller replan interval differs from saved state')
        cached = state['cached_actions']
        self.cached_actions = None if cached is None else np.asarray(cached, np.float32).copy()
        self.chunk_index = int(state['chunk_index'])
        self.inference_calls = int(state['inference_calls'])
        self.inference_latencies = list(state.get('inference_latencies', []))
        if self.cached_actions is not None and not 0 <= self.chunk_index < self.horizon:
            raise ValueError('Invalid cached chunk index in saved state')
