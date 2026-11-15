"""Standalone CPU acceptance: python tests/test_tensor_core.py (no pytest needed)."""

import subprocess
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

from supermimic.learners.policy_learner import BpttPolicyLearner, BpttPolicyLearnerCfg
from supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
from supermimic.models import Dynamics, _CudaGraphNormalization, build_mlp
from supermimic.representations import create_representation
from supermimic.utils.analytic_ik import AnalyticIK
from supermimic.utils.math import matrix_from_quat


def snapshot(module):
  return [p.detach().clone() for p in module.parameters()]


def changed(before, module):
  return any(not torch.equal(a, b) for a, b in zip(before, module.parameters(), strict=True))


class Actor(torch.nn.Module):
  def __init__(self, width):
    super().__init__()
    self.net = build_mlp(width, (16, 16, 16), 23)

  def forward(self, obs):
    return self.net(obs['actor'])


class TensorRollout:
  """Synthetic differentiable fixture using the real Dynamics integration."""
  num_envs = 2
  num_actions = 23
  max_step_reward = float('inf')

  def __init__(self, dynamics, representation, initial):
    self.dynamics, self.representation, self.initial = dynamics, representation, initial

  def sample_initial_states(self):
    return TensorDict({}, batch_size=[2])

  def reset(self, batch):
    self.state = self.representation.represent(self.initial)
    self.aux = self.representation.extract_aux_state(self.initial)
    return self.obs()

  def obs(self):
    return TensorDict({'actor': self.state, 'critic': self.state}, batch_size=[2])

  def step(self, actions):
    self.aux, self.state = self.dynamics(self.state, actions, self.aux, torch.zeros(2, 1))
    return self.obs(), -self.state.square().mean(-1), torch.zeros(2), {}


class TensorCoreAcceptance(unittest.TestCase):
  def test_import_does_not_load_any_simulator(self):
    code = '''
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('mjlab','mujoco','isaaclab','isaacsim','omni'):
   raise AssertionError('Simulator import attempted: '+fullname)
sys.meta_path.insert(0, Guard())
import supermimic.models, supermimic.representations, supermimic.replay_buffer
import supermimic.sim_context, supermimic.tasks.task_spec
import supermimic.learners.policy_learner, supermimic.learners.wm_learner
import supermimic.envs.world_model_env, supermimic.envs.multi_task_world_model_env
assert not any(n.split('.')[0] in ('mjlab','mujoco','isaaclab','isaacsim','omni') for n in sys.modules)
'''
    subprocess.run([sys.executable, '-c', code], check=True)

  def test_normalizer_buffer_address_and_values(self):
    norm = _CudaGraphNormalization(3)
    address = norm._std.data_ptr()
    norm.update(torch.randn(12, 3))
    first = norm._std.clone()
    norm.update(torch.randn(12, 3) * 5)
    self.assertEqual(norm._std.data_ptr(), address)
    self.assertTrue(torch.isfinite(norm._std).all())
    self.assertFalse(torch.equal(first, norm._std))

  def test_single_hinge_constants_solver_and_gradient(self):
    eye = torch.eye(3, dtype=torch.float64)
    zeros = torch.zeros(1, 3, dtype=torch.float64)
    ik = AnalyticIK.from_constants(parent=torch.tensor([0]), child=torch.tensor([1]),
      A=eye[None], B=eye[None], sign=torch.tensor([[1., 0., 0.]], dtype=torch.float64),
      qpos0=zeros, lo=torch.tensor([[-3.,0.,0.]], dtype=torch.float64),
      hi=torch.tensor([[3.,0.,0.]], dtype=torch.float64), perm=torch.tensor([0]))
    angle = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    c, s, z, one = angle.cos(), angle.sin(), angle*0, angle*0+1
    rot = torch.stack([one,z,z,z,c,-s,z,s,c]).reshape(3,3)
    rotations = torch.stack([eye, rot])[None]
    velocities = torch.tensor([[[0.,0.,0.],[0.7,0.,0.]]], dtype=torch.float64)
    model = SimpleNamespace(body_parentid=np.array([0,0]),
      body_quat=np.array([[1.,0.,0.,0.],[1.,0.,0.,0.]]), body_jntadr=np.array([-1,0]),
      jnt_axis=np.array([[1.,0.,0.]]), jnt_qposadr=np.array([0]),
      qpos0=np.array([0.]), jnt_range=np.array([[-3.,3.]]))
    extracted = AnalyticIK(model, torch.tensor([0,1]), torch.tensor([0]),
                           torch.tensor([0,1]), 'cpu', torch.float64)
    q, v = ik(rotations, velocities)
    qe, ve = extracted(rotations, velocities)
    torch.testing.assert_close(qe, q)
    torch.testing.assert_close(ve, v)
    torch.testing.assert_close(q, angle.reshape(1,1))
    torch.testing.assert_close(v, torch.tensor([[0.7]],dtype=torch.float64))
    q.sum().backward()
    self.assertTrue(torch.isfinite(angle.grad))
    self.assertGreater(angle.grad.abs().item(), 0)
    with self.assertRaises(ValueError):
      AnalyticIK.from_constants(parent=ik.parent, child=ik.child, A=ik.A, B=ik.B,
        sign=ik.sign, qpos0=ik.qpos0, lo=ik.lo, hi=ik.hi, perm=torch.tensor([1]))

  def test_arbitrary_hinges_fixed_transforms_and_joint_order(self):
    def quaternion(axis, angle):
      axis = torch.as_tensor(axis, dtype=torch.float64)
      axis = axis / axis.norm()
      angle = torch.as_tensor(angle, dtype=torch.float64)
      return torch.cat([(angle / 2).cos().reshape(1), axis * (angle / 2).sin()])

    axes = torch.tensor([[1., 2., -3.], [-2., 1., 4.]], dtype=torch.float64)
    axes = axes / axes.norm(dim=-1, keepdim=True)
    fixed = torch.stack([quaternion([1., 2., 1.], .7), quaternion([2., -1., 3.], -.5)])
    q0 = torch.tensor([.2, -.1], dtype=torch.float64)
    model = SimpleNamespace(body_parentid=np.array([0, 0, 1]),
      body_quat=torch.cat([torch.tensor([[1.,0.,0.,0.]]), fixed]).numpy(),
      body_jntadr=np.array([-1,0,1]), jnt_axis=axes.numpy(),
      jnt_qposadr=np.array([0,1]), qpos0=q0.numpy(), jnt_range=np.array([[-2.,2.],[-2.,2.]]))
    extracted = AnalyticIK(model, torch.arange(3), torch.tensor([1,0]),
                           torch.arange(3), 'cpu', torch.float64)
    constants = {name: getattr(extracted, name) for name in
                 ('parent','child','A','B','sign','qpos0','lo','hi','perm')}
    ik = AnalyticIK.from_constants(**constants)
    theta = torch.tensor([.3, -.4], dtype=torch.float64, requires_grad=True)
    rates = torch.tensor([.7, -.2], dtype=torch.float64, requires_grad=True)
    root = matrix_from_quat(quaternion([3., -2., 1.], 1.1))
    rots = [root]
    avels = [torch.tensor([.1, -.3, .2], dtype=torch.float64)]
    for i in range(2):
      before_hinge = rots[-1] @ matrix_from_quat(fixed[i])
      rots.append(before_hinge @ matrix_from_quat(quaternion(axes[i], theta[i])))
      avels.append(avels[-1] + (before_hinge @ axes[i]) * rates[i])
    rots, avels = torch.stack(rots)[None], torch.stack(avels)[None]
    q, v = ik(rots, avels)
    torch.testing.assert_close(q, (theta + q0).flip(0)[None])
    torch.testing.assert_close(v, rates.flip(0)[None])
    for actual, expected in zip(extracted(rots, avels), (q, v), strict=True):
      torch.testing.assert_close(actual, expected)
    # A shared-frame rotation must not change reconstructed joints/rates.
    frame = matrix_from_quat(quaternion([-1., 4., 2.], -.8))
    q2, v2 = ik(frame @ rots, (frame @ avels[..., None])[..., 0])
    torch.testing.assert_close(q2, q)
    torch.testing.assert_close(v2, v)
    dq, = torch.autograd.grad(q.sum(), theta, retain_graph=True)
    dv, = torch.autograd.grad(v.sum(), rates)
    # The original _atan2 deliberately damps the derivative by 1/(1+damp).
    torch.testing.assert_close(dq, torch.full_like(theta, 1/1.01))
    torch.testing.assert_close(dv, torch.ones_like(rates))
    rep = create_representation('maxcoord', 39, 2, .02, 4, q0.flip(0), 'proprio')
    limits = torch.tensor([[-2.,2.],[-2.,2.]], dtype=torch.float64)
    rep.bind_native(limits, ik)
    with self.assertRaisesRegex(ValueError, 'body bounds'):
      ik.validate_binding(2, 2, torch.device('cpu'), torch.float64)
    with self.assertRaisesRegex(ValueError, 'joint count'):
      ik.validate_binding(3, 3, torch.device('cpu'), torch.float64)
    with self.assertRaisesRegex(ValueError, 'dtype'):
      ik.validate_binding(3, 2, torch.device('cpu'), torch.float32)
    with self.assertRaisesRegex(ValueError, 'joint count'):
      rep.bind_native(limits, lambda r,w: (w[...,0,:], w[...,0,:]))
    bad_cases = [
      {'parent': torch.tensor([-1,1])},
      {'parent': torch.tensor([2,1])},
      {'child': torch.tensor([1,1])},
      {'sign': torch.tensor([[1.,0.,1.],[1.,0.,0.]], dtype=torch.float64)},
      {'sign': torch.tensor([[-1.,0.,0.],[1.,0.,0.]], dtype=torch.float64)},
      {'qpos0': torch.ones(2,3, dtype=torch.float64)},
      {'A': torch.zeros_like(ik.A)},
      {'B': -ik.B},
      {'A': ik.A.float()},
      {'lo': ik.hi + 1},
      {'perm': torch.tensor([0,0])},
    ]
    for bad in bad_cases:
      with self.subTest(invalid=list(bad)), self.assertRaises(ValueError):
        AnalyticIK.from_constants(**(constants | bad))

  def test_actual_wm_and_bptt_updates_on_both_representations(self):
    torch.set_num_threads(2)
    for name, width in [('gencoord', 59), ('maxcoord', 182)]:
      with self.subTest(representation=name):
        torch.manual_seed(5)
        rep = create_representation(name, width, 23, .02, 4, torch.zeros(23), 'rep_state')
        states = torch.randn(2, 3, width) * .1
        if name == 'gencoord':
          states[..., 3:7] = torch.tensor([1.,0.,0.,0.])
        else:
          states[..., 42:98] = torch.tensor([1.,0.,0.,0.]).repeat(14)
        dyn = Dynamics(rep, 23, 1, torch.ones(23), torch.zeros(23), hidden_dims=(16,16,16))
        learner = WorldModelLearner(dyn, rep, WorldModelLearnerCfg(num_batches=1, batch_size=2, horizon=3))
        before = snapshot(dyn)
        wm_muon = [(p, p.detach().clone()) for opt in learner.optimizers
                   if isinstance(opt, torch.optim.Muon) for group in opt.param_groups
                   for p in group['params'] if p.ndim == 2]
        self.assertTrue(wm_muon, 'WM must have Muon-managed hidden matrices')
        batch = TensorDict({'state': states, 'action': torch.randn(2,3,23),
                            'sim_ctx':torch.zeros(2,3,1)}, batch_size=[2,3])
        stats = learner.update([batch])
        self.assertTrue(all(torch.isfinite(v) for v in stats.values()))
        self.assertTrue(changed(before, dyn))
        self.assertTrue(any(not torch.equal(p, old) for p, old in wm_muon),
                        'A Muon-managed WM hidden matrix must actually update')
        actor = Actor(rep.rep_state_dim)
        policy = BpttPolicyLearner(actor, TensorRollout(dyn, rep, states[:,0]),
                                  BpttPolicyLearnerCfg(batch_size=2, horizon=2))
        before_dyn, before_actor = snapshot(dyn), snapshot(actor)
        actor_muon = [(p, p.detach().clone()) for opt in policy.actor_optimizers
                      if isinstance(opt, torch.optim.Muon) for group in opt.param_groups
                      for p in group['params'] if p.ndim == 2]
        self.assertTrue(actor_muon, 'Actor must have Muon-managed hidden matrices')
        stats = policy.update()
        self.assertTrue(all(torch.isfinite(v) for v in stats.values()))
        self.assertTrue(changed(before_actor, actor))
        self.assertTrue(any(not torch.equal(p, old) for p, old in actor_muon),
                        'A Muon-managed actor hidden matrix must actually update')
        self.assertFalse(changed(before_dyn, dyn))
        self.assertTrue(all(not p.requires_grad for p in dyn.parameters()))
        self.assertTrue(any(isinstance(o, torch.optim.Muon) for o in learner.optimizers))
        self.assertTrue(any(isinstance(o, torch.optim.Muon) for o in policy.actor_optimizers))


if __name__ == '__main__':
  unittest.main()
