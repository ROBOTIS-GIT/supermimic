# Cyclo shared-core patch contract

This branch descends from official [HybridRobotics/SuperMimic](https://github.com/HybridRobotics/supermimic)
commit `abc124c1bd5976341034668af3d12c087f2f1f55`. The original Apache-2.0 LICENSE
is unchanged. README setup commands explicitly select the optional research
runtime; its introduction, project links and acknowledgements are retained.
Native robot assets, configuration, collection, motion
adapters and checkpoint manifests belong to the consumer repositories.
This modification/provenance record ships with the license notices in both
source distributions and wheel metadata; it does not change the license terms.

## Portable core and separate runtimes

The portable package supports Python 3.12 and 3.13. It declares Torch >=2.11,
TensorDict >=0.14.2 and RSL-RL >=5.4.2,<6. These dependencies provide the actual
Muon/Adam optimizers and empirical normalization used by the original learners.
The original PEP 695 type alias and generic class syntax are retained. Python
3.11 is outside this branch's supported contract.

Installing the base package does not require a simulator:

```bash
pip install .
```

Consumers with a pinned native stack should resolve against that stack; do not
replace its dependencies with the standalone research lock. Shared source does
not imply a shared compatible simulator environment. In particular, the Lab
Python 3.12 stack and stable MJLab 1.6.1 use different RSL-RL/MuJoCo requirements.

The optional `mjlab` extra pins the official research fork at
`dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd`. The `research` dependency group selects
that extra on Python 3.13 only when explicitly requested. Default `uv sync`
selects only the portable core and development tools. For the original research
workflow use `uv run --group research --python 3.13
src/supermimic/scripts/train.py ...` (or `play.py`). Use
`uv sync --no-default-groups --python 3.12` for a core-only project;
native consumers normally own their environment/lock instead. Installing the
extra through pip also resolves the research Git revision, not a same-version
PyPI mjlab release. The standalone lock is not a consumer migration plan.

The portable import surface is `models`, `representations`, `replay_buffer`,
`sim_context`, `tasks.task_spec`, `learners.*` and `envs.*`. Importing these does
not load mjlab, MuJoCo or Isaac. Research scripts, environment configurations,
viewers, exporter and runner orchestration remain optional backend-facing code.
Official scripts explicitly call `supermimic.tasks.register_tasks()`; importing
`supermimic.tasks` alone no longer registers anything.

## Native binding API

`create_representation(name, full_state_dim, num_actions, dt, decimation,
default_joint_pos, actor_obs_mode)` is unchanged. `dt` and `decimation` must match
the native simulator exactly; weights do not restore these ordinary attributes.
Actor mode remains explicit: privileged `rep_state` or sensor-input `proprio`.

After constructing the representation, a native adapter can call:

```python
gen.bind_native(soft_joint_pos_limits, fk)
maxcoord.bind_native(soft_joint_pos_limits, analytic_ik, encoder_bias=None)
```

- Joint limits have shape `(num_joints, 2)` in native action-joint order.
- FK takes `(root_pos_w, root_quat_w, joint_pos, root_lin_vel_w,
  root_ang_vel_w, joint_vel)` and returns ordered body positions, wxyz
  quaternions, link-origin linear velocities and angular velocities in world
  coordinates. It must preserve gradients needed by the imagined reward.
- Representation-bound IK takes tracked-body rotation matrices `(..., B, 3, 3)`
  and angular velocities `(..., B, 3)` in the root-local frame, returning joint
  positions and rates in native action
  order. Standalone IK accepts any consistently shared frame. A callable/subclass
  is sufficient; no simulator class is required.
- Optional encoder bias is `(slice, SimContext)` using the existing context API.
- `AnalyticIK.from_constants(parent=..., child=..., A=..., B=..., sign=...,
  qpos0=..., lo=..., hi=..., perm=...)` exposes the original tensor solver to
  native topology extraction. See its docstring for shapes, padding and ordering.
  Topology extraction and robot names/assets stay in the consumer. The existing
  MuJoCo-model constructor remains supported and now accepts one-hinge chains.

Bindings retain ordinary tensor/callable references. Bind on the final device
and dtype before graph capture, and rebind after relocation. Native binding
checks joint limits, IK output shapes/device/dtype, and constants-based IK body
bounds/joint count. Constants validate signs, zero padding, ordered limits,
proper rotation bases and acyclic body indices before solving.

MaxCoord still integrates the original independent body state; no FK projection
has been added. GenCoord and MaxCoord representation/reward/integration methods
are otherwise unchanged.

The original `WorldModelEnv` and subclasses accept a duck-typed native facade.
It supplies `num_actions`, `cfg.rewards` (terms with `weight`),
`cfg.terminations['anchor_pos'/'anchor_ori'].params['threshold']`, and `unwrapped`
with `step_dt` and `command_manager.get_term('motion')`. The motion command
provides `motion` and `motion_anchor_body_index`. The motion provider supplies
`joint_pos`, `joint_vel`, `root_pos_w`, `root_quat_w`, `motion_ends` and
`body_state(frame_indices)`. Reward scaling, replay/history boundaries, goals,
termination and imagined rollout equations stay in the official classes.

For native inference metadata, `SuperMimicRunner.checkpoint_extra_files(log_dir)`
is an explicit override returning additional artifact paths. The default returns
an empty list. Consumers write/version their own metadata; the core knows no K1
manifest schema. The runner saves final trained weights on normal loop completion
and awaits the final artifact future. Offline/disabled/dryrun artifact logging
skips remote API enumeration/pruning. This is not a full optimizer/replay resume
checkpoint or a transactional checkpoint set. Final-upload failures run the
cleanup hook, but exceptions earlier in the inherited training/save loop are
not covered by that final cleanup block.

## Source modifications and notices

| Files | Change |
| --- | --- |
| `README.md` | Fork clone URL, explicit opt-in research setup/commands, portable-core and BSD notice links |
| `pyproject.toml`, `uv.lock` | Portable dependencies, optional pinned research runtime, Python 3.12–3.13 support, bundled license metadata |
| `utils/math.py` | Twelve tensor functions extracted unchanged from the pinned official research dependency's `src/mjlab/utils/lab_api/math.py`; BSD-3-Clause notice retained in the file and `licenses/IsaacLab-BSD-3-Clause.txt` |
| `representations.py`, `tasks/task_spec.py`, `utils/{helpers,gencoord_helpers,maxcoord_helpers,tracking_reward}.py` | Replace eager mjlab math imports with the local licensed tensor functions; add explicit native representation bindings |
| `utils/analytic_ik.py` | MuJoCo becomes a type-checking-only import; generic one-hinge reduction and validated constants constructor; original solver methods unchanged |
| `tasks/__init__.py`, `scripts/{train,train_ppo,play,benchmark}.py` | Explicit optional research-task registration |
| `runners/supermimic_runner.py` | Final-iteration checkpoint, final future completion, native extra-file hook |
| `utils/wandb_artifacts.py` | No remote enumeration/pruning in offline/disabled/dryrun mode |
| Tests | Pure-core fixtures/imports, optional backend test, numerical/learner/normalizer/IK acceptance, final checkpoint/offline regressions |

The original model, replay buffer, policy/WM learners, imagined environments and
simulation-context files are byte-identical to the official commit. Original
representation mathematical methods and IK solver methods are AST-identical;
only binding/topology setup changes. The extracted math functions are AST-identical
to the pinned research source. No unofficial reproduction was used.

## Verification

The standalone CPU check needs no pytest:

```bash
PYTHONPATH=src TORCHDYNAMO_DISABLE=1 CUDA_VISIBLE_DEVICES='' \
  python tests/test_tensor_core.py
```

It blocks simulator imports, checks normalizer `_std` buffer address/value
behavior, verifies single-hinge IK/gradients with arbitrary axes, fixed transforms,
shared-frame changes and reversed native joint ordering, and executes actual supervised
multi-step WM and BPTT optimizer updates for widths 59/182 with 23 actions.
Both learners update parameters, including Muon hidden matrices; WM parameters
remain unchanged/frozen during the actor update. These are synthetic tensor
fixtures, not simulator collection, convergence or robot-deployment evidence.

The full research suite is `PYTHONPATH=src TORCHDYNAMO_DISABLE=1 python -m pytest
-p no:cacheprovider tests -q`. In a core-only environment, omit
`test_analytic_ik.py`, `test_buffer_schema.py`, `test_runner_history.py`,
`test_run_config.py` and `test_runtime_fixes.py`, which require research/backend
or orchestration dependencies. The remaining environment-noise configuration
check skips explicitly when mjlab is absent. Consumer simulator regressions and
submodule migrations must be validated separately against the selected commit.

The candidate was exercised with these existing dependency sets on CPU:

| Environment | Torch / TensorDict / RSL-RL | Checks |
| --- | --- | --- |
| Lab Python 3.12.14, no mjlab | 2.11.0 / 0.14.2 / 5.5.1 | 205 core tests passed, one optional backend test skipped |
| Stable MJLab 1.6.1, Python 3.12.16 | 2.14.0 / 0.15.6 / 5.4.2 | 232 full tests passed |
| Official research mjlab 1.6.0, Python 3.13.18 | 2.15.0 / 0.15.6 / 5.5.2 | 232 full tests passed |

These include actual CPU eager WM/BPTT updates and normalizer buffer checks;
the full suites also launch both official G1 train/play entrypoints with
`--help`, exercising explicit registration without simulation or W&B access.
Stable MJLab passes these orchestration imports and unit checks, so the shared
core does not require a second MJLab runtime. Native collector/context/viewer
and K1 checkpoint/NPZ integration remain consumer acceptance gates. These
results do not establish native proprio/history rollout or tracking convergence.
