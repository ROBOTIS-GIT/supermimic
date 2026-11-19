import copy
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("script", ["train", "play"])
@pytest.mark.parametrize("representation", ["GenCoord", "MaxCoord"])
def test_research_cli_explicitly_registers_official_tasks(script, representation):
    result = subprocess.run(
        [sys.executable, "-m", f"supermimic.scripts.{script}",
         f"SuperMimic-{representation}-G1", "--help"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "WANDB_MODE": "disabled"},
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()


@pytest.mark.parametrize("mode", ["offline", "disabled", "dryrun"])
def test_offline_artifacts_never_call_remote_api(monkeypatch, mode):
    from supermimic.utils import wandb_artifacts as artifacts

    logged = []
    monkeypatch.setattr(
        artifacts.wandb,
        "run",
        SimpleNamespace(
            id="local", settings=SimpleNamespace(mode=mode), log_artifact=logged.append
        ),
    )
    monkeypatch.setattr(
        artifacts.wandb,
        "Artifact",
        lambda *a, **kw: SimpleNamespace(add_file=lambda _: None),
    )
    monkeypatch.setattr(
        artifacts.wandb, "Api", lambda: pytest.fail("remote API called")
    )
    artifacts.log_model_artifact(["actor.pt"])
    assert len(logged) == 1


@pytest.mark.parametrize(
    "iterations,interval,upload_failure",
    [(16, 1000, False), (3, 2, False), (16, 1000, True)],
)
def test_real_runner_loop_saves_final_parameters(
    monkeypatch, tmp_path, iterations, interval, upload_failure
):
    from supermimic.runners import supermimic_runner as module

    actor = torch.nn.Linear(1, 1, bias=False)
    critic = torch.nn.Linear(1, 1, bias=False)
    dynamics = torch.nn.Linear(1, 1, bias=False)
    for model in (actor, critic, dynamics):
        model.weight.data.zero_()
    count = {"wm": 0, "policy": 0}

    def update_wm(_):
        dynamics.weight.data.add_(1)
        count["wm"] += 1
        return {}

    def update_policy(starts):
        assert len(list(starts)) == 1
        actor.weight.data.add_(1)
        critic.weight.data.add_(1)
        count["policy"] += 1
        return {}

    buffer = SimpleNamespace(num_transitions=0, sample=lambda horizon, batch: torch.zeros(batch, horizon))

    def step():
        buffer.num_transitions += 1
        return 1

    cfg = SimpleNamespace(
        max_iterations=iterations,
        save_interval=interval,
        log_interval=100,
        wm=SimpleNamespace(num_batches=1, batch_size=1, horizon=8),
        policy=SimpleNamespace(num_batches=1, batch_size=1),
        history_length=1,
        replay_ratio=8,
    )
    cleanup = []
    runner = SimpleNamespace(
        cfg=cfg,
        checkpoint_extra_files=lambda _: [],
        env=SimpleNamespace(num_envs=1, reset=lambda: None),
        _create_log_dir=lambda: tmp_path,
        _init_wandb=lambda _: None,
        _seed_sim_history=lambda: None,
        _cleanup=lambda: cleanup.append(True),
        step=step,
        replay_buffer=buffer,
        world_model_learner=SimpleNamespace(update=update_wm),
        policy_learner=SimpleNamespace(
            update=update_policy, actor=actor, critic=critic
        ),
        _wm=SimpleNamespace(bundle=lambda: copy.deepcopy(dynamics.state_dict())),
        _representation=SimpleNamespace(actor_obs_mode="rep_state"),
        logger=SimpleNamespace(log=lambda **kw: None),
    )
    logged = []
    monkeypatch.setattr(module.wandb, "run", SimpleNamespace(name="local"))

    def log(files):
        logged.append(torch.load(files[0], weights_only=True)["weight"].item())
        if upload_failure and len(logged) == 2:
            raise RuntimeError("final artifact failed")

    monkeypatch.setattr(module, "log_model_artifact", log)
    if upload_failure:
        with pytest.raises(RuntimeError, match="final artifact failed"):
            module.SuperMimicRunner.run(runner)
    else:
        module.SuperMimicRunner.run(runner)
    assert cleanup == [True]
    for name, model in [
        ("world_model", dynamics),
        ("actor", actor),
        ("critic", critic),
    ]:
        saved = torch.load(tmp_path / (name + ".pt"), weights_only=True)
        assert torch.equal(saved["weight"], model.weight)
    assert logged[-1] == count["wm"]
    if iterations == 16:
        assert count == {"wm": 9, "policy": 9}
        assert logged == [0, 9]
    else:
        assert len(logged) == 2  # final iteration also lies on an interval
