"""Regression for #24: update_job() must not mutate the caller's updates dict.

Programmatic callers (retry loops, batch edits in hermes_cli/cron.py) reuse
their updates dict across calls; normalization inside update_job used to
rewrite workdir/monitor/reasoning_effort values in place.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def cron_home(tmp_path, monkeypatch):
    home = tmp_path / "cron-home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _seed_job(cron_home):
    from cron import jobs as jobs_mod

    return jobs_mod.create_job(prompt="ping", schedule="every 10m", name="mut-regression")


def test_update_job_does_not_mutate_caller_dict(cron_home, tmp_path):
    from cron import jobs as jobs_mod

    job = _seed_job(cron_home)
    real_dir = tmp_path / "wd"
    real_dir.mkdir()
    updates = {
        "workdir": str(real_dir),
        "monitor_script": " run.sh ",
        "reasoning_effort": "high",
    }
    snapshot = json.dumps(updates, sort_keys=True)

    jobs_mod.update_job(job["id"], updates)

    assert json.dumps(updates, sort_keys=True) == snapshot, (
        "update_job rewrote the caller's updates dict"
    )


def test_update_job_none_workdir_does_not_mutate_caller_dict(cron_home):
    from cron import jobs as jobs_mod

    job = _seed_job(cron_home)
    updates = {"workdir": ""}
    jobs_mod.update_job(job["id"], updates)
    assert updates == {"workdir": ""}, "caller's workdir='' was rewritten to None"
