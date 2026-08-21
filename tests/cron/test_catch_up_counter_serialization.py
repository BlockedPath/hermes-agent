"""Regression for #23: record_catch_up_occurrence's read-modify-write must be
serialized by _jobs_lock() so concurrent processes cannot lose increments.

Two behavioral proofs:

1. test_record_blocks_while_jobs_lock_held — deterministic: before the fix the
   worker completed while another thread held the jobs lock (no serialization);
   after the fix it blocks until release.
2. test_concurrent_increments_do_not_lose_updates — 8 racing threads must all
   land (the in-process half of the race).

Store scoping uses HERMES_HOME (process-global) rather than use_cron_store()
because the latter is a ContextVar and does not propagate to worker threads.
"""
from __future__ import annotations

import threading

from cron import jobs


def test_record_blocks_while_jobs_lock_held(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    done = threading.Event()

    def worker():
        jobs.record_catch_up_occurrence()
        done.set()

    with jobs._jobs_lock():
        t = threading.Thread(target=worker)
        t.start()
        # Pre-fix the worker ignored the lock entirely and completed here.
        assert not done.wait(0.5), (
            "record_catch_up_occurrence ran while the jobs lock was held "
            "— read-modify-write is not serialized (#23)"
        )
    t.join(5)
    assert done.is_set()
    assert jobs.get_catch_up_occurrence_count() == 1


def test_concurrent_increments_do_not_lose_updates(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    jobs.record_catch_up_occurrence()  # seed = 1
    threads = [
        threading.Thread(target=jobs.record_catch_up_occurrence)
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert jobs.get_catch_up_occurrence_count() == 9
