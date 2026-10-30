"""Unit tests for af3lis.pack / af3lis.calibrate / ipsae_runner parsing.

Run:  python -m pytest tests/test_pack.py -q      (or plain `python tests/test_pack.py`)
No cluster, no AF3 outputs needed -- everything is synthetic.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from af3lis.calibrate import calibrate
from af3lis.ipsae_runner import parse_ipsae_txt
from af3lis.pack import (DEFAULT_COMPILE, DEFAULT_STARTUP, bucket_of,
                         build_groups, token_count)


def test_bucket_of():
    assert bucket_of(100) == 256
    assert bucket_of(512) == 512
    assert bucket_of(513) == 768
    assert bucket_of(2049) == 2560
    assert bucket_of(3100) == 3584
    # the complex that hit its walltime: AF3 chose 3584, we used to say 4096
    assert bucket_of(3275) == 3584
    assert bucket_of(4500) == 4608
    assert bucket_of(5120) == 5120
    # beyond the ladder AF3 makes a bucket of exactly the token count
    assert bucket_of(5121) == 5121


def test_buckets_match_af3_v301_default():
    """pack.BUCKETS must be AF3's own ladder, or groups are sized for a
    bucket AF3 does not use. Source: the --buckets flag default of
    run_alphafold.py at tag v3.0.1."""
    from af3lis.pack import BUCKETS
    af3_v301 = [256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072,
                3584, 4096, 4608, 5120]
    assert BUCKETS == af3_v301, BUCKETS


def test_default_table_covers_every_bucket_and_is_monotonic():
    from af3lis.pack import BUCKETS, DEFAULT_CALIB, default_sec
    assert sorted(DEFAULT_CALIB) == BUCKETS, "a bucket has no built-in time"
    prev = 0.0
    for b in BUCKETS:
        assert DEFAULT_CALIB[b] >= prev, (b, DEFAULT_CALIB[b], prev)
        prev = DEFAULT_CALIB[b]
    # above the ladder: extrapolated upwards, never the old flat 700 s
    assert default_sec(6000) > DEFAULT_CALIB[5120]
    assert default_sec(8192) > default_sec(6000)


def test_no_size_class_gets_less_time_than_before_the_ladder_changed():
    """Adding AF3's 2560/3584/4608 buckets must never LOWER the walltime ask
    for an input: the old table is what the last campaign ran and finished
    with, and an unmeasured interpolation is not a reason to cut it. (2560
    at a ^2.2 interpolation of ~1150 s would have cut 2049-2560-token inputs
    from 1600 s to 1150 s on no measurement.)"""
    import math
    from af3lis.pack import default_sec
    old_buckets = [256, 512, 768, 1024, 1280, 1536, 2048, 3072, 4096, 5120]
    old_table = {256: 15, 512: 35, 768: 85, 1024: 160, 1280: 260, 1536: 380,
                 2048: 700, 3072: 1600, 4096: 3000, 5120: 4800}

    def old_sec(tokens):
        for b in old_buckets:
            if tokens <= b:
                return old_table[b]
        return old_table.get(int(math.ceil(tokens / 1024.0) * 1024), 700.0)

    for tokens in range(1, 9000, 7):
        new = default_sec(bucket_of(tokens))
        assert new >= old_sec(tokens), (tokens, bucket_of(tokens), new,
                                        old_sec(tokens))


def test_measured_3584_condition_fits_its_walltime():
    """One 3,275-token condition needed 1:28:02 of job time on the
    large-model tier and was killed at the 1:10 it was sized for. With the
    default startup/compile/safety it must now get at least that."""
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "data_x.json")
        open(d, "w").write("{}")
        recs = [dict(lname="x", data=d, bucket=bucket_of(3275), nseeds=1)]
        from af3lis.pack import DEFAULT_CALIB
        calib = {int(k): float(v) for k, v in DEFAULT_CALIB.items()}
        g, = build_groups(recs, os.path.join(td, "pack"), target_hours=1.5,
                          safety=1.3, calib=calib, startup=DEFAULT_STARTUP,
                          compile_s=DEFAULT_COMPILE)
        h, m, sec = (int(x) for x in g["walltime"].split(":"))
        measured = 1 * 3600 + 28 * 60 + 2
        assert h * 3600 + m * 60 + sec >= measured, g["walltime"]
        assert g["bucket"] == 3584


def test_beyond_ladder_groups_are_sized_upwards():
    """A 6000-token input must not be sized below a 5120 one (it was: an
    untabled bucket fell back to a flat 700 s)."""
    from af3lis.pack import DEFAULT_CALIB
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "data_big.json")
        open(d, "w").write("{}")
        calib = {int(k): float(v) for k, v in DEFAULT_CALIB.items()}
        g, = build_groups([dict(lname="big", data=d, bucket=bucket_of(6000),
                                nseeds=1)],
                          os.path.join(td, "pack"), 1.5, 1.0, calib, 0.0, 0.0)
        assert g["bucket"] == 6000
        assert g["est_s"] > DEFAULT_CALIB[5120], g["est_s"]


def test_token_count_multiplicity():
    j = {"sequences": [
        {"protein": {"id": ["A", "B"], "sequence": "M" * 100}},
        {"protein": {"id": "C", "sequence": "M" * 50}},
        {"ligand": {"id": ["X", "Y"], "ccdCodes": ["ATP"]}},
    ]}
    assert token_count(j) == 100 * 2 + 50 + 2


def test_build_groups_single_bucket_and_walltime():
    with tempfile.TemporaryDirectory() as td:
        # 4 small (bucket 512) + 1 big (bucket 1024) conditions
        recs = []
        for i in range(4):
            d = os.path.join(td, f"data_s{i}.json")
            open(d, "w").write("{}")
            recs.append(dict(lname=f"s{i}", data=d, bucket=512, nseeds=1))
        d = os.path.join(td, "data_big.json")
        open(d, "w").write("{}")
        recs.append(dict(lname="big", data=d, bucket=1024, nseeds=1))

        calib = {512: 600.0, 1024: 3000.0}  # 10 min / 50 min per condition
        groups = build_groups(recs, os.path.join(td, "pack"),
                              target_hours=0.5, safety=1.0,
                              calib=calib, startup=0.0, compile_s=0.0)
        # bucket 512: 1800s target / 600s -> 3 per group -> groups of 3 + 1
        # bucket 1024: 3000s > target -> forced 1 per group
        sizes = sorted((g["bucket"], len(g["members"])) for g in groups)
        assert sizes == [(512, 1), (512, 3), (1024, 1)]
        # every group is single-bucket by construction
        for g in groups:
            assert len({r for r in g["members"]}) == len(g["members"])
            # members are symlinked into the group dir
            for m in g["members"]:
                assert os.path.islink(os.path.join(g["dir"], f"data_{m}.json")) or \
                       os.path.islink(os.path.join(g["dir"], os.listdir(g["dir"])[0]))
        # walltime string format
        assert all(len(g["walltime"]) == 8 for g in groups)


def _hms(t: float) -> str:
    t %= 86400.0
    return "%02d:%02d:%09.6f" % (t // 3600, (t % 3600) // 60, t % 60)


def _absl_log(n_cond, nseeds, per_seed, bucket=512, tokens0=480, t0=36000.0,
              startup=30.0, compile_extra=60.0, feat_gap=4.0,
              seed_print=True, durations=None):
    """Synthetic packed-job log shaped like AF3 3.0.1's: every seed of a
    condition is featurised up front (one bucket-line pair per seed, a few
    seconds apart), then inference runs; the first condition also pays the
    XLA compile."""
    out = [f"I1020 {_hms(t0)} 1 run_alphafold.py] job start\n"]
    t = t0 + startup
    seeds = ", ".join(str(i + 1) for i in range(nseeds))
    for i in range(n_cond):
        tok = tokens0 + i                       # neighbours differ in size
        if seed_print:
            out.append(f"Featurising data for seeds ({seeds}{',' if nseeds == 1 else ''})...\n")
        for k in range(nseeds):
            ts = t + k * feat_gap
            out.append(f"I1020 {_hms(ts)} 1 pipeline.py:257] Calculating bucket "
                       f"size for input with {tok} tokens.\n")
            out.append(f"I1020 {_hms(ts + 0.3)} 1 pipeline.py:263] Got bucket "
                       f"size {bucket} for input with {tok} tokens, resulting in "
                       f"{bucket - tok} padded tokens.\n")
        dur = (durations[i] if durations else per_seed) * nseeds
        t += dur + (compile_extra if i == 0 else 0.0)
    return "".join(out)


def _write(text, suffix=".out"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    return path


def test_calibrate_single_seed_conditions():
    """One seed: each condition's gap is the whole cycle. The first
    condition carries the compile and the last has no end, so 6 conditions
    give 4 measurements."""
    p = _write(_absl_log(6, 1, per_seed=90.0))
    out = calibrate(p)
    assert abs(out["512"] - 90.0) < 0.2, out
    assert out["_meta"]["n_per_bucket"]["512"] == 4
    assert out["_meta"]["excluded_compile"] == 1
    assert abs(out["startup"] - 30.0) < 0.01
    assert out["compile"] == 40.0


def test_calibrate_multiseed_is_not_featurisation_time():
    """Regression for the ~5x-too-fast calibration. With several seeds AF3
    logs one bucket line per seed back to back (featurisation), so most gaps
    between successive bucket lines are seconds long. Taking the median of
    those -- the old method -- returns featurisation time, not condition
    time. The per-seed condition time here is 100 s."""
    for seed_print in (True, False):            # print present / GCD fallback
        p = _write(_absl_log(7, 5, per_seed=100.0, bucket=1024, tokens0=900,
                             seed_print=seed_print))
        out = calibrate(p)
        assert abs(out["1024"] - 100.0) < 0.5, (seed_print, out)
        assert out["_meta"]["seeds"][os.path.basename(p)].startswith("5 ")
        # what the old gap-between-any-two-lines median would have said
        from af3lis.calibrate import parse_log
        import statistics
        ts = [x[0] for x in parse_log(p)["lines"]]
        old = statistics.median(b - a for a, b in zip(ts, ts[1:]))
        assert old < 10.0, old                  # featurisation, ~25x too fast


def test_calibrate_logs_are_independent_and_small_buckets_dropped():
    """Each log is parsed on its own (no gap spans two jobs, even when the
    second job's clock reads EARLIER), and a bucket measured on fewer than
    --min-per-bucket conditions is dropped rather than trusted."""
    a = _write(_absl_log(4, 1, per_seed=150.0, bucket=1024, t0=80000.0))
    b = _write(_absl_log(5, 1, per_seed=150.0, bucket=1024, t0=3600.0))
    bridge = _write("[pack] 12 conditions -> 2 packed jobs\nno timestamps\n")
    alone = calibrate(a)
    assert "1024" not in alone, alone           # 2 measured < 3
    assert "1024" in alone["_meta"]["dropped_buckets"]
    both = calibrate([a, b, bridge])
    assert abs(both["1024"] - 150.0) < 0.2, both
    assert both["_meta"]["n_per_bucket"]["1024"] == 5
    assert both["_meta"]["logs_without_bucket_lines"] == 1


def test_calibrate_drops_implausibly_fast_conditions():
    """A condition that 'finishes' in seconds was skipped or failed; it must
    not pull the median down."""
    durs = [90.0, 90.0, 2.0, 90.0, 90.0, 90.0, 90.0]
    p = _write(_absl_log(7, 1, per_seed=90.0, durations=durs))
    out = calibrate(p)
    assert out["_meta"]["excluded_fast"] == 1
    assert abs(out["512"] - 90.0) < 0.2, out


def test_check_calib_refuses_the_too_fast_calibration():
    from af3lis.pack import check_calib
    # the values that nearly sized a real campaign
    bad = check_calib({1024: 17.6, 1280: 78.2})
    assert len(bad) == 2, bad
    # what that campaign's own probe measured: inside the 2x limit
    assert check_calib({1024: 87.0, 1280: 140.0}) == []


def test_calibrate_cli_refuses_unless_forced():
    import json
    from af3lis import calibrate as cal
    from af3lis.pack import load_calib
    log = _write(_absl_log(6, 1, per_seed=20.0, bucket=1024, tokens0=900))
    out = os.path.join(tempfile.mkdtemp(), "calib.json")
    try:
        cal.main([log, "--out", out])
        raise AssertionError("an 8x-too-fast calibration was accepted")
    except SystemExit as e:
        assert e.code == 2, e.code
    assert not os.path.exists(out), "refused calibration was written"
    rej = json.load(open(out + ".rejected"))
    assert rej["_rejected"] and "1024" in rej["_rejected"][0]

    cal.main([log, "--out", out, "--force"])
    c = json.load(open(out))
    assert c["_forced"], c
    # the recorded --force is honoured downstream without a second flag
    calib, _, _, info = load_calib(out)
    assert info["status"] == "forced" and calib[1024] == c["1024"], info


def test_load_calib_never_crashes_and_falls_back_safely():
    import json
    from af3lis.pack import DEFAULT_CALIB, load_calib
    d = tempfile.mkdtemp()
    # missing path: was an unguarded open() -> crash in the bridge
    calib, st, co, info = load_calib(os.path.join(d, "gone.json"))
    assert info["status"] == "missing"
    assert calib == {k: float(v) for k, v in DEFAULT_CALIB.items()}
    # unreadable
    bad = os.path.join(d, "bad.json")
    open(bad, "w").write("{not json")
    assert load_calib(bad)[3]["status"] == "unreadable"
    # refused -> built-in table, unless forced
    fast = os.path.join(d, "fast.json")
    json.dump({"1024": 17.6, "startup": 5, "compile": 1,
               "_meta": {"anything": 1}}, open(fast, "w"))
    calib, st, co, info = load_calib(fast)
    assert info["status"] == "refused" and calib[1024] == DEFAULT_CALIB[1024]
    assert st != 5, "a refused file's startup leaked through"
    calib, st, co, info = load_calib(fast, force=True)
    assert info["status"] == "forced" and calib[1024] == 17.6 and st == 5
    # plausible -> used, with _meta ignored
    good = os.path.join(d, "good.json")
    json.dump({"1024": 120.0, "_meta": {"x": 1}}, open(good, "w"))
    calib, _, _, info = load_calib(good)
    assert info["status"] == "measured" and calib[1024] == 120.0
    assert calib[512] == DEFAULT_CALIB[512]       # untouched buckets kept


def _mini_run_dir():
    """A packable run dir: one ~1000-token pair with its align output."""
    import json
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "jsons"))
    pj = {"name": "bait___preyA", "modelSeeds": [1],
          "sequences": [{"protein": {"id": "A", "sequence": "M" * 600}},
                        {"protein": {"id": "B", "sequence": "K" * 400}}]}
    json.dump(pj, open(os.path.join(d, "jsons", "bait___preyA.json"), "w"))
    od = os.path.join(d, "out", "bait___preya")
    os.makedirs(od)
    json.dump(pj, open(os.path.join(od, "bait___preya_data.json"), "w"))
    open(os.path.join(d, "pack_infer.sbatch"), "w").write("#!/bin/bash\n")
    cfg = os.path.join(d, "empty_config.yaml")
    open(cfg, "w").write("")
    return d, cfg


def _pipeline(*args):
    import subprocess
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run([sys.executable, os.path.join(here, "pipeline.py"),
                           *args], capture_output=True, text=True)


def test_pack_never_autoloads_a_run_dir_calibration():
    """A calib.json in the run dir used to size the campaign without
    --calib. Now it is reported and ignored; an explicit one that is
    missing or implausible falls back to the built-in table without a
    traceback, and calib_used.json records what sized the pack."""
    import json
    from af3lis.pack import DEFAULT_CALIB
    d, cfg = _mini_run_dir()
    json.dump({"1024": 5.0}, open(os.path.join(d, "calib.json"), "w"))
    used = os.path.join(d, "af3pack", "calib_used.json")

    r = _pipeline("--config", cfg, "--pack", "--outdir", d)
    assert r.returncode == 0, r.stderr
    assert "is NOT used" in r.stdout, r.stdout
    u = json.load(open(used))
    assert u["status"] == "default", u
    assert u["sec_per_condition"]["1024"] == DEFAULT_CALIB[1024]

    r = _pipeline("--config", cfg, "--pack", "--outdir", d,
                  "--calib", os.path.join(d, "calib.json"))
    assert r.returncode == 0, r.stderr
    assert "REFUSED" in r.stderr, r.stderr
    assert json.load(open(used))["status"] == "refused"

    r = _pipeline("--config", cfg, "--pack", "--outdir", d,
                  "--calib", os.path.join(d, "gone.json"))
    assert r.returncode == 0 and "Traceback" not in r.stderr, r.stderr
    assert "MISSING" in r.stderr, r.stderr
    assert json.load(open(used))["status"] == "missing"


def _probe_run_dir(n, tokens):
    """A packable run dir with n pairs of ~`tokens` tokens each."""
    import json
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "jsons"))
    for i in range(n):
        nm = f"bait___prey{i}"
        pj = {"name": nm, "modelSeeds": [1],
              "sequences": [{"protein": {"id": "A", "sequence": "M" * 600}},
                            {"protein": {"id": "B",
                                         "sequence": "K" * (tokens - 600 - i)}}]}
        json.dump(pj, open(os.path.join(d, "jsons", nm + ".json"), "w"))
        od = os.path.join(d, "out", nm.lower())
        os.makedirs(od)
        json.dump(pj, open(os.path.join(od, nm.lower() + "_data.json"), "w"))
    open(os.path.join(d, "pack_infer.sbatch"), "w").write("#!/bin/bash\n")
    return d


def test_probe_is_one_job_so_calibrate_can_measure_it():
    """calibrate measures inside one log only (first condition = compile,
    last = no end), so a probe split across jobs measures little or nothing.
    Packed to the default 1.5 h target, a 10-condition probe at bucket 3584
    became ten 1-condition jobs and calibrate stopped with 'no bucket
    measured'. The probe must be ONE job at every bucket, with a walltime
    that covers all of it, and its log must then calibrate."""
    import contextlib, io
    from af3lis import calibrate as cal
    from af3lis.pack import MIN_USEFUL_PROBE, cmd_pack, default_sec
    assert MIN_USEFUL_PROBE == cal.MIN_PER_BUCKET + 2
    for tokens, bucket in ((1000, 1024), (3200, 3584), (4500, 4608)):
        d = _probe_run_dir(10, tokens)
        with contextlib.redirect_stdout(io.StringIO()):
            cmd_pack(d, target_hours=1.5, safety=1.3, probe=10)
        rows = open(os.path.join(d, "af3pack_probe", "groups.tsv")
                    ).read().strip().splitlines()[1:]
        assert len(rows) == 1, (bucket, rows)
        f = rows[0].split("\t")
        assert int(f[1]) == bucket and int(f[2]) == 10, f
        h, m, _ = (int(x) for x in f[4].split(":"))
        need = DEFAULT_STARTUP + DEFAULT_COMPILE + 10 * default_sec(bucket)
        assert h * 3600 + m * 60 >= need * 1.3, (bucket, f[4], need)
        # a log of that one job calibrates the bucket (8 of 10 measured)
        log = _write(_absl_log(10, 1, per_seed=0.8 * default_sec(bucket),
                               bucket=bucket, tokens0=tokens - 20))
        out = cal.calibrate(log)
        assert out["_meta"]["n_per_bucket"][str(bucket)] == 8, out["_meta"]


def test_probe_too_small_to_calibrate_is_flagged():
    import contextlib, io
    from af3lis.pack import cmd_pack
    d = _probe_run_dir(4, 1000)
    err = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(err):
        cmd_pack(d, probe=4)
    assert "at most 2 measurement" in err.getvalue(), err.getvalue()


def test_build_validates_pins_and_hands_calibration_to_the_bridge():
    """--calib at build time used to be accepted and ignored. It is now
    checked while someone is watching, copied into the run dir, and rendered
    into the bridge's --pack line; a bad one stops the build."""
    import json
    d = tempfile.mkdtemp()
    cfg = os.path.join(d, "c.yaml")
    open(cfg, "w").write("")
    good = os.path.join(d, "good.json")
    json.dump({"1024": 120.0}, open(good, "w"))
    run = os.path.join(d, "run")
    base = ["--config", cfg, "--name", "t", "--outdir", run,
            "--chainA", "bait=" + "M" * 30, "--chainB", "preyA=" + "K" * 30,
            "--cache", os.path.join(d, "cache")]
    r = _pipeline(*base, "--calib", good)
    assert r.returncode == 0, r.stderr
    pinned = os.path.join(run, "pack_calib.json")
    assert json.load(open(pinned)) == {"1024": 120.0}

    def pack_line():
        txt = open(os.path.join(run, "pack_bridge.sbatch")).read()
        line, = [l for l in txt.splitlines()
                 if l.startswith('"$PY" pipeline.py --pack')]
        return line
    assert "--calib " in pack_line() and pinned in pack_line(), pack_line()
    # without --calib the bridge carries none
    r = _pipeline(*base, "--rebuild")
    assert r.returncode == 0, r.stderr
    assert "--calib" not in pack_line(), pack_line()
    # refused / missing stop the build
    fast = os.path.join(d, "fast.json")
    json.dump({"1024": 17.6}, open(fast, "w"))
    r = _pipeline(*base, "--rebuild", "--calib", fast)
    assert r.returncode == 2 and "refused" in r.stderr, r.stderr
    r = _pipeline(*base, "--rebuild", "--calib", fast, "--force-calib")
    assert r.returncode == 0, r.stderr
    assert pack_line().endswith("--force-calib"), pack_line()
    r = _pipeline(*base, "--rebuild", "--calib", os.path.join(d, "gone.json"))
    assert r.returncode == 2 and "missing" in r.stderr, r.stderr


def test_parse_ipsae_txt_max_rows_only():
    txt = (
        "\n"
        "Chn1 Chn2  PAE Dist  Type   ipSAE    ipSAE_d0chn ipSAE_d0dom  ipTM_af  "
        "ipTM_d0chn     pDockQ     pDockQ2    LIS       n0res  n0chn  n0dom   "
        "d0res   d0chn   d0dom  nres1   nres2   dist1   dist2  Model\n"
        "A    B     10   10   asym  0.100000    0.700000    0.400000    0.310    "
        "0.250000    0.300000    0.010000  0.1400        70    500    300    "
        "3.00   10.00    8.00     50     60     40     45  m0\n"
        "B    A     10   10   asym  0.050000    0.600000    0.300000    0.310    "
        "0.200000    0.300000    0.010000  0.1400        60    500    300    "
        "2.50   10.00    8.00     40     50     35     40  m0\n"
        "B    A     10   10   max   0.100000    0.700000    0.400000    0.310    "
        "0.250000    0.328000    0.013800  0.1477        70    500    300    "
        "3.00   10.00    8.00     50     60     40     45  m0\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(txt)
        path = fh.name
    try:
        rows = parse_ipsae_txt(path)
        assert len(rows) == 1                      # only the max row
        r = rows[0]
        assert r["ipSAE_d0res"] == 0.1
        assert r["pDockQ"] == 0.328
        assert r["pDockQ2"] == 0.0138
        assert r["LIS_ipsae"] == 0.1477
        assert {r["chain_i"], r["chain_j"]} == {"A", "B"}
    finally:
        os.unlink(path)


def test_resources_for_ladder():
    """Size-aware SLURM resources: the 64gb gpu4 cutoff is the load-bearing rule."""
    from af3lis.pack import resources_for
    # small + mid buckets must stay UNDER 64gb, or they lose the 26 abundant
    # 40GB gpu4 nodes and queue behind 4 scarce gpu8 nodes instead.
    for b in (256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072):
        r = resources_for(b)
        mem_gb = int(r["mem"].replace("gb", ""))
        assert mem_gb < 64, (b, r["mem"])
        assert r["flash_attn"] == "triton"
        # must NOT pin a card type: measured on bwHelix, pinning A100 costs
        # ~2h15m of queue and A40 ~3h23m versus an unpinned request, for no
        # benefit at these sizes.
        assert r["gres"] == "gpu:1", (b, r["gres"])
    # big buckets escalate deliberately (3584 is where AF3's ladder first
    # crosses 3072 tokens)
    for b in (3584, 4096, 4608, 5120, 8192):
        r = resources_for(b)
        assert int(r["mem"].replace("gb", "")) >= 64
        assert r["flash_attn"] == "xla"
        # big models must ALSO stay unpinned: A100 is the contended card and
        # pinning it costs ~2h of queue versus an unpinned request, while
        # --mem >=64gb is what actually reaches the large nodes.
        assert r["gres"] == "gpu:1", (b, r["gres"])
        assert r["xla_mem_frac"] >= 3.2
    # no tier anywhere may name a card type
    for b in (256, 1536, 3072, 4096, 8192):
        assert resources_for(b)["gres"] == "gpu:1", b

    # monotonic, and every bucket resolves
    prev = 0
    for b in (256, 1536, 3072, 4096):
        m = int(resources_for(b)["mem"].replace("gb", ""))
        assert m >= prev
        prev = m


def test_submit_all_carries_per_group_resources():
    """submit_all.sh must override gres/mem per group and export the XLA knobs."""
    import os, tempfile
    from af3lis.pack import write_submit
    d = tempfile.mkdtemp()
    groups = [dict(dir=os.path.join(d, "group_000_b512"), bucket=512,
                   members=["a___b"], est_s=600.0, walltime="00:30:00"),
              dict(dir=os.path.join(d, "group_001_b4096"), bucket=4096,
                   members=["c___d"], est_s=6000.0, walltime="03:00:00")]
    sub = write_submit(groups, d, os.path.join(d, "pack_infer.sbatch"),
                       os.path.join(d, "out_pack"), 90)
    txt = open(sub).read()
    assert "--mem=48gb" in txt and "--mem=96gb" in txt, txt
    assert "AF3LIS_FLASH_ATTN=triton" in txt and "AF3LIS_FLASH_ATTN=xla" in txt
    assert "AF3LIS_XLA_MEM_FRAC=" in txt
    tsv = open(os.path.join(d, "groups.tsv")).read().splitlines()
    assert tsv[0].split("\t")[5:9] == ["gres", "mem", "flash_attn", "xla_mem_frac"]
    assert len(tsv[1].split("\t")) == len(tsv[0].split("\t")), "groups.tsv column mismatch"


def test_chain_tsv_trailing_comments_and_forms():
    """Annotated rows, whitespace separation, dedup, and the error paths."""
    import tempfile, os
    from af3lis.chain_input import read_chain_tsv
    d = tempfile.mkdtemp()

    def w(txt):
        p = os.path.join(d, "c.tsv")
        open(p, "w").write(txt)
        return p

    # trailing comments must not land in the sequence column
    a, b, ov = read_chain_tsv(w(
        "chain\tid\tsequence\n"
        "A\tP0CG48\t# ubiquitin C, 76 aa\n"
        "B\tP63165\t# SUMO1\n"))
    assert (a, b) == ("P0CG48", "P63165"), (a, b)
    assert ov == {}, ov

    # inline sequence still works, and is upper-cased
    _, _, ov = read_chain_tsv(w("A\tX\tmkvl\nB\tY\n"))
    assert ov == {"X": "MKVL"}, ov

    # whitespace-separated, no header, duplicate row dropped
    a, b, _ = read_chain_tsv(w("A UFL1\nA UFL1\nB DDRGK1\nB CDK5RAP3\n"))
    assert a == "UFL1" and b == "DDRGK1,CDK5RAP3", (a, b)

    # one-sided input is refused rather than silently folding nothing
    try:
        read_chain_tsv(w("A\tX\nA\tY\n"))
        raise AssertionError("one-sided TSV should be refused")
    except SystemExit:
        pass


def test_smoke_fixture_parses():
    """The shipped smoke fixture must stay valid -- it is the on-cluster test."""
    import os
    from af3lis.chain_input import read_chain_tsv
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    a, b, _ = read_chain_tsv(os.path.join(here, "examples", "smoke.tsv"))
    na, nb = len(a.split(",")), len(b.split(","))
    assert na == 2 and nb == 6, f"expected 2 baits x 6 preys, got {na}x{nb}"


def test_monomer_mode_dedups_and_merges_correctly():
    """Monomer alignment: one input per unique chain, and the merge must put
    the right MSA on the right chain. A silent chain swap here would corrupt
    every fold in a screen, so this asserts identity, not just plumbing."""
    import json, os, tempfile, glob, sys, subprocess
    from af3lis import monomer

    d = tempfile.mkdtemp()
    jd = os.path.join(d, "jsons")
    os.makedirs(jd)
    # 2 baits x 3 preys: 6 pairs, 12 chain-MSAs per-pair, 5 unique -> 2.4x
    baits = {"BAITX": "MKVLSPADKTNV", "BAITY": "MVHLTPEEKSAV"}
    preys = {"PREY1": "MSEQLTNAVLDA", "PREY2": "MQIFVKTLTGKT", "PREY3": "MADQLTEEQIAE"}
    for bn, bs in baits.items():
        for pn, ps in preys.items():
            nm = f"{bn}___{pn}"
            json.dump({"name": nm, "dialect": "alphafold3", "version": 1,
                       "modelSeeds": [1],
                       "sequences": [{"protein": {"id": "A", "sequence": bs}},
                                     {"protein": {"id": "B", "sequence": ps}}]},
                      open(os.path.join(jd, nm + ".json"), "w"))

    lf, n = monomer.write_monomers(d, [1])
    assert n == 5, f"expected 5 unique chains, got {n}"
    assert len(open(lf).read().split()) == 5

    # fake AF3 align output, embedding the query so the merge can be audited
    for mp in glob.glob(os.path.join(d, "monomers", "*.json")):
        md = json.load(open(mp))
        lab = md["name"]
        p = md["sequences"][0]["protein"]
        p["unpairedMsa"] = f">q\n{p['sequence']}\n"
        p["pairedMsa"] = f">q\n{p['sequence']}\n"
        p["templates"] = []
        od = os.path.join(d, "msa", lab)
        os.makedirs(od, exist_ok=True)
        json.dump(md, open(os.path.join(od, f"{lab}_data.json"), "w"))

    nmerged, missing = monomer.merge_all(d, [1], python_exe=sys.executable)
    assert nmerged == 6 and not missing, (nmerged, missing)

    for mp in glob.glob(os.path.join(d, "out", "*", "*_data.json")):
        md = json.load(open(mp))
        prot = [x["protein"] for x in md["sequences"] if "protein" in x]
        assert len(prot) == 2
        assert [x["id"] for x in prot] == ["A", "B"]
        lname = os.path.basename(os.path.dirname(mp))
        want = json.load(open(os.path.join(jd, lname.upper() + ".json")))
        wseq = [x["protein"]["sequence"] for x in want["sequences"]]
        assert [x["sequence"] for x in prot] == wseq, "chain order/sequence swapped"
        for x in prot:
            # the MSA carried onto this chain must be the one built FOR it
            q = x["unpairedMsa"].split("\n")[1]
            assert q == x["sequence"], "MSA belongs to a different sequence"

    # a missing chain MSA is reported, not fatal, and does not fake a result
    import shutil
    shutil.rmtree(glob.glob(os.path.join(d, "msa", "prey1*"))[0])
    shutil.rmtree(os.path.join(d, "out"))
    n2, miss2 = monomer.merge_all(d, [1], python_exe=sys.executable)
    assert n2 == 4 and len(miss2) == 2, (n2, miss2)


def _fake_align(d, labels=None):
    """Write a fake AF3 align output per monomer input, embedding the query
    so a merged chain can be traced back to the MSA it was given."""
    import glob, json
    for mp in glob.glob(os.path.join(d, "monomers", "*.json")):
        md = json.load(open(mp))
        lab = md["name"]
        if labels is not None and lab not in labels:
            continue
        p = md["sequences"][0]["protein"]
        p["unpairedMsa"] = f">q\n{p['sequence']}\n"
        p["pairedMsa"] = f">q\n{p['sequence']}\n"
        p["templates"] = []
        od = os.path.join(d, "msa", lab)
        os.makedirs(od, exist_ok=True)
        json.dump(md, open(os.path.join(od, f"{lab}_data.json"), "w"))


def _pair_jsons(d, pairs):
    import json
    jd = os.path.join(d, "jsons")
    os.makedirs(jd, exist_ok=True)
    for (an, aseq), (bn, bseq) in pairs:
        nm = f"{an}___{bn}"
        json.dump({"name": nm, "dialect": "alphafold3", "version": 1,
                   "modelSeeds": [1],
                   "sequences": [{"protein": {"id": "A", "sequence": aseq}},
                                 {"protein": {"id": "B", "sequence": bseq}}]},
                  open(os.path.join(jd, nm + ".json"), "w"))


def _assert_each_chain_got_its_own_msa(d):
    import glob, json
    merged = glob.glob(os.path.join(d, "out", "*", "*_data.json"))
    for mp in merged:
        md = json.load(open(mp))
        for x in (s["protein"] for s in md["sequences"] if "protein" in s):
            q = x["unpairedMsa"].split("\n")[1]
            assert q == x["sequence"], (mp, "MSA belongs to another sequence")
    return merged


def test_monomer_merge_resolves_labels_that_share_a_sequence():
    """Regression: write_monomers dedups by SEQUENCE but merge_all looked
    chains up by LABEL, so a second label carrying an already-aligned
    sequence -- a deliberate homodimer, a duplicated isoform -- had no MSA
    dir and its pair was reported MISSING. And two different sequences whose
    labels sanitise to the same name had the second one's pair silently
    merged onto the FIRST one's MSA. Merge must go label -> sequence ->
    the MSA computed for that sequence, with no hardlinks."""
    import sys, tempfile
    from af3lis import monomer
    d = tempfile.mkdtemp()
    bait, prey_a = "MKVLSPADKTNVKAAW", "MSEQLTNAVLDAGHLR"
    prey_c1, prey_c2 = "MQIFVKTLTGKTITLE", "MADQLTEEQIAEFKEA"
    _pair_jsons(d, [
        (("bait", bait), ("bait", bait)),             # literal self-pair
        (("bait", bait), ("bait_copy", bait)),        # homodimer, 2nd label
        (("bait", bait), ("preyA", prey_a)),
        (("bait", bait), ("preyA_copy", prey_a)),     # duplicated isoform
        (("bait", bait), ("prey+c", prey_c1)),        # sanitises to prey_c
        (("bait", bait), ("prey_c", prey_c2)),        # ... different sequence
    ])
    lf, n = monomer.write_monomers(d, [1])
    assert n == 4, f"4 unique sequences, got {n}"
    _fake_align(d)
    # only what was aligned exists: no msa dir for the duplicate labels
    msa_dirs = sorted(os.listdir(os.path.join(d, "msa")))
    assert "bait_copy" not in msa_dirs and "preya_copy" not in msa_dirs, msa_dirs
    assert len(msa_dirs) == 4, msa_dirs

    nmerged, missing = monomer.merge_all(d, [1], python_exe=sys.executable)
    assert not missing, missing
    assert nmerged == 6, nmerged
    merged = _assert_each_chain_got_its_own_msa(d)
    assert len(merged) == 6
    # no hardlink or copy was needed to get there
    for lab in ("bait_copy", "preya_copy"):
        assert not os.path.exists(os.path.join(d, "msa", lab))


def test_monomer_merge_skips_stale_msa_for_a_changed_sequence():
    """A rebuilt run dir can hold msa/<label>/ from an EARLIER sequence next
    to AF3's timestamped sibling with the new one. The stale MSA must never
    be merged onto the new sequence; with no matching MSA the pair is
    reported, not faked."""
    import json, shutil, sys, tempfile
    from af3lis import monomer
    d = tempfile.mkdtemp()
    bait, prey_new, prey_old = "MKVLSPADKTNVKAAW", "MSEQLTNAVLDAGHLR", "MSEQLTNAVLDAGHLK"
    _pair_jsons(d, [(("bait", bait), ("preyA", prey_new))])
    monomer.write_monomers(d, [1])
    _fake_align(d)
    good = os.path.join(d, "msa", "preya", "preya_data.json")
    sib = os.path.join(d, "msa", "preya_20261020_101010")
    os.makedirs(sib)
    shutil.move(good, os.path.join(sib, "preya_data.json"))
    stale = json.load(open(os.path.join(sib, "preya_data.json")))
    stale["sequences"][0]["protein"]["sequence"] = prey_old
    stale["sequences"][0]["protein"]["unpairedMsa"] = f">q\n{prey_old}\n"
    json.dump(stale, open(good, "w"))

    n, missing = monomer.merge_all(d, [1], python_exe=sys.executable)
    assert n == 1 and not missing, (n, missing)
    _assert_each_chain_got_its_own_msa(d)

    # only the stale one left -> reported missing, nothing written
    shutil.rmtree(sib)
    shutil.rmtree(os.path.join(d, "out"))
    n, missing = monomer.merge_all(d, [1], python_exe=sys.executable)
    assert n == 0 and missing == ["bait___preyA"], (n, missing)


def test_monomer_merge_ignores_inputs_left_by_an_earlier_build():
    """--rebuild rewrites af3_monomer_list.txt but leaves monomers/ alone.
    A stale input from the earlier build -- same sequence, an older label
    that sorts first -- must not become the label the merge looks for, or a
    pair whose MSA is on disk is reported MISSING. The list also holds
    absolute paths from wherever the build ran, so it must still resolve
    after the run dir has moved."""
    import shutil, sys, tempfile
    from af3lis import monomer
    root = tempfile.mkdtemp()
    d = os.path.join(root, "built_here")
    bait, prey = "MKVLSPADKTNVKAAW", "MSEQLTNAVLDAGHLR"
    _pair_jsons(d, [(("bait", bait), ("aaa_old", prey))])
    monomer.write_monomers(d, [1])
    shutil.rmtree(os.path.join(d, "jsons"))
    _pair_jsons(d, [(("bait", bait), ("preyA", prey))])   # the rebuild
    monomer.write_monomers(d, [1])
    assert sorted(os.listdir(os.path.join(d, "monomers"))) == \
        ["aaa_old.json", "bait.json", "preya.json"]
    _fake_align(d, labels={"bait", "preya"})              # what was aligned
    moved = os.path.join(root, "synced_elsewhere")
    shutil.move(d, moved)

    n, missing = monomer.merge_all(moved, [1], python_exe=sys.executable)
    assert n == 1 and not missing, (n, missing)
    _assert_each_chain_got_its_own_msa(moved)

    # no list at all (glob fallback picks the stale label first): the
    # chain's own label still finds its MSA, because its query matches
    os.remove(os.path.join(moved, "af3_monomer_list.txt"))
    shutil.rmtree(os.path.join(moved, "out"))
    assert monomer.monomer_index(moved)[prey] == "aaa_old"
    n, missing = monomer.merge_all(moved, [1], python_exe=sys.executable)
    assert n == 1 and not missing, (n, missing)
    _assert_each_chain_got_its_own_msa(moved)


def test_monomer_label_fallback_never_trusts_a_name():
    """The own-label fallback accepts only an MSA whose query IS the chain's
    sequence: an msa/<label>/ built for something else stays MISSING."""
    import json, shutil, sys, tempfile
    from af3lis import monomer
    d = tempfile.mkdtemp()
    bait, prey, other = "MKVLSPADKTNVKAAW", "MSEQLTNAVLDAGHLR", "MQIFVKTLTGKTITLE"
    _pair_jsons(d, [(("bait", bait), ("preyA", prey))])
    monomer.write_monomers(d, [1])
    _fake_align(d)
    wrong = os.path.join(d, "msa", "preya", "preya_data.json")
    md = json.load(open(wrong))
    md["sequences"][0]["protein"]["sequence"] = other
    md["sequences"][0]["protein"]["unpairedMsa"] = f">q\n{other}\n"
    json.dump(md, open(wrong, "w"))
    os.remove(os.path.join(d, "af3_monomer_list.txt"))
    shutil.copy(os.path.join(d, "monomers", "preya.json"),
                os.path.join(d, "monomers", "aaa.json"))
    md = json.load(open(os.path.join(d, "monomers", "aaa.json")))
    md["name"] = "aaa"
    json.dump(md, open(os.path.join(d, "monomers", "aaa.json"), "w"))
    n, missing = monomer.merge_all(d, [1], python_exe=sys.executable)
    assert n == 0 and missing == ["bait___preyA"], (n, missing)


def test_submit_all_captures_job_ids_for_analyse_chaining():
    """The packed group ids must be captured, because ANALYSE has to depend on
    them and they do not exist until submit_all.sh runs. Chaining analyse on
    the bridge instead was a real bug: the bridge exits in seconds, so analyse
    ran before any model existed and died with "no models"."""
    import os, tempfile, subprocess, sys
    from af3lis.pack import write_submit
    d = tempfile.mkdtemp()
    groups = [dict(dir=os.path.join(d, "group_000_b512"), bucket=512,
                   members=["a___b"], est_s=600.0, walltime="00:30:00"),
              dict(dir=os.path.join(d, "group_001_b768"), bucket=768,
                   members=["c___d"], est_s=900.0, walltime="00:45:00")]
    sub = write_submit(groups, d, os.path.join(d, "pack_infer.sbatch"),
                       os.path.join(d, "out_pack"), 90)
    txt = open(sub).read()
    assert "--parsable" in txt, "ids cannot be captured without --parsable"
    assert "job_ids.txt" in txt, "no id file written"
    assert txt.count("IDS=") >= 3, "ids not accumulated per group"

    # run it with a stub sbatch on PATH and check the id file lands
    bindir = os.path.join(d, "bin")
    os.makedirs(bindir, exist_ok=True)
    stub = os.path.join(bindir, "sbatch")
    with open(stub, "w") as fh:
        fh.write("#!/bin/sh\necho 99$$\n")
    os.chmod(stub, 0o755)
    env = dict(os.environ, PATH=bindir + os.pathsep + os.environ["PATH"])
    r = subprocess.run(["bash", sub], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    idf = os.path.join(d, "job_ids.txt")
    assert os.path.exists(idf), "job_ids.txt not created"
    ids = [x for x in open(idf).read().split() if x]
    assert len(ids) == 2, f"expected 2 ids, got {ids}"
    assert "PACK_SUBMITTED" in r.stdout


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
