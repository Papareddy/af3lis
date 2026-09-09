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
    assert bucket_of(3100) == 4096
    assert bucket_of(5121) == 6144      # beyond ladder: ceil to 1024


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


def test_calibrate_median_and_startup():
    log = (
        "I0825 10:00:00.000000 start\n"
        "I0825 10:00:01.500000 Calculating bucket size for input with 500 tokens.\n"
        "I0825 10:00:02.000000 Got bucket size 512 for input with 500 tokens.\n"
        "I0825 10:01:10.000000 Calculating bucket size for input with 490 tokens.\n"
        "I0825 10:01:10.500000 Got bucket size 512 for input with 490 tokens.\n"
        "I0825 10:01:44.000000 Calculating bucket size for input with 505 tokens.\n"
        "I0825 10:01:44.200000 Got bucket size 512 for input with 505 tokens.\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as fh:
        fh.write(log)
        path = fh.name
    try:
        out = calibrate(path)
        # gaps attributed to the PREVIOUS condition's bucket: 68.5, 34.0
        assert out["512"] == 51.2 or abs(out["512"] - 51.2) < 0.2  # median of 2
        assert abs(out["startup"] - 1.5) < 0.01
        assert out["compile"] == 40.0
    finally:
        os.unlink(path)


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
    for b in (256, 512, 768, 1024, 1280, 1536, 2048, 3072):
        r = resources_for(b)
        mem_gb = int(r["mem"].replace("gb", ""))
        assert mem_gb < 64, (b, r["mem"])
        assert r["flash_attn"] == "triton"
    # big buckets escalate deliberately
    for b in (4096, 5120, 8192):
        r = resources_for(b)
        assert int(r["mem"].replace("gb", "")) >= 64
        assert r["flash_attn"] == "xla"
        assert r["xla_mem_frac"] >= 3.2
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
