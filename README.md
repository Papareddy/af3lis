# af3lis — AlphaFold 3 interface screening with LIS / actifpTM / PEAK

AlphaFold-3 mirror of [`boltzlis`](https://github.com/Papareddy/boltzlis). Give **chain A**
and **chain B** as protein IDs (each one *or many*), get AF3 input JSONs + a two-stage SLURM
chain (CPU MSA -> GPU inference), and -- after the run -- a single ranked table with
**every interface metric**:

| metric | what it is |
|---|---|
| **iLIS** = sqrt(LIS*cLIS) | AFM-LIS primary score; project house cutoff **iLIS >= 0.22** (AFM-paper value: 0.223) |
| LIS / cLIS | local interaction score (all / contact-restricted) |
| LIA / cLIA | local interaction *area* (interface size) |
| **actifpTM** | interface-restricted ipTM (flank-robust) |
| ipSAE | Dunbrack aligned-error interface score |
| **PEAK** | 1 - min(inter-chain PAE)/30 -- house metric |
| ipTM / pTM / pLDDT | AlphaFold 3 reports these — pTM is Zhang & Skolnick 2004; ipTM from AlphaFold-Multimer (Evans 2021) |

Metrics come from vendored [AFM-LIS](https://github.com/flyark/AFM-LIS) `lis.py`
(`--platform alphafold3`, parallel `-w`) plus a PEAK pass on the AF3 PAE + `token_chain_ids`,
aggregated per-seed-then-cross over diffusion samples. The output TSV schema is
**byte-identical to boltzlis** in `--agg flat` mode, so the same beeswarm plotter and
PEAK-ranking convention work across engines.

### Reading the metrics — PEAK vs actifpTM vs iLIS (+ a scipy gotcha)
- **PEAK and actifpTM are both *confidence* axes and tend to correlate** (PEAK = 1−min-interchain-PAE/30; actifpTM = interface-restricted ipTM). A high value on either means a *locally* confident contact — but a few low-PAE residues can inflate PEAK/actifpTM even when no real interface forms.
- **iLIS is the orthogonal check** — it scores interface contact *extent/density*, so it is **not** inflated by a tiny confident contact. A **high-PEAK / low-iLIS** fold is the signature of a small or spurious interface. Require **both** (house bar: PEAK ≥ 0.7 *and* iLIS ≥ 0.22); treat single-metric (PEAK-only) hits as provisional, and prefer multi-seed + cross-engine (Boltz↔AF3) agreement.
- **iLIS needs `scipy`.** `lis.py` imports `scipy.spatial.distance`; if the collect Python lacks scipy, **lis.py fails silently** — `metrics.tsv` comes back with PEAK populated but **iLIS empty/zero**, and `collect` still exits 0. Always collect with a numpy + **scipy** + pandas env and sanity-check that the iLIS column is non-empty.

> **What this is.** `af3lis` is a thin **orchestration wrapper** -- built by
> **Ranjith Papareddy** -- that runs **AlphaFold 3** for structure prediction and computes
> *published* interface-confidence metrics (**LIS / iLIS**, **actifpTM**, **ipSAE**, **PEAK**).
> The modelling and the metrics are other people's science; af3lis just automates the
> boring part end-to-end -- IDs -> sequences -> folds -> all metrics -> one ranked table,
> across a cluster. **If you use it, please cite the underlying methods** (see
> [Credits & citation](#credits--citation)).

---

## Step-by-step guide (local + Helix / bwForCluster)

**Who runs what, where** -- the work splits across four contexts:

| step | runs on | needs |
|---|---|---|
| 1. build inputs (resolve IDs -> sequences -> JSONs) | **your laptop** (or any internet machine) | internet (UniProt), Python |
| 2. AF3 MSA (CPU) + inference (GPU) | **Helix CPU + GPU nodes** | `bio/alphafold/3.0.1` + AF3 weights + DBs |
| 3. collect metrics | **Helix** (or laptop, if outputs copied) | numpy + scipy + pandas |
| 4. plot / inspect | laptop | matplotlib |

> A coding agent (e.g. Claude Code) can do all four end-to-end: give it this repo +
> SSH access to Helix and it can install, build, submit, collect, and pull results.

### 0. One-time install -- laptop
```bash
git clone https://github.com/Papareddy/af3lis.git af3lis_pipeline
cd af3lis_pipeline
python -m venv .venv && source .venv/bin/activate     # or: conda create -n af3lis python=3.11
pip install -r requirements.txt                        # numpy scipy pandas matplotlib pytest
pip install -e .                                       # installs the 'af3lis' console script
pytest -xvs tests/test_metrics.py                      # -> "all tests passed"
```
(Test fixtures live under `tests/data/`; if absent on a fresh clone, the suite
will skip the fixture-dependent tests and report which ones it ran.)

### 1. One-time setup -- Helix

AlphaFold 3 weights are **gated by DeepMind** and not redistributable. You must request
access yourself (https://github.com/google-deepmind/alphafold3) and place the params under
`$HOME/af3-models/` on the cluster.

```bash
ssh helix                                               # your bwForCluster login
# (a) a workspace to hold runs (60-day, extendable):
ws_allocate af3lis 60                                   # -> prints a path; call it $WS
export WS=$(ws_find af3lis)

# (b) clone the repo on Helix too (needed for collect + lis.py):
git clone https://github.com/Papareddy/af3lis.git $WS/af3lis_pipeline

# (c) AF3 weights (gated, one-time, ~1 GB compressed):
#     after you've been approved by DeepMind, place the params (`af3.bin.zst`)
#     under $HOME/af3-models/ and decompress with zstd:
mkdir -p $HOME/af3-models
#     zstd -d af3.bin.zst -o $HOME/af3-models/af3.bin
ls $HOME/af3-models                                     # should list af3.bin (+ checksums)
#     model_dir in config.yaml points at this directory (not the file itself).

# (d) AF3 reference DBs (~628 GB). DEFAULT = system-wide $ALPHAFOLD_DATABASES from the
#     bio/alphafold module. If your site already exports it, skip; otherwise:
#       module avail bio/alphafold                      # -> exact module tag on your site
#       module load bio/alphafold/3.0.1
#       echo $ALPHAFOLD_DATABASES                       # -> sanity-check the path
#     If $ALPHAFOLD_DATABASES is unset on your cluster, you MUST point `db_dir`
#     in config.yaml at a local mirror — see DeepMind's `fetch_databases.sh`.
#     Only mirror a private copy if your site does NOT ship the DBs.

# (e) an analysis env for collect/lis (numpy+scipy+pandas):
conda create -y -n analysis python=3.11 numpy scipy pandas matplotlib pytest
conda activate analysis
pip install -e $WS/af3lis_pipeline                      # installs the 'af3lis' console script
#     (AF3 itself runs from the module's own env; this env is for post-processing only.)

# (f) write your config (gitignored; never commit real paths):
cd $WS/af3lis_pipeline
cp config.example.yaml config.yaml
#     edit config.yaml ->  af3_module: bio/alphafold/3.0.1
#                          model_dir:  $HOME/af3-models
#                          db_dir:     ${ALPHAFOLD_DATABASES}
#                          python:     <analysis python with numpy/scipy/pandas>
#                          workspace:  $WS
#                          align/infer: partition/time/mem/cpus/gpu_gres for your site
```

### 2. Build inputs -- laptop
```bash
af3lis build --name UFM_screen \
    --chainA UFL1,UFC1 \
    --chainB DDRGK1,CDK5RAP3 \
    --fasta examples/ufm_machinery.fasta \
    --outdir runs/UFM_screen \
    --seeds 1 --num-samples 5 --num-recycles 10
# writes runs/UFM_screen/{jsons/*.json, af3_json_list.txt,
#                         align.sbatch, infer.sbatch, RUNBOOK.md}
```
- IDs = a **name in your `--fasta`** file (above), a **UniProt accession** (`P61960`),
  a **TAIR locus** (`AT1G01010`), or **`LABEL=SEQUENCE`**. `LABEL=` is just a display name.
  Bundling a local FASTA keeps your targets off any web lookup.
- `--chainA` / `--chainB` are comma-separated lists -> "single or multiple" scales the
  grid: `1x1`, `Nx1`, or `NxM` pairwise 2-chain folds.
- One multi-chain assembly instead of a grid: `--mode complex --complex-name MyComplex`.
- `--seeds` are **baked into the JSON in stage 1** (they key the diffusion noise; the MSA is
  sequence-keyed, but AF3 still requires re-running align if you change `modelSeeds` because
  the augmented `<lname>_data.json` archives the seed list). To change seeds, rebuild and
  re-align; `af3lis submit` precondition-checks this and refuses to silently mismatch.
- Pair-name separator is **triple underscore** (`___`); labels survive single underscores
  cleanly (no `FOO_2` vs `FOO__2` collision).

### 3. Sync + submit -- Helix
```bash
# $WS expands on the LAPTOP (it has no $WS). Two options:
#   (a) hard-code the workspace path printed by ws_allocate, or
#   (b) ssh helix 'echo $WS' first to capture it locally.
WS_REMOTE=$(ssh helix 'echo $WS')                       # one-shot capture
rsync -av runs/UFM_screen  helix:$WS_REMOTE/runs/        # copy the built run dir over
ssh helix
cd $WS/af3lis_pipeline
conda activate analysis                                  # 'af3lis' lives in this env
af3lis submit --outdir $WS/runs/UFM_screen --config config.yaml
# under the hood:
#   align_id=$(sbatch --parsable --array=1-N align.sbatch)
#   sbatch --dependency=afterok:$align_id --array=1-N infer.sbatch
squeue --me                                              # logs in runs/.../logs/
```
AF3 writes models into `$WS/runs/UFM_screen/out/<lname>/...` (the
`{OUTDIR}/out` directory the sbatch templates create).

Two-stage flow:

| stage | partition | what it does | wallclock |
|---|---|---|---|
| **align** | `cpu-single` | jackhmmer/nhmmer MSA + templates; writes `<lname>_data.json` per pair | 12 h default |
| **infer** | `gpu-single` (A100) | diffusion sampling from MSA; writes models + confidences | 2 h default; bump to 6 h for >2500 residues |

Useful submit flags:
- `--align-only` / `--infer-only` -- run one stage at a time (debugging MSAs / re-running GPU).
- `--no-dependency` -- submit infer without an `afterok` gate. Combine with `--infer-only`
  to skip align entirely; otherwise BOTH stages still run, just unchained.
- `--dry-run` -- write `SUBMIT_CMDS.sh` without sbatching (dependency placeholders preserved).
- `--time-align HH:MM:SS` / `--time-infer HH:MM:SS` -- per-submission wallclock overrides.

> **Large complexes (~3100 residues).** Default `XLA_CLIENT_MEM_FRACTION=3.2`
> + `TF_FORCE_UNIFIED_MEMORY=true` (in the infer template) enables GPU->host spill on
> A100-40 GB. Switch `flash_attn: xla` and bump to A100-80 GB beyond ~3500 residues.
> JAX cold-start compiles the model graph (~5-10 min/node) on the first job of every node
> -- bake into wallclock estimates even for tiny test pairs.

### 3b. PACKED inference -- recommended for >~30 folds on low fairshare

Packing strategy after **Mau's bwHelix AF3 pipeline toolkit** (Aug 2026). The per-task
array is the wrong shape on a scarce-GPU, low-fairshare account: bwHelix accrues age
priority on only ~100 jobs (`MaxJobsAccruePU=100`), backfill-tests 128 per cycle, and
**array tasks forfeit their accrued age** -- while every task re-pays weights-load + XLA
compile (~5-10 min). Packed inference instead runs MANY `*_data.json` through ONE
`run_alphafold.py --input_dir` process per job: weights load once, XLA compiles once per
token bucket. `af3lis.pack` builds **single-bucket groups** sized to hit a target
walltime (small complexes pack many per job, big ones few), so a job never blows its
wallclock because sizes were mixed. Measured on bwHelix: a ~100-condition bucket-512
screen ≈ 1 h in one packed job vs days queued as an array.

```bash
# after the ALIGN array completes, on the cluster:

# (a) new size class or GPU? probe + calibrate first (one ~20-min GPU job):
python pipeline.py --pack --probe 10 --outdir $WS/runs/UFM_screen
bash $WS/runs/UFM_screen/af3pack_probe/submit_all.sh          # -> probe_out/ (isolated)
python -m af3lis.calibrate $WS/runs/UFM_screen/logs/pack_*.out \
       --out $WS/runs/UFM_screen/calib.json

# (b) size + submit the campaign (auto-uses <outdir>/calib.json when present):
python pipeline.py --pack --outdir $WS/runs/UFM_screen [--target-hours 1.5]
bash $WS/runs/UFM_screen/af3pack/submit_all.sh
# models -> out_pack/ (separate from out/, which holds the align-stage data JSONs);
# collect exactly as in step 4 but on out_pack/
```

Pack mechanics worth knowing:
- Groups are **symlinks** into `out/<lname>/` (`--pack-copy` to copy); conditions whose
  align task failed are excluded and listed in `af3pack_missing.txt` (re-align, re-pack).
- Walltimes come from a per-bucket seconds table; only buckets 512/768 were measured
  (A100) -- **always probe + `--calib` for a new size class**. Safety margin `--safety 1.3`.
- AF3 writes each condition's output as it finishes, so a timed-out group loses only its
  un-run tail; the job log prints `MISSING: <lname>` lines for exactly those (delete those
  partial dirs before re-running the group, else AF3 diverts to timestamped siblings).
- Keep total pending jobs <= ~100; never `scancel` a pending job to retime it --
  `scontrol update JobId=<id> TimeLimit=<minutes>` (decrease only) preserves accrued age.

### 4. Collect every metric -- Helix (one command)
```bash
# after the array finishes (conda env 'analysis' carries numpy/scipy/pandas):
conda activate analysis
af3lis collect $WS/runs/UFM_screen/out -o $WS/runs/UFM_screen/metrics.tsv -w 8
# or back-compat shortcut (matches boltzlis muscle memory):
af3lis --collect $WS/runs/UFM_screen/out -o $WS/runs/UFM_screen/metrics.tsv -w 8
```
`metrics.tsv` = one row per pair, ranked by `iLIS_max_max` (or `iLIS_max` in flat mode),
with **per-seed-then-cross mean+max of every metric** (iLIS / LIS / cLIS / LIA / cLIA /
ipSAE / actifpTM / ipTM / pTM / PEAK / pLDDT) **plus the Dunbrack `ipsae.py` family**
(`ipSAE_d0res` / `ipSAE_d0chn` / `ipSAE_d0dom` / `ipTM_d0chn` / `pDockQ` / `pDockQ2` /
`LIS_ipsae`; skip with `--no-ipsae`). The per-model CSV with residue-level LIR /
cLIR lives alongside as `metrics.tsv.permodel.csv`.

> Cross-check for free: our `lis.py`-derived `ipSAE` and `ipsae.py`'s `ipSAE_d0res` are
> independent implementations of the same score -- they agree to ~1e-3 on the same model
> (verified on ECT5 x NOT1-SH). A large disagreement = investigate the inputs.
> `--agg flat` is byte-identical to the boltzlis schema **only with `--no-ipsae`**
> (the ipsae columns are appended after pLDDT otherwise).

Aggregation modes (`--agg`):

| mode | columns per metric | when to use |
|---|---|---|
| `per_seed` (default) | 4 (`_mean_mean`, `_mean_max`, `_max_mean`, `_max_max`) | multi-seed AF3 runs -- avoids the upward bias of `max` over correlated diffusion samples within a seed |
| `flat` | 2 (`_mean`, `_max`) -- **byte-identical to boltzlis** | direct cross-engine comparison vs Boltz-2 (match `--num-samples` between engines for fairness) |

`--rank-by iLIS_max` (boltzlis vocabulary) auto-translates to `iLIS_max_max` in `per_seed` mode.

### 5. Pull + inspect -- laptop
```bash
rsync -av helix:$WS/runs/UFM_screen/metrics.tsv         runs/UFM_screen/
rsync -av helix:$WS/runs/UFM_screen/metrics.tsv.permodel.csv  runs/UFM_screen/
column -t -s$'\t' runs/UFM_screen/metrics.tsv | less -S

# the house beeswarm (PEAK, one dot per A-side, grouped by B-side, control anchor):
af3lis plot --tsv runs/UFM_screen/metrics.tsv \
            -o    runs/UFM_screen/beeswarm_PEAK.pdf \
            --metric PEAK_max_max \
            --group-by prey \
            --control UFL1___DDRGK1 \
            --threshold-peak 0.7 --threshold-ilis 0.22
# auto-opens via `open` on macOS; no-op on Linux/cluster.
```
Rule of thumb: a confident interaction passes **iLIS >= 0.22 AND PEAK >= 0.7**.

---

## Metrics -- provenance

Every column in `metrics.tsv` comes from one of three places: (1) AF3 JSON read by `lis.py`,
(2) AF3 PAE + `token_chain_ids` read by `af3lis.collect.peak_table`, (3) `peak_per_chainpair`
arithmetic on the **token axis**. No metric is computed in `af3lis/` that boltzlis doesn't
also produce.

| metric | source file | computed by | notes |
|---|---|---|---|
| `pTM` | `<lname>_summary_confidences_<N>.json` -> `ptm` | lis.py | scalar 0-1 |
| `ipTM` | same -> `iptm` | lis.py | scalar 0-1; global ipTM (all interfaces) |
| chain-pair ipTM | same -> `chain_pair_iptm` | lis.py | `[n_chains, n_chains]`; row order = `label_asym_id` |
| `pLDDT_i/j` | `<lname>_full_data_<N>.json` -> `atom_plddts` + `atom_chain_ids` | lis.py | per-chain mean |
| **PAE matrix** | `<lname>_full_data_<N>.json` -> `pae` | lis.py + af3_io | asymmetric, **token x token**, NaN->31.0 |
| `token_chain_ids` | `<lname>_full_data_<N>.json` -> `token_chain_ids` | af3_io | canonical chain assignment for PEAK |
| `iLIS / LIS / cLIS` | derived from PAE + Cbeta contacts (12/8 A) | lis.py | recomputed (AF3 doesn't ship these) |
| `iLIA / LIA / cLIA` | derived from PAE + contacts | lis.py | interface residue counts |
| `actifpTM` | derived from PAE + ipTM head | lis.py | recomputed per-interface |
| `ipSAE` | derived from PAE (hardcoded cutoff 10) | lis.py | Dunbrack score |
| `ipSAE_d0res/d0chn/d0dom`, `ipTM_d0chn`, `pDockQ`, `pDockQ2`, `LIS_ipsae` | per-sample `confidences.json` + `model.cif` | vendored `ipsae.py` (Dunbrack v3) via `ipsae_runner` | `max`-type row per chain pair; cutoffs `--ipsae-pae-cutoff 10 --ipsae-dist-cutoff 10`; from Mau's toolkit |
| `LIR_i/j`, `cLIR_i/j` | per-model CSV only | lis.py | not in final TSV; same as boltzlis |
| **PEAK** | `<lname>_full_data_<N>.json` -> `pae` + `token_chain_ids` | af3lis | `max(0, 1 - min(off-diag PAE block)/30)`; token-axis, matches AF3's native `chain_pair_pae_min` to JSON-rounding precision (~3e-4; unit-tested) |
| `ranking_score`, `has_clash`, `fraction_disordered`, `chain_pair_pae_min` | `<lname>_summary_confidences_<N>.json` | af3_io (loaded, not in TSV) | available for downstream filtering / sanity checks |
| `seed`, `sample`, `flat_index` | `<lname>_ranking_scores.csv` | af3_io | strict header-checked parse; flat_index = AF3 `<N>` |

PEAK is computed on the **token axis** of the AF3 PAE (length `pae.shape[0]`, not residue
count) so that any ligand or PTM tokens come along for the ride. The CIF parser is used only
for chain ordering and unit-test cross-checks.

---

## Layout
```
pipeline.py                       thin shim -> af3lis.pipeline:main (boltzlis-style entry)
af3lis/
  fetch.py                        ID (UniProt acc | TAIR locus | seq) -> sequence (+cache)
  json_build.py                   chain-sets -> AF3 input JSON(s)  (grid | complex)
  structure.py                    CIF parser + token-axis PEAK chain split
  af3_io.py                       single source of truth for AF3 output paths + FoldResult
  collect.py                      lis.py + PEAK + ipsae.py -> ranked per-pair metric table
                                  [af3lis collect ...   or   af3lis --collect ...]
  lis.py                          vendored AFM-LIS (iLIS/LIS/cLIS/LIA/ipSAE/actifpTM)
  ipsae.py                        vendored Dunbrack ipsae.py v3 (ipSAE_d0*/pDockQ/pDockQ2)
  ipsae_runner.py                 drives ipsae.py per sample, parses the max rows
  pack.py                         packed-inference grouper (strategy after Mau's toolkit)
  calibrate.py                    per-bucket seconds from a packed job log -> calib.json
slurm/
  af3_align.sbatch.tmpl           CPU MSA stage
  af3_infer.sbatch.tmpl           GPU inference stage (afterok chained; per-task array)
  af3_pack_infer.sbatch.tmpl      PACKED GPU inference (--input_dir; driven by af3pack/)
pyproject.toml                    package metadata (entry point: af3lis -> pipeline.main)
requirements.txt                  pinned numpy/scipy/pandas/matplotlib/pytest
.gitignore                        excludes config.yaml + seq_cache + runs/
LICENSE                           MIT
config.example.yaml               cluster paths/resources (generic placeholders)
tests/                            local unit tests (no cluster/GPU needed)
<per-run outputs>
  runs/<name>/RUNBOOK.md          regenerated on each build — submit + collect commands
  runs/<name>/af3_json_list.txt   one absolute JSON path per line (SLURM array driver)
  runs/<name>/{align,infer}.sbatch  rendered SLURM templates
  runs/<name>/jsons/<pair>.json   AF3 input JSON per pair (or per complex)
```

---

## Notes

- Sequence fetch runs **locally** (compute nodes have no internet); AF3 MSA and inference
  run on the cluster; metric collection runs anywhere with numpy+scipy.
- iLIS 0.223 is AF-Multimer/Y2H-calibrated -- treat as approximate on AF3 (AFM-LIS authors
  report AF3 PAE is on a slightly different scale than AFM, but the threshold is robust in
  practice). PEAK is engine-agnostic.
- `collect.py` ingests standard AF3 CLI output dirs (`<out>/<lname>/<lname>_model_<N>.cif`
  etc., flat sample index across seeds*samples) via `lis.py`'s native AF3 adapter
  (`--platform alphafold3`).
- AF3 lowercases the JSON `name` field into the output directory. A JSON named
  `UFL1___DDRGK1.json` becomes `out/ufl1___ddrgk1/`. `af3_io.iter_jobs` is case-blind.
- `<lname>_data.json` is the per-pair reproducibility artifact (augmented MSAs + seed list).
  **Never `rm -rf` per-pair output dirs after collect** -- there is no cleanup script and
  re-running the MSA is the most expensive step.
- 26-chain cap (`A..Z`); AF3 supports more via multi-letter `label_asym_id` but our screens
  never exceed 26 and the cap simplifies downstream chain-ID arithmetic.
- `pTM < 0.05` floor for systems with <20 tokens -- don't use pTM/ipTM as a confidence
  filter for tiny complexes; PEAK/iLIS remain meaningful. `af3lis plot` warns on stderr.

---

## Credits & citation

`af3lis` (this wrapper) is by **Ranjith Papareddy**, MIT-licensed. It does **not** introduce
a new method -- it orchestrates and scores the tools below. **If `af3lis` is useful in your
work, cite the underlying methods** (and a link to this repo is appreciated):

| component | what it does here | cite |
|---|---|---|
| **AlphaFold 3** | structure prediction (the folds) | Abramson *et al.* 2024, *Nature* [10.1038/s41586-024-07487-w](https://doi.org/10.1038/s41586-024-07487-w) |
| **iLIS / LIS / cLIS / LIA** | local interaction scores (vendored `lis.py`) | LIS+LIA: Kim *et al.* 2024, bioRxiv [10.1101/2024.02.19.580970](https://doi.org/10.1101/2024.02.19.580970) -- repo: [flyark/AFM-LIS](https://github.com/flyark/AFM-LIS). iLIS (sqrt(LIS*cLIS)) was introduced in the same line of work; confirm the exact iLIS citation/title at the upstream repo before publishing. |
| **actifpTM** | interface-restricted ipTM | Varga, Ovchinnikov & Schueler-Furman 2025, *Bioinformatics* [10.1093/bioinformatics/btaf107](https://doi.org/10.1093/bioinformatics/btaf107) |
| **ipSAE** | aligned-error interface score | Dunbrack 2025, bioRxiv [10.1101/2025.02.10.637595](https://doi.org/10.1101/2025.02.10.637595) |
| **ipsae.py v3** (vendored) | ipSAE_d0res/d0chn/d0dom + pDockQ + pDockQ2 + LIS reference implementation | same Dunbrack 2025 preprint; script MIT-style (header retained). pDockQ: Bryant *et al.* 2022 [10.1038/s41467-022-28865-w](https://doi.org/10.1038/s41467-022-28865-w); pDockQ2: Zhu *et al.* 2023 [10.1093/bioinformatics/btad424](https://doi.org/10.1093/bioinformatics/btad424) |
| **packed inference** | `--input_dir` batching + size-aware grouping + bwHelix queue playbook | strategy and measured queue numbers from **Mau's bwHelix AF3 pipeline toolkit** (personal communication, Aug 2026) -- ask before redistributing |
| **ipTM / pTM** | confidence scalars reported by AF3 — pTM from Zhang & Skolnick 2004, ipTM from AlphaFold-Multimer | Evans *et al.* 2021 [10.1101/2021.10.04.463034](https://doi.org/10.1101/2021.10.04.463034) |

`PEAK` ( = 1 - min-interchain-PAE/30 ) is a convenience scalar defined in this repo; no separate citation.

<details><summary>BibTeX</summary>

> DOIs and author-years are verified from source. The Kim *et al.* (LIS/LIA) **title**
> below is abbreviated -- confirm the exact title/author list at the DOI before citing.
> The iLIS-specific citation entry was removed pending verification — cite the 2024 LIS
> bioRxiv as the primary AFM-LIS reference and consult the upstream repo for the latest.

```bibtex
@article{abramson2024alphafold3, title={Accurate structure prediction of biomolecular interactions with AlphaFold 3}, author={Abramson, Josh and Adler, Jonas and Dunger, Jack and Evans, Richard and Green, Tim and Pritzel, Alexander and Ronneberger, Olaf and Willmore, Lindsay and Ballard, Andrew J. and Bambrick, Joshua and others}, journal={Nature}, year={2024}, doi={10.1038/s41586-024-07487-w}}
@article{kim2024lis, title={Enhancing Protein-Protein Interaction Prediction with Local Interaction Score from AlphaFold-Multimer}, author={Kim, Ah-Ram and others}, journal={bioRxiv}, year={2024}, doi={10.1101/2024.02.19.580970}}
@article{varga2025actifptm, title={actifpTM: a refined confidence metric of AlphaFold2 predictions involving flexible regions}, author={Varga, Julia K. and Ovchinnikov, Sergey and Schueler-Furman, Ora}, journal={Bioinformatics}, year={2025}, doi={10.1093/bioinformatics/btaf107}}
@article{dunbrack2025ipsae, title={Res ipSAE loquunt: What's wrong with AlphaFold's ipTM score and how to fix it}, author={Dunbrack, Roland L.}, journal={bioRxiv}, year={2025}, doi={10.1101/2025.02.10.637595}}
@article{evans2021afm, title={Protein complex prediction with AlphaFold-Multimer}, author={Evans, Richard and O'Neill, Michael and Pritzel, Alexander and others}, journal={bioRxiv}, year={2021}, doi={10.1101/2021.10.04.463034}}
```
</details>
