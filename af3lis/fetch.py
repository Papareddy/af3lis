"""Resolve protein IDs -> amino-acid sequences.

Accepts, per ID:
  * UniProt accession   (e.g. P69905, Q9FPE7)          -> uniprotkb/<acc>.fasta
  * Arabidopsis TAIR locus (e.g. AT1G01010)            -> UniProt xref:tair search
  * name=ID  or  name=SEQUENCE                          -> explicit label / inline seq
A local FASTA (via load_fasta) overrides any network lookup.
Results are cached under cache_dir to avoid refetching.
"""
import os, re, json, sys, time, urllib.request, urllib.parse
from urllib.error import HTTPError

UNIPROT_ACC = re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$")
TAIR_LOCUS = re.compile(r"^AT[1-5MC]G\d{5}$", re.I)
AA = set("ACDEFGHIKLMNPQRSTVWYXBZUO")


def _get(url, tries=3):
    """HTTP GET with retries. Backoff = 2*(i+1)s = 2s, 4s.
    v1 improvement over boltzlis: HTTP 404 is treated as a non-retryable
    hard miss and re-raised on the first attempt (saves ~10s per typo
    during interactive `build`)."""
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.read().decode()
        except HTTPError as e:
            # 404 = hard miss: do not retry.
            if e.code == 404:
                raise
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def _fasta_seq(text):
    return "".join(l.strip() for l in text.splitlines() if l and not l.startswith(">"))


def load_fasta(path):
    """Parse a FASTA file -> {name: sequence}. Use to override network lookups."""
    seqs, name = {}, None
    for line in open(path):
        line = line.rstrip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            seqs[name] = ""
        elif name:
            seqs[name] += line.strip()
    return seqs


def resolve_one(token, cache_dir=None, overrides=None, organism_id=3702):
    """token -> (label, sequence). token may be 'LABEL=VALUE' or just 'VALUE'.
    VALUE is a UniProt acc, a TAIR locus, or a raw sequence. overrides: {label/id: seq}."""
    label, _, value = token.partition("=")
    if not value:                       # no '=' -> value is the whole token, label = it
        label, value = token, token
    overrides = overrides or {}
    for key in (label, value):
        if key in overrides:
            return label, overrides[key]
    v = value.strip().upper()
    # raw sequence?
    if len(v) >= 20 and set(v) <= AA and not TAIR_LOCUS.match(v):
        return label, v
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cf = os.path.join(cache_dir, value + ".fasta")
        if os.path.exists(cf):
            return label, _fasta_seq(open(cf).read())
    if UNIPROT_ACC.match(v):
        text = _get("https://rest.uniprot.org/uniprotkb/%s.fasta" % value)
    elif TAIR_LOCUS.match(v):
        q = "(xref:tair-%s) AND (organism_id:%d)" % (value, organism_id)
        url = "https://rest.uniprot.org/uniprotkb/search?query=%s&format=fasta&size=1" % urllib.parse.quote(q)
        text = _get(url)
        if not text.strip():
            raise ValueError("no UniProt entry for TAIR locus %s" % value)
    else:
        raise ValueError("unrecognised ID '%s' (not UniProt acc / TAIR locus / sequence). "
                         "Pass it via --fasta or as LABEL=SEQUENCE." % value)
    seq = _fasta_seq(text)
    if cache_dir:
        open(os.path.join(cache_dir, value + ".fasta"), "w").write(text)
    return label, seq


def resolve_chain(spec, cache_dir=None, overrides=None, organism_id=3702):
    """Comma-separated IDs -> list of (label, sequence) for one chain-set side."""
    return [resolve_one(t.strip(), cache_dir, overrides, organism_id=organism_id)
            for t in spec.split(",") if t.strip()]
