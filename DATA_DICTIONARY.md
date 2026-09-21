# Data dictionary

All paths below are immutable raw inputs covered by `data/checksums.sha256`.
Semantic roles and exact filenames are assigned in `config/artifact.json`.

## MZI acquisitions

Each `data/raw/mzi/<timestamp>/<destination>/` directory contains:

- `plan.json` — scan plan, voltage sequence, point duration, and acquisition
  settings.
- `points.jsonl` — one JSON object per acquisition point. Records contain the
  piezo voltage, exposure timing, singles counts, coincidence counts, and
  acquisition metadata.
- `dark.json` — associated dark-run counts and duration.
- `meta.json` — acquisition identity, destination, device state, and scan
  settings.

Every configured dataset assigns one timestamp as the launch run and one as
the preserve/control run. Each timestamp has `lab/` and `launch/` destinations.

The labels used in the laboratory, and therefore the raw
acquisition field names, used "signal" and "idler" opposite to the standard
convention adopted in the paper. In particular, the raw `signal_destination`
field records the destination of the paper's idler. This is a naming
translation only. The resulting
hardware-to-paper channel mapping is:

| JSON field | Hardware channel | Paper quantity |
|---|---|---|
| `N_i` | S1 | signal detector 1 singles |
| `N_i2` | S2 | signal detector 2 singles |
| `N_s` | I | detected-idler singles |
| `N_c` | C1 | signal 1–idler coincidences |
| `N_c2` | C2 | signal 2–idler coincidences |

Coincidences use the full effective 25 ns window. The analysis derives
`detected_idler` from the configured semantic condition: it is false
for Launch and true for Erase and Preserve. It is constructed as run
metadata by `dataset_runs()` in `analysis/artifact_config.py`.

## Environmental archives

Environmental inputs are grouped under `data/raw/conditions/<launch-timestamp>/`.

- `metar/*.json` — original KSBA aviation-weather response.
- `aeronet/*.csv` — original AERONET solar and lunar extracts.
- `goes/*.nc` — original GOES-18 AOD, COD, and TPW NetCDF records. Each
  conditions profile explicitly lists the records assigned to every product.
- `irsa/*.ecsv` — original IRSA responses, including provider metadata. The
  per-sample ECSV tables cover the integration interval.
- `irsa/*.csv` — derived, metadata-stripped conversions of the ECSV responses.
- `irsa/*.json` — project-generated retrieval and integration manifests
  containing computed transmissions and selected-sample summaries.

These archived source and retrieval records are never rewritten during replay.
