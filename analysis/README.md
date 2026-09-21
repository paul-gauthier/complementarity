# Analysis code

The manuscript is the authoritative description of the scientific methods,
models, and statistical interpretation. This directory is its executable
companion: it contains the implementation needed to trace reported results
back to archived inputs.

The canonical end-to-end workflow is
[`scripts/reproduce.py`](../scripts/reproduce.py), driven by the dataset roles,
analysis settings, environmental profiles, and fixed random seed in
[`config/artifact.json`](../config/artifact.json). Individual modules can be
useful for inspection, but the reproduction workflow records how they are
composed and which outputs are canonical. See the [repository
README](../README.md) for the supported reproduction and verification commands.

### Transmission

[`conditions_sources.py`](conditions_sources.py) stages the explicitly configured
environmental archives. [`goes.py`](goes.py) loads one configured scan per GOES
while [`conditions.py`](conditions.py) coordinates the complete archived replay.
[`run_libradtran_transmission.py`](run_libradtran_transmission.py) computes the
atmospheric transmission, while [`igm_transmission.py`](igm_transmission.py)
and [`transmission_values.py`](transmission_values.py) provide the future
intergalactic model and assemble the finite-path and infinite-future survival
factors consumed by the fits.

### Methods

[`mzi_io.py`](mzi_io.py), [`dark_analysis.py`](dark_analysis.py), and
[`darks_accidentals.py`](darks_accidentals.py) cover validation and alignment
of MZI records, conversion of counts to corrected rates, and the dark-background
and accidental-coincidence accounting used downstream.

### Fringe fits

[`mzi_analysis.py`](mzi_analysis.py) and [`mzi_joint.py`](mzi_joint.py) contain
the scan-level likelihood and weighted fits, simultaneous fits of the two MZI
outputs, covariance propagation, and fit products used in figures and tables.

### Paired launch–erase contrast fit

[`mzi_null_bounds.py`](mzi_null_bounds.py) constructs the paired launch and
erase contrasts, estimates the launch-only quadratures and covariance, and
produces dataset-level confidence bounds under the configured normalizations.

### Combined analysis

[`mzi_pooled_analysis.py`](mzi_pooled_analysis.py) implements cross-dataset
inference for a common restoration fraction, profiles dataset-specific phases,
and runs the null-calibration, coverage, and goodness-of-fit simulations.

### Figures

The principal renderers are
[`mzi_plot_alternating_by_pass.py`](mzi_plot_alternating_by_pass.py),
[`mzi_null_plot.py`](mzi_null_plot.py), and
[`mzi_normalized_quadrature_plot.py`](mzi_normalized_quadrature_plot.py). They
expose, respectively, the scan-by-scan fit structure, dataset-level bounds, and
normalized cross-dataset confidence regions.

### Supporting code and outputs

[`artifact_config.py`](artifact_config.py) loads and validates the authoritative
configuration, resolves semantic dataset roles, and defines the required
canonical products. 
Modules named `generate_*`
translate machine-readable analysis records into captionless TeX tables,
semantic manuscript macros, dataset summaries, and browsable result indexes.

### Related documentation

The raw field and channel definitions are in the [data
dictionary](../DATA_DICTIONARY.md). The [reviewed-results guide](../results/)
is the most direct route from a generated figure or table back to its
machine-readable record.
Focused validation lives in [`tests/`](../tests/), and
[`scripts/verify_results.py`](../scripts/verify_results.py) performs the
end-to-end comparison of a local reproduction with the reviewed products.

### Data flow

[`data/raw/`](../data/raw/) + [`config/artifact.json`](../config/artifact.json)
&rarr; per-dataset conditions and fits &rarr; dataset analysis records &rarr;
pooled analysis &rarr; local `build/` products &rarr; verification against
reviewed [`results/`](../results/).
