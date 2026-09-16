# Hybrid-IMC

**Hybrid In-Memory Computing Design Based on Data Movement for Transformer Inference Acceleration**

> *Does in-memory computing actually transcend the Von Neumann bottleneck — or just move it?*

This repository is the simulation engine behind that question. It's a controlled, cross-technology benchmarking framework that puts four fundamentally different transformer-accelerator designs — a conventional digital baseline, ReRAM analog in-memory computing, FeFET compute-in-memory, and a hybrid SRAM+ReRAM architecture — through the *same* workloads, the *same* peripheral circuits, and the *same* accounting rules, so the differences you see are the technology, not a confound. It supports my PhD dissertation research at Prairie View A&M University and the methodology behind *"Transcending Von Neumann: A Comparative Analysis of In-Memory AI Accelerators for Energy-Efficient Transformer Model Inference"* (IEEE Open Journal of the Computer Society).

The short version of what it finds: in-memory computing genuinely eliminates the compute-array bottleneck it's built to solve — but the Von Neumann wall doesn't disappear, it just moves one level up, to DRAM. That's the kind of result you only get by holding everything else fixed and measuring carefully, which is exactly what this notebook is built to do.

**Want to see it for yourself?** Open the notebook in Colab, run Phase 1, and you'll have a real BERT model quantized and measured within the hour — no local GPU required. Details below.

---

## Motivation

Transformer-based models are increasingly bottlenecked by data movement rather than raw compute — the classic Von Neumann bottleneck. In-memory computing (IMC) addresses this by performing computation directly within or near memory, but different IMC implementations (analog, digital, hybrid) make very different tradeoffs in precision, area, and energy efficiency.

Published silicon results for these technologies are scattered across incompatible process nodes, precisions, and benchmark workloads — which makes it genuinely hard to tell whether one design beats another because the *technology* is better, or just because it happened to be measured under more favorable conditions. This project builds a **seven-layer validated benchmarking methodology** to settle that on a shared, controlled-variable basis, rather than relying on cross-paper comparisons that were never designed to be compared.

---

## The Four Configurations Compared

| Configuration | Compute Paradigm | Technology Node |
|---|---|---|
| **Baseline** | Digital CMOS systolic MAC (TPU v4 MXU-equivalent) | 7 nm |
| **Config A** | ReRAM Analog In-Memory Computing (AIMC), 1T1R HfO₂ | 22 nm |
| **Config B** | FeFET Compute-in-Memory (threshold-voltage-domain MAC) | 28 nm |
| **Config C** | Hybrid: SRAM CIM (attention) + ReRAM AIMC (feedforward) | 5 nm / 22 nm |

**Workloads:** BERT-Base-uncased (primary, fine-tuned on GLUE SST-2, sequence length 128), extended with BERT-Large, ViT-16, and ViT-32 to test whether the findings generalize across model families and modalities — plus GPT-2 and ViT-Base in a separate sensitivity/multi-workload block. See [The Lab Manual](#the-lab-manual-notebookshybrid_imc_crosstech_simulationipynb) below for exactly which workload runs where.

---

## The Seven-Layer Validation Methodology

Every configuration is evaluated against the same seven validation criteria to ensure a fair, controlled-variable comparison — this is the actual methodological contribution, as much as any single number the simulation produces:

1. **Shared baseline** — identical SRAM buffer hierarchy, ADC/DAC parameters, packaging overhead, and DRAM interface across all four configs
2. **Common simulation framework** — all configs modeled through the same NeuroSim V1.4-based analytical pipeline
3. **1b-normalized TOPS/W** — bit-normalized efficiency metric enabling fair comparison across different weight precisions
4. **Iso-accuracy targeting** — each configuration operates at the precision that keeps task accuracy within a stated tolerance of FP32, not an arbitrary fixed precision
5. **Array / chip / system level reporting** — results reported at all three levels of abstraction, not just idealized array-level numbers
6. **Technology-node normalization** — configurations additionally normalized to a common 7nm reference using IRDS 2022 scaling factors, so raw comparisons aren't confounded by process-node differences
7. **Full parameter disclosure** — every technology-intrinsic parameter (Ron/Roff ratios, Vth window, endurance, etc.) is disclosed and individually sourced to a named, peer-reviewed silicon measurement — not assumed

---

## Key Findings

Based on the full-system simulation results in this repository:

- **DRAM access dominates system-level energy** across all configurations — roughly 40–55% of total system energy depending on configuration and workload, confirming data movement, not compute, as the primary bottleneck that survives even after in-memory computing does its job
- **At the chip level, Config C (Hybrid) and Config A (ReRAM AIMC) are close competitors, not a clear runaway win for either** — the ranking depends on which level of the stack you're measuring at, which is exactly why this repository reports all three (array, chip, system) rather than just the flattering one
- **Config B (FeFET CIM)** trades efficiency for higher I/O overhead due to its lower native throughput requiring more I/O cycles per inference
- The **TOMAS operation taxonomy** shows ~97% of BERT-Base's MAC operations (QKV projections, output projections, feedforward layers) map cleanly to static-weight, AIMC-friendly execution, while the remainder (attention scores and attention×value operations) require dynamic SRAM CIM execution — this asymmetry is the empirical basis for the hybrid architecture's design, and the same static/dynamic split that several independent prior accelerator designs converged on without ever formalizing it generally

Full per-configuration results, including energy/area/latency breakdowns and sensitivity analysis (ADC resolution, DRAM technology, subarray size sweeps), are in `results/`.

---

## Repository Structure

```
Hybrid-IMC/
├── notebooks/
│   └── Hybrid_IMC_CrossTech_Simulation.ipynb   # The full simulation pipeline (Phases 1-5)
├── results/
│   ├── data/                       # Raw JSON results (architecture, ops mapping, per-config metrics)
│   ├── figures/                    # Publication figures (accuracy/bitwidth, system dashboard, TOMAS taxonomy)
│   ├── tables/                     # LaTeX tables ready for paper insertion
│   └── sensitivity/                # ADC, DRAM, and subarray-size sensitivity sweep results
└── README.md
```

---

## The Lab Manual (`notebooks/Hybrid_IMC_CrossTech_Simulation.ipynb`)

This is the actual simulation — every number in the paper traces back to a cell in this notebook. It's organized into five phases, numbered sequentially (Steps 1.0 through 13.5) so you always know where you are and what depends on what. Designed to run top-to-bottom in Google Colab with no local GPU required.

**Phase 1 — Workload Preparation & Validation.** Downloads four models (BERT-Base, BERT-Large, ViT-16, ViT-32) from HuggingFace, extracts each one's real architecture parameters, maps BERT-Base's operations to their TOMAS execution category (AIMC vs. SRAM CIM vs. digital peripheral), runs FP32 baseline inference on SST-2, and quantization-sweeps every model from 2-bit through 8-bit to establish the accuracy-vs-precision curve each hardware configuration's target precision is chosen from. This is the only phase that touches real model weights or real accuracy numbers — everything downstream is hardware simulation built on top of what this phase measures.

**Phase 2 — Baseline + Config A (chip → system).** Estimates array-level and chip-level area, latency, and energy for the digital baseline and the ReRAM AIMC configuration, using device parameters calibrated against peer-reviewed silicon (ISSCC, IEDM, Nature Communications), then integrates each with the shared memory hierarchy (CACTI-modeled SRAM buffers, HBM2E/HBM3E DRAM) to produce full-system numbers. Includes a self-validating generalized pipeline that extends Config A's simulation to BERT-Large, ViT-16, and ViT-32 automatically.

**Phase 3 — Config B + Config C (chip → system).** Same treatment for the FeFET CIM and hybrid SRAM+ReRAM configurations, plus an array-level energy breakdown for both — the same component-level granularity (device read, ADC, DAC, peripheral, I/O) that Config A's own cells already expose.

**Phase 4 — Cross-Config Analysis, Validation & Publication Outputs.** Consolidates all four configurations into unified comparison tables and figures, validates results against published silicon data points (IBM, Mythic, TSMC, and other test-chip publications), and generates the LaTeX tables and figures used directly in the paper.

**Phase 5 — Sensitivity Analysis & Multi-Workload Validation.** Tests how sensitive the system-level results are to subarray size, ADC resolution, and DRAM bandwidth, and separately validates the paper's findings against a second, independent workload set (BERT-Large, GPT-2, and ViT-Base) — pulling BERT-Large and ViT-Base from Phase 1's real downloaded data where the two workload sets overlap, rather than a second set of assumed constants, and flagging automatically if the two ever disagree.

**Getting oriented fast:** every cell's docstring starts with its phase-relative step number (e.g., `STEP 9.1`), so you can always tell where a given cell sits in the dependency chain just by reading its header — no need to count cells or guess execution order.

---

## Data Provenance

Every technology parameter used in this framework (ReRAM Ron/Roff ratios, FeFET threshold voltage window, SRAM bitcell area, ADC energy-per-conversion, HBM bandwidth/energy) is sourced from a specific peer-reviewed publication — primarily ISSCC, IEDM, VLSI Symposium, and Nature Communications papers from 2020–2024. This is a deliberate design choice: rather than using idealized or vendor-marketing numbers, every parameter is traceable to a measured silicon result. See `results/data/master_config.json` for the full parameter set and inline source citations.

---

## Related Publication

D. N. Okeke, S. Musa, J. Foreman, C. M. Akujuobi, I. Nzekwe, and S. Cui, **"Transcending Von Neumann: A Comparative Analysis of In-Memory AI Accelerators for Energy-Efficient Transformer Model Inference,"** *IEEE Open Journal of the Computer Society*, 2026.

---

## Getting Started

The pipeline is designed to run in Google Colab — no local GPU required (see the notebook's own setup cells for the exact package/environment steps):

```bash
# Clone the repository
git clone https://github.com/DominicOkeke/Hybrid-IMC.git
cd Hybrid-IMC

# Open notebooks/Hybrid_IMC_CrossTech_Simulation.ipynb in Google Colab
# Run Cell 0 (Master Setup) first, then follow the phases in order --
# each cell's docstring tells you exactly what it needs and produces.
```

**Note on large files:** raw model weight checkpoints (BERT-Base, BERT-Large, ViT-16, ViT-32) are not included in this repository, since they're trivially reproducible via the `transformers` library and exceed GitHub's practical file size guidance. The notebook downloads each one automatically in Phase 1.

---

## Author

**Dominic Nze Okeke**
PhD Candidate, Electrical Engineering — Prairie View A&M University
Signal Integrity & Customer Enabling Engineer Intern, Intel Corporation (2024–2026)
ORCID: [0009-0007-6335-2628](https://orcid.org/0009-0007-6335-2628) · [LinkedIn](https://www.linkedin.com/in/dominic-okeke-601b8930/) · dnokeke@gmail.com

---

## License

This project is licensed under the BSD 3-Clause License — see the [LICENSE](LICENSE) file for details.
