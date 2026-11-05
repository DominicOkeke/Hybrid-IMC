# Hybrid-IMC

**Simulation code and data for "TOMAS: Operation-Aware Hybrid of In-Memory AI Accelerators for Energy-Efficient Transformer Inference"**

> *Which transformer operations belong in analog in-memory arrays, which in digital compute-in-memory, and when does the split pay?*

This repository is a controlled, cross-technology benchmark that runs four transformer-accelerator designs (a digital baseline, ReRAM analog in-memory computing, FeFET compute-in-memory and a hybrid ReRAM + SRAM architecture) through the *same* workload, the *same* peripheral circuits and the *same* accounting rules, so the differences that remain are the technology. On top of it sit two methods: **TOMAS**, which partitions transformer matrix-vector operations by weight stationarity and device-resolvable precision, and **TOMAS-Map**, which derives the rest of the hybrid architecture (subarray dimensions, input slicing, converter provisioning, engine balance and schedule) from the model's shape, the device limits and an area budget. Seven hypotheses are tested, each by changing one factor with everything else fixed.

## The four configurations

| Configuration | Static-weight engine | Dynamic-operand engine ($QK^\top$, $A\cdot V$) |
|---|---|---|
| **Baseline** | 7 nm digital systolic array (one TPU v4 MXU) | same |
| **Config A** | 22 nm HfO2 1T1R ReRAM AIMC, 4-bit cells | 7 nm digital logic |
| **Config B** | 28 nm FeFET CIM, 3-bit cells (fails Layer 4) | 7 nm digital logic |
| **Config C** | ReRAM AIMC identical to Config A | 5 nm SRAM CIM |

Configs A and C differ only in the dynamic-operand engine, which makes their comparison a single-variable test of the partition.

## The seven-layer validation

1. **Shared periphery**: identical buffers, converters, DAC, interconnect and DRAM.
2. **One toolchain**: one array, digital and system model for every design.
3. **Bit normalization**: TOPS/W and 1-bit-normalized TOPS/W reported.
4. **Iso-accuracy**: at least 90.0% on GLUE SST-2 (FP32 reference 92.43%).
5. **Level separation**: array, chip and system results never mixed.
6. **Node normalization**: peripherals scaled to 7 nm; memory cells native.
7. **Full disclosure**: every parameter listed with its source (`PARAMETER_PROVENANCE.csv`).

## Key findings (BERT-Base, 128 tokens, unless stated)

- TOMAS places **97.3%** of the multiply-accumulates on the analog array; the dynamic products $QK^\top$ and $A\cdot V$ are the remaining 2.7%.
- Mapped at equal logic area, the hybrid reaches **55.31 TOPS/W** at chip level: **1.27x** standalone ReRAM AIMC, **11.1x** the 8-bit digital baseline and **3.0x** a 4-bit one. Its lead over AIMC grows to **3.87x** at 2,048 tokens.
- In GPT-2 decoding, holding the key/value cache in the hybrid's writable SRAM engine gives the lowest energy per token, **2.31 uJ**, at 13.38 us per token: **5.9x** faster than the 79.4 us an A100 needs at batch one just to stream the 8-bit weights at 1.56 TB/s (a computed lower bound, not a GPU measurement).
- At system level with weights resident, the hybrid reaches **30.07 TOPS/W**, **13.6x** the 8-bit digital baseline that streams its weights. DRAM is still **34.6-39.6%** of system energy (55.4% for the weight-streaming digital baseline); with weights reloaded every inference it is 84-87%.
- Converters, not cells, set an analog engine's throughput and area, so standalone AIMC leads in throughput per area.
- Lossless conversion is not needed: 8-bit conversion of the 4-bit cell sums costs only 0.35 points of SST-2 accuracy (90.71%, above the 90% floor). Under analog non-idealities (STEP 5.0b, 5 seeds), the disclosed programming noise (sigma 15-25%) and 5% uncompensated IR drop lower SST-2 accuracy to **89.63%** (min 88.88%), just below the 90% floor. Configs A and C share this engine, so their comparison is unaffected; write-verify programming, IR-drop compensation or hardware-aware training would be required and are not modeled.

## Repository structure

```
Hybrid-IMC/
├── notebooks/Hybrid_IMC_CrossTech_Simulation.ipynb   # the full pipeline (Google Colab)
├── results/                       # every output of the reported run (JSON)
│   ├── shared/master_config.json  # every shared and device parameter (Layer 1)
│   ├── results/                   # Phase 1: model shapes, operation inventory, SST-2 accuracy,
│   │                              #   accuracy under analog non-idealities (STEP 5.0b)
│   ├── baseline_digital/ config_a_reram/ config_b_fefet/ config_c_heterogeneous/
│   ├── cross_config_comparison/   # the four designs at declared dimensions
│   ├── phase4_analysis/           # manuscript_numbers.json (every reported number),
│   │                              #   silicon validation, model sensitivity, I/O check
│   ├── phase6_tomas_map/          # TOMAS-Map ladder, context sweep, decoding, macro validation
│   └── sensitivity/ multiworkload/
├── MANUAL.md                      # how to run, step by step, and how to check the results
├── PARAMETER_PROVENANCE.csv       # every parameter -> value -> source
├── requirements.txt               # package versions of the reported run
├── CITATION.cff                   # how to cite this repository
└── LICENSE                        # BSD-3-Clause
```

## Release notes

- **v2.1** (2026-11-04): designs that do not hold their weights now stream the weights their compute model uses plus the embedding rows the input reads (85.1 M for BERT-Base), instead of every parameter including the full embedding tables (109.5 M). This raises the weight-streaming system figures (8-bit digital baseline: 1.92 to 2.21 TOPS/W) and leaves every chip-level and weight-resident result unchanged. Set `STREAM_ALL_PARAMETERS = True` in STEP 7.5 to reproduce v2.0. A leftover workflow note was removed from the sensitivity cell. The macro validation (STEP 14.2) now uses the configurations tabulated in Extended Data Table 1 of Wan et al. (Nature 2022) for Hung et al. (4 rows, 64 columns) and adds Xue et al. (ISSCC 2019; Nat. Electron. 2021); the earlier rows, which assumed the row count and precision, are kept and marked superseded. Two optional GPU steps were added at the end of the notebook: GPU STEP A1 measures SST-2 accuracy with modeled write-verify programming, IR-drop compensation and 64-row subarrays, and GPU STEP A2 repeats Layer 4 on MRPC, RTE, QNLI and CoLA. Both need a CUDA GPU and are skipped elsewhere.
- **v2.0** (2026-10-31): engine v2, TOMAS-Map, analog accuracy check, updated documentation.

## Reproducing the results

See `MANUAL.md`. In short: open the notebook in Google Colab and run it top to bottom. The final cell compares 52 reported values with the run and stops with an error if any differs by more than one part in 10^6. A simulator-only path (no GPU, no downloads) runs Phases 2-6 in a few minutes from the recorded Phase 1 results and reproduces all 52 values.

## Citation

D. N. Okeke, S. M. Musa, J. Foreman, C. M. Akujuobi, I. Nzekwe and S. Cui, "TOMAS: Operation-Aware Hybrid of In-Memory AI Accelerators for Energy-Efficient Transformer Inference," manuscript, 2026. Code and data: https://github.com/DominicOkeke/Hybrid-IMC (release v2.1). See also `CITATION.cff`.

## License

BSD-3-Clause. See `LICENSE`.
