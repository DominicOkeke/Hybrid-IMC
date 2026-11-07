# User Manual — Hybrid IMC Cross-Technology Simulation (TOMAS / TOMAS-Map)

Controlled-variable simulation behind *TOMAS: Operation-Aware Hybrid of In-Memory AI Accelerators for Energy-Efficient Transformer Inference*.
It evaluates a 7 nm digital baseline, 22 nm ReRAM AIMC (Config A),
28 nm FeFET CIM (Config B) and the hybrid ReRAM AIMC + 5 nm SRAM CIM (Config C).
TOMAS partitions the static/dynamic MVM operations; TOMAS-Map derives the architecture,
converter provisioning and schedule. Every converter the timing model uses is counted in area.

## How to reproduce

**Platform.** Google Colab, with Google Drive mounted. All results are written to
`/content/drive/MyDrive/Hybrid_IMC_CrossTech_Sim/`. A GPU runtime (T4 or better) is
recommended for Phase 1 only.

**Order.** Run the cells top to bottom, with one interruption:

1. **MASTER SETUP**: writes `shared/master_config.json` (every shared parameter, Layer 1).
2. **Package install**: installs Hugging Face libraries and **restarts the runtime on purpose**.
   After the restart, continue from **STEP 1.0** and do not re-run the install cell.
3. **Phase 1 (STEP 1.0 – 5.3, FINAL)**: downloads BERT-Base, BERT-Large and ViT, measures FP32
   and 2–8-bit SST-2 accuracy, and enumerates the operations TOMAS partitions.
4. **Phase 1 → Phase 2 hand-off**: checks the four Phase 1 files the simulator needs.
5. **STEP 7.5 onward (Phases 2–6)**: the unified engine, all four configurations, the seven-layer
   validation, TOMAS-Map, context sweep, decoding, silicon validation, sensitivity, the
   paper's numbers and Phase 5. This part needs no GPU and runs in a few minutes.
6. **REPRODUCIBILITY CHECK**: compares 52 figures quoted in the paper with
   this run and stops with an error if any differs by more than 1 part in 10⁶.

**Simulator-only reproduction (no GPU, no downloads).** Run MASTER SETUP, skip the package
install and Phase 1, set `USE_RECORDED_PHASE1 = True` in the hand-off cell, then run from there
to the end. The Phase 1 files are then written from the authors' recorded Phase 1 run. In that
mode the multi-workload ViT row uses its built-in architecture estimate.

**Determinism.** Phases 2–6 contain no random numbers. The one fitted parameter, the SRAM
cell energy, is solved in closed form from Dong et al. (351 TOPS/W) and checked in **STEP 9.0b**.
STEP 9.0b also checks that the earlier engine still reproduces the original Config A
result. Phase 1 accuracies can move by a few hundredths of a percent across library versions;
the simulator reads them only through the 90.0% Layer 4 floor.

**Packages.** `requirements.txt` lists the versions of the reported run. Phases 2–6 need only
NumPy and Matplotlib; Phase 1, including STEP 5.0b (accuracy under analog non-idealities, about
30 minutes on a T4), needs PyTorch, Transformers and Datasets and a GPU.

**Checking a run against the reported one.** Every output lands under
`Hybrid_IMC_CrossTech_Sim/` with the same layout as `results/` in this repository, so any file
can be compared directly; `results/phase4_analysis/paper_numbers.json` holds every reported
number.

**Key outputs.**
- `phase4_analysis/paper_numbers.json`: every number the paper quotes.
- `phase6_tomas_map/`: `ladder.json`, `decode.json`, `context_sweep.json` and `validation.json`.
- `phase4_analysis/model_sensitivity.json`: the model-assumption sensitivity cases.
- `sensitivity/`, `multiworkload/`: Phase 5.
- `results/accuracy/analog_nonideal_accuracy.json`: SST-2 accuracy under programming noise, IR drop and partial-sum conversion (STEP 5.0b).
- `results/accuracy/mitigation_accuracy.json`, `glue_tasks_accuracy.json`, `glue_precision_diagnostic.json`: the optional GPU steps A1-A3.
- `phase4_analysis/table_*.csv`: the summary tables of STEP 13.5.
- `phase4_analysis/io_and_accuracy_checks.json`: interface-energy break-even and the TPU v4 comparison.
