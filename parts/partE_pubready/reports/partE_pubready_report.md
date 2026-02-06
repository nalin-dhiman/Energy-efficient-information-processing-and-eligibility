# Part E (Plasticity) — Publication-ready figure set

This folder contains a hardened, Nat Comms–style figure set for Part E.

## What these figures show

- **FigE1** (Patch generalization): Which plasticity rule best matches the connectome geometry across spatially diverse patches, and how consistent that win is.
- **FigE2** (Dual baseline): The alignment ranking is stable whether the target geometry is defined globally or per patch.
- **FigE3** (Cost model robustness): The winner does not depend on the specific wiring-cost exponent/threshold.
- **FigE4** (Compute fairness): Under *matched forward-pass budgets*, eligibility-trace learning (EProp) achieves tighter alignment than stochastic perturbation gradients (REINFORCE).
- **FigE5** (Scale robustness): The ordering remains stable from 500→2000 neurons.
- **FigE6** (Failure handling): Sensitivity analysis for N=2000 demonstrates conclusions are invariant to reasonable treatments of occasional instability.

## Notes

- The script intentionally places legends outside plot areas to avoid covering data.
- Outputs are written in both **PNG (600 dpi)** and **PDF (vector)**.

