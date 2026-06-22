# M2.31 Ablation Notes

Date: 2026-06-22

Status update: this document records the M2.31/M2.32 negative ablations before M2.34. M2.34 fill-inset candidates later superseded M2.30 as the current best; see `docs/m2_34_fill_inset_notes.md`.

This ablation tested whether M2.30 could be improved by selector retuning or safer fill-mask generation. None of the tested variants is promoted over M2.30.

## Baseline

| Variant | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| M2.30 public LOO | 0 | 0.068519 | 6.636364 | 1.848485 | 0.105806 | 0.911108 | 0.798703 | Keep current best |

## Selector Ablations

| Variant | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| precision-small LOO | 0 | 0.065631 | 6.333333 | 1.787879 | 0.105806 | 0.888958 | 0.802090 | Reject: lower loss is partly caused by low-coverage flat-icon choices |
| min-coverage guard 0.20 LOO | 0 | 0.062873 | 6.060606 | 1.515152 | 0.155833 | 0.899002 | 0.794234 | Reject: still admits low-coverage OpenMoji candidates around 0.27 coverage |
| min-coverage guard 0.30 LOO | 0 | 0.068720 | 6.454545 | 2.060606 | 0.126533 | 0.918077 | 0.800934 | Not promoted: coverage improves slightly, but loss/off-mask/trim worsen |
| min-coverage guard 0.30 core-to-full | 0 | 0.078392 | 7.312500 | 2.625000 | 0.092906 | 0.884218 | 0.813718 | Reject: worse than M2.30 core-to-full loss 0.074071 and coverage 0.927512 |
| min-coverage guard 0.30 incoming LOO | 0 | 0.084419 | 6.750000 | 4.750000 | 0.000000 | 0.856344 | 0.793664 | Tied with M2.30 incoming result |

Key finding: a lower mean loss alone is not enough. Some selector configurations choose low-loss skeleton candidates with incomplete flat-icon coverage, for example OpenMoji sun and lady beetle variants around 0.27 coverage. These are not acceptable for the practical image-to-DST goal.

## Fill-Mask Generation Ablations

| Variant | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| eval-mask-as-fill nearest-row p40 | 0 | 0.113428 | 7.363636 | 0.969697 | 2.061933 | 0.999997 | 0.646435 | Reject: precision and off-mask degrade badly |
| strict short nearest-row, no mask path | 5 | 0.158351 | 19.000000 | 1.363636 | 0.128455 | 0.999654 | 0.763392 | Reject: jump and hard_fail explode |
| strict nearest-row p20 | 0 | 0.101801 | 10.545455 | 1.363636 | 0.263558 | 0.999686 | 0.755671 | Reject: too many jumps |
| strict nearest-row p40 | 0 | 0.077217 | 7.515152 | 0.909091 | 0.412527 | 0.999689 | 0.749214 | Reject: not better than M2.30 selector |
| quantized segment validation p40 | 0 | 0.087732 | 9.060606 | 0.969697 | 0.294418 | 0.999686 | 0.751009 | Reject: lowers off-mask vs strict p40, but increases jump and loss |

## Selector Ablation With Quantized Candidate

| Variant | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| M2.30 public LOO | 0 | 0.068519 | 6.636364 | 1.848485 | 0.105806 | 0.911108 | 0.798703 | Keep current best |
| M2.32 + quantized validation candidate LOO | 0 | 0.069334 | 6.818182 | 1.848485 | 0.075076 | 0.911108 | 0.798854 | Reject: off-mask improves, but loss and jump worsen |

The quantized-validation candidate was selected on only 2 of 33 leave-one-out samples. It is useful as a diagnostic candidate because it reduces boundary spill after DST 0.1mm quantization, but as a hard gate it sacrifices too much local continuity.

## Conclusion

At the time of this ablation, M2.30 remained the current best model. The nearest-endpoint row-order candidate was still the strongest useful addition, but simply tightening coverage floors, using the evaluation mask as the fill source, or hard-validating quantized segments did not produce a better practical planner.

Next useful direction: create a genuinely better high-coverage candidate that reduces off-mask without sacrificing path continuity. Quantized boundary risk should become a soft selector feature or repair veto, not a hard routing gate, because the hard gate reduces spill but increases jump.
