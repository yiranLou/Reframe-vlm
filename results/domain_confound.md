
total test samples: 5712

=== Table A: Domain × question_type × frame_type (n samples) ===

  SCANNET (n=2878)
    Camera perspective - Relative Direction                  1773   [camera=1773]
    Person perspective - Scene Simulation Relative Direction  1105   [person=1105]

  COCO (n=2834)
    Camera perspective - Object View Orientation              996   [camera=996]
    Person perspective - Object View Orientation              996   [person=996]
    Person perspective - Relative Direction                   842   [person=842]

=== Table B: Accuracy by domain × question_type (%) ===
| domain | question_type | n | ZS | Pr | Naive | TextInstr | Frame | Full |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| coco | Camera perspective - Object View Orientation | 996 | 28.41 | 29.82 | 19.78 | 29.02 | 21.99 | 21.59 |
| coco | Person perspective - Object View Orientation | 996 | 40.46 | 45.68 | 64.56 | 72.09 | 68.37 | 61.95 |
| coco | Person perspective - Relative Direction | 842 | 35.63 | 35.15 | 36.34 | 35.87 | 38.84 | 39.55 |
| scannet | Camera perspective - Relative Direction | 1773 | 45.57 | 46.93 | 58.60 | 56.63 | 59.84 | 59.73 |
| scannet | Person perspective - Scene Simulation Relative Direction | 1105 | 26.06 | 27.69 | 47.69 | 53.39 | 46.88 | 47.33 |

=== Table C: Frame LoRA − Naive LoRA per group (pp) ===
| domain | question_type | n | Δ | Naive | Frame |
|---|---|---:|---:|---:|---:|
| scannet | Camera perspective - Relative Direction | 1773 | +1.24 | 58.60 | 59.84 |
| scannet | Person perspective - Scene Simulation Relative Direction | 1105 | -0.81 | 47.69 | 46.88 |
| coco | Camera perspective - Object View Orientation | 996 | +2.21 | 19.78 | 21.99 |
| coco | Person perspective - Object View Orientation | 996 | +3.82 | 64.56 | 68.37 |
| coco | Person perspective - Relative Direction | 842 | +2.49 | 36.34 | 38.84 |

=== Table D: text-instr SFT − Naive LoRA per group (pp) ===
| domain | question_type | n | Δ | Naive | TextInstr |
|---|---|---:|---:|---:|---:|
| scannet | Camera perspective - Relative Direction | 1773 | -1.97 | 58.60 | 56.63 |
| scannet | Person perspective - Scene Simulation Relative Direction | 1105 | +5.70 | 47.69 | 53.39 |
| coco | Camera perspective - Object View Orientation | 996 | +9.24 | 19.78 | 29.02 |
| coco | Person perspective - Object View Orientation | 996 | +7.53 | 64.56 | 72.09 |
| coco | Person perspective - Relative Direction | 842 | -0.48 | 36.34 | 35.87 |

=== Table E: text-instr SFT − Frame LoRA per group (pp) ===
| domain | question_type | n | Δ | Frame | TextInstr |
|---|---|---:|---:|---:|---:|
| scannet | Camera perspective - Relative Direction | 1773 | -3.21 | 59.84 | 56.63 |
| scannet | Person perspective - Scene Simulation Relative Direction | 1105 | +6.52 | 46.88 | 53.39 |
| coco | Camera perspective - Object View Orientation | 996 | +7.03 | 21.99 | 29.02 |
| coco | Person perspective - Object View Orientation | 996 | +3.71 | 68.37 | 72.09 |
| coco | Person perspective - Relative Direction | 842 | -2.97 | 38.84 | 35.87 |
