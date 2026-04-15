
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
| domain | question_type | n | ZS | Pr | Naive | Frame | Full |
|---|---|---:|---:|---:|---:|---:|---:|
| coco | Camera perspective - Object View Orientation | 996 | 28.41 | 29.82 | 19.78 | 21.99 | 21.59 |
| coco | Person perspective - Object View Orientation | 996 | 40.46 | 45.68 | 64.56 | 68.37 | 61.95 |
| coco | Person perspective - Relative Direction | 842 | 35.63 | 35.15 | 36.34 | 38.84 | 39.55 |
| scannet | Camera perspective - Relative Direction | 1773 | 45.57 | 46.93 | 58.60 | 59.84 | 59.73 |
| scannet | Person perspective - Scene Simulation Relative Direction | 1105 | 26.06 | 27.69 | 47.69 | 46.88 | 47.33 |

=== Table C: Frame LoRA − Naive LoRA per group (pp) ===
| domain | question_type | n | Δ | Naive | Frame |
|---|---|---:|---:|---:|---:|
| scannet | Camera perspective - Relative Direction | 1773 | +1.24 | 58.60 | 59.84 |
| scannet | Person perspective - Scene Simulation Relative Direction | 1105 | -0.81 | 47.69 | 46.88 |
| coco | Camera perspective - Object View Orientation | 996 | +2.21 | 19.78 | 21.99 |
| coco | Person perspective - Object View Orientation | 996 | +3.82 | 64.56 | 68.37 |
| coco | Person perspective - Relative Direction | 842 | +2.49 | 36.34 | 38.84 |
