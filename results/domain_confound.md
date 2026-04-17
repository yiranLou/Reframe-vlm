total test samples: 5712

## Table A: Domain × question_type × frame_type (n samples)

### SCANNET (n=2878)

- Camera perspective - Relative Direction: 1773 [camera=1773]
- Person perspective - Scene Simulation Relative Direction: 1105 [person=1105]

### COCO (n=2834)

- Camera perspective - Object View Orientation: 996 [camera=996]
- Person perspective - Object View Orientation: 996 [person=996]
- Person perspective - Relative Direction: 842 [person=842]

## Table B: Accuracy by domain × question_type (%)

| domain | question_type | n | ZS | Pr | Naive | TextInstr | Frame | FrameGated | TokenGated | Full |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coco | Camera perspective - Object View Orientation | 996 | 28.41 | 29.82 | 19.78 | 29.02 | 22.69 | — | — | 22.49 |
| coco | Person perspective - Object View Orientation | 996 | 40.46 | 45.68 | 64.56 | 72.09 | 74.00 | — | — | 68.98 |
| coco | Person perspective - Relative Direction | 842 | 35.63 | 35.15 | 36.34 | 35.87 | 42.87 | — | — | 45.13 |
| scannet | Camera perspective - Relative Direction | 1773 | 45.57 | 46.93 | 58.60 | 56.63 | 61.59 | — | — | 62.61 |
| scannet | Person perspective - Scene Simulation Relative Direction | 1105 | 26.06 | 27.69 | 47.69 | 53.39 | 54.03 | — | — | 57.19 |

## Table C: Frame LoRA − Naive LoRA per group (pp)

| domain | question_type | n | Δ | Naive | Frame |
|---|---|---:|---:|---:|---:|
| scannet | Camera perspective - Relative Direction | 1773 | +2.99 | 58.60 | 61.59 |
| scannet | Person perspective - Scene Simulation Relative Direction | 1105 | +6.33 | 47.69 | 54.03 |
| coco | Camera perspective - Object View Orientation | 996 | +2.91 | 19.78 | 22.69 |
| coco | Person perspective - Object View Orientation | 996 | +9.44 | 64.56 | 74.00 |
| coco | Person perspective - Relative Direction | 842 | +6.53 | 36.34 | 42.87 |

## Table D: text-instr SFT − Naive LoRA per group (pp)

| domain | question_type | n | Δ | Naive | TextInstr |
|---|---|---:|---:|---:|---:|
| scannet | Camera perspective - Relative Direction | 1773 | -1.97 | 58.60 | 56.63 |
| scannet | Person perspective - Scene Simulation Relative Direction | 1105 | +5.70 | 47.69 | 53.39 |
| coco | Camera perspective - Object View Orientation | 996 | +9.24 | 19.78 | 29.02 |
| coco | Person perspective - Object View Orientation | 996 | +7.53 | 64.56 | 72.09 |
| coco | Person perspective - Relative Direction | 842 | -0.48 | 36.34 | 35.87 |

## Table E: text-instr SFT − Frame LoRA per group (pp)

| domain | question_type | n | Δ | Frame | TextInstr |
|---|---|---:|---:|---:|---:|
| scannet | Camera perspective - Relative Direction | 1773 | -4.96 | 61.59 | 56.63 |
| scannet | Person perspective - Scene Simulation Relative Direction | 1105 | -0.63 | 54.03 | 53.39 |
| coco | Camera perspective - Object View Orientation | 996 | +6.33 | 22.69 | 29.02 |
| coco | Person perspective - Object View Orientation | 996 | -1.91 | 74.00 | 72.09 |
| coco | Person perspective - Relative Direction | 842 | -7.01 | 42.87 | 35.87 |
