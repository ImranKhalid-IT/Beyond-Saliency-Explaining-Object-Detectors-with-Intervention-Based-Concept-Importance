# RCE Shared-Detector Baseline Benchmark

A reproducible benchmark for comparing **Relational Concept Explainer (RCE)** outputs with representative XAI baselines on the **same frozen object detector, the same image, and the same selected object pair**.

The repository is designed for qualitative comparison and research reproducibility. It does not manually sharpen, recolor, or edit one method differently from another. Every supplied heatmap is resized, normalized, and visualized through one shared pipeline.

---

## 1. What this repository does

The current implementation uses a frozen **torchvision Faster R-CNN ResNet-50 FPN v2** detector and explains a selected `person--bicycle` pair.

The pipeline:

1. Loads an input image.
2. Detects the target person and bicycle, or accepts manually supplied boxes.
3. Keeps the detector parameters frozen.
4. Generates pair-level maps for:
   - D-RISE;
   - ODAM-style fixed-ROI gradients;
   - detector-adapted Finer-CAM.
5. Loads externally generated maps for:
   - RCE;
   - official CRAFT output.
6. Applies the same resizing, normalization, opacity, and figure layout to all methods.
7. Saves publication-ready PDF and PNG figures, raw NumPy maps, and a metadata JSON file.

> **Important:** This benchmark package does not silently replace official CRAFT or the full RCE pipeline with simplified approximations. RCE and CRAFT maps are provided as inputs through `--rce-map` and `--craft-map`.

---

## 2. Why one shared detector is important

A visual comparison is not fair when different methods use different detectors, target objects, image sizes, or normalization rules. This repository controls those variables by using:

- one frozen Faster R-CNN detector;
- one fixed target pair;
- one target feature layer for gradient-based methods;
- one image resolution;
- one heatmap normalization function;
- one overlay opacity;
- one output layout.

The detector is used only for inference and explanation. Its weights are not retrained or fine-tuned.

---

## 3. Pair-level explanation target

For the selected object pair, the shared relational score is based on the confidence of both target detections.

For fixed-ROI gradient methods, the pair probability is computed as:

```text
pair_score = probability(person) × probability(bicycle)
```

For D-RISE, the two instance-similarity values are multiplied for each random mask. A mask receives a high weight only when it preserves evidence for both selected objects.

This makes the comparison pair-oriented rather than producing two unrelated single-object explanations.

---

## 4. Important note about K-Means concept IDs

RCE uses unsupervised clustering to discover image-specific latent concepts. K-Means cluster numbers are **permutation-invariant**. Therefore, labels such as `Concept 1`, `Concept 2`, or `Concept 3` are not permanent semantic identities.

For example, two runs may produce:

```text
Run 1: Concept 1 = pedal region
Run 2: Concept 4 = pedal region
```

This label change does not necessarily mean that the explanation has changed. The underlying spatial concept and its interventional importance may remain equivalent even when the numeric cluster ID changes.

### Correct interpretation

Concept stability should be checked using:

- spatial overlap of activation maps;
- dominant activated region;
- concept-effect score;
- ranking consistency;
- deletion/AOC behavior;
- similarity after optimal cluster matching.

Concept stability should **not** be judged only by comparing raw cluster numbers.

### Across different images

A concept index is local to one image and one selected object pair. `Concept 1` in one image is not assumed to represent the same visual evidence as `Concept 1` in another image.

### Reproducible clustering

For repeatable RCE map generation, use a fixed seed and multiple K-Means initializations:

```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=8,
    random_state=17,
    n_init=20,
)
```

A fixed seed improves run-to-run repeatability, but the scientific interpretation should still depend on spatial evidence and concept-effect ranking rather than the numeric cluster label.

> Do not claim that clustering is always identical or always correct. The defensible statement is that **cluster IDs may permute while the spatial explanation and interventional ranking can remain functionally consistent**.

---

## 5. Method status

### D-RISE

Implemented as a black-box perturbation method for object detection. Random masks are applied to the image, and each mask is weighted using the product of the two target-instance similarities.

### ODAM

Implemented as a paper-based, fixed-ROI gradient adaptation. The target boxes are held fixed, passed through ROIAlign and the frozen classification head, and explained using the pair probability.

This implementation should be described as an **ODAM-style or paper-based adaptation**, not as an official repository reproduction unless it is independently validated against an official implementation.

### Finer-CAM

Implemented using a detector-adapted contrastive objective over fixed ROI logits. For each target ROI, the target-class logit is contrasted with strong non-target reference logits.

Because the original method is class-logit oriented, the object-pair adaptation must be disclosed in any manuscript or report.

### CRAFT

CRAFT requires an appropriate model split and a calibration image collection. This repository accepts an externally generated official CRAFT heatmap through:

```text
--craft-map results/craft_official.npy
```

Do not label a CRAFT panel as official unless it was produced using the official CRAFT implementation and a documented calibration protocol.

### RCE

The benchmark accepts the raw RCE heatmap through:

```text
--rce-map results/rce.npy
```

The RCE map should be exported before manual contrast adjustment or figure editing. The shared visualization pipeline performs the final normalization.

---

## 6. Repository structure

```text
rce_baseline_benchmark/
├── baseline_xai/
│   ├── __init__.py
│   ├── common.py
│   ├── detector.py
│   ├── drise.py
│   ├── gradient_methods.py
│   └── visualize.py
├── data/
│   └── person_bicycle.jpg
├── results/
│   ├── rce.npy
│   └── craft_official.npy
├── outputs/
├── run_comparison.py
├── requirements.txt
├── CAPTION.tex
└── README.md
```

The `data/` and `results/` directories may be created by the user if they are not already present.

---

## 7. Installation

### Requirements

- Python 3.10 or newer;
- PyTorch 2.2 or newer;
- torchvision 0.17 or newer;
- NumPy;
- Pillow;
- Matplotlib;
- tqdm.

### Create an environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# Linux or macOS
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The first pretrained run downloads the torchvision Faster R-CNN weights.

---

## 8. Input preparation

### Input image

Use an RGB image containing the target object pair:

```text
data/person_bicycle.jpg
```

### RCE and CRAFT maps

Accepted heatmap formats:

- `.npy` containing a two-dimensional array;
- a grayscale image readable by Pillow.

Examples:

```python
import numpy as np

np.save("results/rce.npy", rce_heatmap.astype(np.float32))
np.save("results/craft_official.npy", craft_heatmap.astype(np.float32))
```

The map can have a different spatial resolution from the input image. It will be resized using bilinear interpolation and normalized by the shared loader.

The heatmap must be two-dimensional after squeezing singleton dimensions.

---

## 9. Running the benchmark

### Automatic target selection

The script selects the highest-confidence person and bicycle detections above the score threshold:

```bash
python run_comparison.py \
  --image data/person_bicycle.jpg \
  --rce-map results/rce.npy \
  --craft-map results/craft_official.npy \
  --drise-masks 2000 \
  --output outputs/figure9_baseline_comparison
```

### Manual fixed boxes

Manual boxes are recommended when exact target consistency is required across repeated experiments:

```bash
python run_comparison.py \
  --image data/person_bicycle.jpg \
  --person-box 170,45,330,360 \
  --bicycle-box 85,210,420,430 \
  --rce-map results/rce.npy \
  --craft-map results/craft_official.npy \
  --drise-masks 2000 \
  --output outputs/figure9_baseline_comparison
```

Box format:

```text
x1,y1,x2,y2
```

Both boxes must be supplied together.

### CPU execution

```bash
python run_comparison.py \
  --image data/person_bicycle.jpg \
  --device cpu \
  --rce-map results/rce.npy \
  --craft-map results/craft_official.npy
```

D-RISE can be slow on CPU because it requires many masked detector evaluations.

---

## 10. Main command-line options

| Option | Meaning | Default |
|---|---|---:|
| `--image` | Input image path | required |
| `--output` | Output filename prefix | `outputs/baseline_comparison` |
| `--device` | `cuda` or `cpu` | CUDA when available |
| `--person-box` | Fixed person box | automatic |
| `--bicycle-box` | Fixed bicycle box | automatic |
| `--score-threshold` | Minimum automatic detection score | `0.25` |
| `--drise-masks` | Number of D-RISE masks | `1000` |
| `--drise-batch` | D-RISE inference batch size | `8` |
| `--seed` | Random seed for D-RISE masks | `17` |
| `--finer-alpha` | Finer-CAM contrast strength | `0.7` |
| `--finer-references` | Number of reference classes | `3` |
| `--rce-map` | External RCE heatmap | optional |
| `--craft-map` | External official CRAFT heatmap | optional |
| `--no-pretrained` | Disable pretrained weights for code testing | false |

`--no-pretrained` is not appropriate for scientific comparison because an untrained detector does not provide meaningful explanations.

---

## 11. Outputs

A standard run creates:

```text
outputs/figure9_baseline_comparison.pdf
outputs/figure9_baseline_comparison.png
outputs/figure9_baseline_comparison.json
outputs/figure9_baseline_comparison_maps/
```

The map directory contains:

```text
drise.npy
odam.npy
finer_cam_detector_adapted.npy
```

The JSON file records:

- detector name;
- selected boxes;
- D-RISE mask count and seed;
- Finer-CAM adaptation parameters;
- method-status notes.

The metadata file should be retained with every reported figure.

---

## 12. Reproducibility recommendations

Use the same:

- input image file;
- detector checkpoint;
- target boxes;
- device type where practical;
- D-RISE seed;
- number of D-RISE masks;
- RCE clustering seed;
- RCE concept number `K`;
- RCE feature layer;
- intervention threshold;
- image preprocessing;
- heatmap normalization.

Recommended seed setup for the external RCE pipeline:

```python
import os
import random
import numpy as np
import torch

SEED = 17
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Optional stricter determinism; may reduce speed or reject some operations.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

Exact floating-point equality across different GPUs, CUDA versions, or library versions is not guaranteed. Functional consistency should be evaluated using spatial and ranking metrics rather than requiring every pixel to be numerically identical.

---

## 13. Fair-comparison rules

For publication-quality use:

1. Use the same image and selected target boxes for every method.
2. Keep detector weights frozen.
3. Do not manually modify individual heatmaps.
4. Do not apply method-specific smoothing or sharpening.
5. Use one resizing and normalization pipeline.
6. Report unsupported method-detector combinations as `N/A`.
7. Clearly distinguish official implementations from adaptations.
8. Record all random seeds and hyperparameters.
9. Retain raw `.npy` maps and metadata.
10. Do not select examples only because they make one method look better.

A qualitative panel should illustrate behavior; it should not replace quantitative faithfulness evaluation.

---

## 14. How to interpret the maps

A concentrated map indicates that a method assigns high relevance to a limited image region. A diffuse map indicates broader relevance distribution. Neither pattern is automatically correct.

Interpret a map together with:

- the selected target pair;
- detector confidence;
- deletion/AOC behavior;
- stability across seeds;
- localization overlap;
- concept-effect ranking;
- known limitations of the method.

RCE explains evidence associated with the selected detector output. It does not independently prove that a real-world semantic relationship exists between the detected objects.

---

## 15. Known limitations

- The current script is configured for a person-bicycle pair.
- The shared detector is Faster R-CNN ResNet-50 FPN v2.
- ODAM and Finer-CAM are detector-adapted implementations.
- Official CRAFT and RCE maps must be generated externally.
- D-RISE runtime increases with the number of masks.
- Fixed target boxes avoid differentiating through NMS but explain the selected ROI classification responses rather than the complete end-to-end proposal-selection process.
- Numeric concept IDs are not global semantic labels.
- A visually plausible map is not sufficient evidence of faithfulness.

---

## 16. Extending the code to another object pair

The current automatic labels use COCO IDs:

```text
person = 1
bicycle = 2
```

To support another pair, modify the label IDs and names passed to `select_best_pair` in `run_comparison.py`, or refactor them into command-line options.

Example concept:

```python
targets = select_best_pair(
    model,
    image,
    first_label=1,   # person
    second_label=73, # laptop in the torchvision COCO label convention
    score_threshold=0.25,
)
```

Always verify the class-ID convention used by the detector weights before changing labels.

---

## 17. Troubleshooting

### No person or bicycle detection found

Lower the threshold cautiously:

```bash
--score-threshold 0.15
```

Or provide both target boxes manually.

### CUDA out-of-memory during D-RISE

Reduce the batch size:

```bash
--drise-batch 2
```

The mask count controls sampling quality, while the batch size mainly controls memory use.

### Heatmap shape error

Ensure the supplied file becomes a two-dimensional array after `numpy.squeeze`:

```python
print(np.load("results/rce.npy").shape)
```

Valid examples include:

```text
(H, W)
(1, H, W)
(H, W, 1)
```

### RCE concept numbers change between runs

This is usually cluster-label permutation. Compare spatial maps and concept-effect rankings, set a fixed seed, and use optimal cluster matching before concluding that the explanation is unstable.

### Output appears blank

Check whether the raw heatmap contains meaningful variation:

```python
heatmap = np.load("results/rce.npy")
print(heatmap.min(), heatmap.max(), heatmap.std())
```

A constant map normalizes to zeros.

---

## 18. Suggested figure caption

```latex
\caption{Qualitative comparison of RCE with CRAFT, Finer-CAM,
D-RISE, and ODAM for a selected person--bicycle pair. All explanation
maps were generated or imported for the same frozen detector output and
processed using an identical resizing, normalization, and visualization
protocol.}
```

---

## 19. Citation

When the associated RCE manuscript is publicly available, replace the placeholder below with its final bibliographic record:

```bibtex
@article{rce_relational_concept_explainer,
  title   = {RCE: Deconstructing Object Relationships through Architecture-Agnostic Concept Interventions},
  author  = {Khalid, Muhammad Imran and Mi, Jian-Xun and others},
  journal = {To be updated},
  year    = {2026}
}
```

Users should also cite the original papers for each baseline they use.

---

## 20. License and responsible use

Add a repository license before public release, for example `LICENSE` using MIT, Apache-2.0, or the license required by the project institution.

Third-party methods, model weights, and datasets remain subject to their original licenses. This repository does not transfer ownership of external implementations or data.

The software is intended for research and evaluation. Explanation maps should not be treated as proof of physical causation or used as the sole basis for safety-critical decisions.

---

## 21. Contact

For questions, bug reports, or reproducibility issues, open a GitHub issue and include:

- operating system;
- Python version;
- PyTorch and torchvision versions;
- GPU and CUDA version;
- complete command;
- metadata JSON;
- error traceback;
- a minimal reproducible example where possible.
