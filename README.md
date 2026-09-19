# 🌕 Pareidolia Paradox — Lunar Surface Depth/Rise Classifier

A deep learning pipeline that classifies 256×256 grayscale lunar surface crops as
either a **depression** (crater, hole) or a **protrusion** (mound, hill, boulder) —
a classic case of *pareidolia*, where the same 2D shading pattern can look either
concave or convex depending on where the light is coming from.

This repo was built for a computer-vision hackathon, but the approach — physics-aware
preprocessing, transfer learning, cross-validation, and test-time augmentation — is a
general template for any binary image classification problem with a known confound
(here, lighting direction) that needs to be normalized out before training.

## The problem

Craters and boulders can look identical in a single 2D image: a crater lit from the
left casts a shadow on its right side, and a boulder lit from the right casts a shadow
on its left — the shading pattern is nearly indistinguishable. Human vision (and naive
CNNs) resolve this ambiguity using an implicit assumption about where the light is
coming from. If that assumption doesn't hold for a given image, the shape reads as
"inside-out."

Each image in this dataset comes with a `sun_azimuth_angle` — the direction of the
light source at capture time. The key idea is to **rotate every image so the sun
direction is standardized before the model ever sees it**, removing the ambiguity
instead of asking the model to learn around it.

## Approach

1. **Illumination normalization** — each image is rotated by `-sun_azimuth_angle`
   (padded and reflected first so no black corners are introduced, then cropped back
   to the original size), applied identically at training and inference time.
2. **Transfer learning** — a pretrained CNN backbone (EfficientNetV2, via `timm`) is
   fine-tuned on the normalized images, with the single grayscale channel replicated
   to three channels to match the pretrained input format.
3. **Class-balanced training** — a weighted loss function and balanced-accuracy-based
   checkpointing correct for any skew between the two classes, rather than optimizing
   for raw accuracy.
4. **Cross-validation** — stratified k-fold training gives a much more trustworthy
   estimate of real-world performance than a single train/validation split, and the
   resulting fold models double as a free ensemble.
5. **Test-time augmentation** — predictions are averaged across several
   label-preserving views (flips, rotations) of each test image to reduce variance.
6. **Validated output** — the final predictions file is checked in code (correct row
   count, columns, value range, and exact match against the test set) before being
   written, to catch formatting mistakes before submission.

## Repo contents

| File | Description |
|---|---|
| `Pareidolia_Paradox_Solution.ipynb` | End-to-end Google Colab notebook: data loading, preprocessing, training, inference, and submission generation. |
| `submission.csv` | Final predictions (`image_id,label`) for the evaluation set, produced by the notebook. |
| `model_fold*.pt` | PyTorch checkpoints, one per cross-validation fold. |
| `best_model_weights.h5` | HDF5 export of the best-performing fold's weights. |

## Getting started

The notebook is written for **Google Colab** with a GPU runtime, and expects the
dataset to be available in a connected Google Drive folder.

1. Open `Pareidolia_Paradox_Solution.ipynb` in Colab.
2. Set **Runtime → Change runtime type → GPU**.
3. Make sure your Drive contains the four dataset files (two image archives, two
   metadata CSVs) in one folder, and update the folder path in the config cell if
   yours differs from the default.
4. Run the notebook top to bottom. Training time scales with the number of
   cross-validation folds and epochs configured at the top of the notebook — both
   are exposed as simple settings you can reduce if you're short on GPU time.
5. The final `submission.csv` and model weight files are written back to the same
   Drive folder.

No dataset is included in this repository — you'll need to supply your own
images and metadata in the same format (`image_id`, `label`, `sun_azimuth_angle`
columns) to reproduce or adapt this pipeline.

## Method summary (for the curious)

- **Backbone:** EfficientNetV2-S, ImageNet-pretrained, fine-tuned end-to-end.
- **Validation:** 5-fold stratified cross-validation.
- **Loss:** class-weighted cross-entropy with light label smoothing.
- **Optimizer/schedule:** AdamW with cosine-annealing warm restarts, mixed-precision
  training.
- **Metric:** balanced accuracy (mean recall across both classes), used for both
  checkpoint selection and reporting.
- **Inference:** 4-view test-time augmentation, averaged across all cross-validation
  folds.

## License

Add a license of your choice (e.g. MIT) if you intend for others to reuse this code.

## Acknowledgments

Built for a computer vision hackathon challenge on lunar surface feature
classification. Thanks to the organizers for an interesting problem — and to the
moon, for holding still long enough to be photographed.
