# The Pareidolia Paradox - lunar crater (Depth) vs mound (Rise) classification

Binary classification of 256x256 grayscale lunar crops: **Class 0 = Depth** (craters, holes) and **Class 1 = Rise** (mounds, boulders).
Metric: **balanced accuracy**.

## Why it is hard - and how the sun azimuth is used

A crater lit from the left and a mound lit from the right produce the *same* picture. Without knowing where the sun is, the two
classes are indistinguishable (pareidolia). The metadata column `sun_azimuth_angle` resolves the ambiguity, so **every image is
rotated counter-clockwise by `-sun_azimuth_angle` before training and inference** (`physics.rotate_ccw`; reflect-padded to the
image diagonal, rotated, centre-cropped, so no black corners leak information). After this step the sun always comes from the
same direction, and "bright on the sun side => mound, bright on the far side => crater" becomes a rule the CNN can learn.

### Verifying the rotation instead of trusting it
We do not assume the convention silently; `physics.discover_physics` measures it on the training set:

1. For each image we compute a *brightness-dipole vector* (high-passed, Gaussian-windowed first moment). It points toward the bright
   side of the object: along the sun for a mound, away from the sun for a crater.
2. Using the labels, we fit `light_angle = phi0 + a * azimuth` for `a in {-2..2}`.
   `a = +1` means the competition rule ("rotate CCW by -az") is right, `a = -1` means the opposite sign, and a strong fit for **both**
   signs means the data mixes two conventions.
3. Candidate input representations are compared with a short, identical pilot run (small EfficientNet-B0, same split):
   * `spec`   - rotate by `-az` (competition rule)
   * `opp`    - rotate by `+az`
   * `dual`   - channels `[spec, opp, raw]`, the network decides
   * `routed` - per image, choose `spec` or `opp` from that image's own shading axis (label-free, so it also works on test data)
   
   The winner is stored in `pipeline_config.json` and used identically by `train.py` and `inference.py`.
4. Each mode also adds a constant rotation so the light arrives from angle 0 (from the right). A vertical (top<->bottom) flip then keeps
   the light direction, so it is a label-preserving augmentation/TTA. It is enabled only when the probe shows a strong, consistent fit. No 90/180-degree
   rotations or left<->right flips are ever used because they would turn a crater into a mound.

## Model
ImageNet-21k-pretrained EfficientNetV2-S (`timm`), 320 px, class-weighted focal loss with label smoothing, AdamW (10x LR on the head), warm-up + cosine LR,
EMA weights, stochastic depth, gradient clipping, mixed precision. 5-fold stratified CV; model selection by balanced accuracy.
Test predictions = mean of the 5 fold models (with physics-safe TTA); the decision threshold is tuned on out-of-fold predictions only.

## Repository layout
| file | purpose |
|---|---|
| `physics.py` | rotation normalisation, light-direction probe, input construction (numpy/OpenCV only) |
| `common.py` | data loading, model, training loop, K-fold, inference helpers |
| `train.py` | full training pipeline (probe -> mode selection -> K-fold -> OOF score) |
| `inference.py` | builds `submission.csv` from trained weights |
| `requirements.txt` | dependencies |

## Usage
```bash
pip install -r requirements.txt

# data folder must contain: train_images.zip, test_images.zip, train_metadata.csv, test_metadata.csv
python train.py --data_dir /path/to/data --out_dir ./outputs          # ~1-2 h on a T4 (5 folds)
python inference.py --data_dir /path/to/data --weights_dir ./outputs --out submission.csv
```
Useful flags: `--mode {auto,spec,opp,dual,routed}`, `--img_size 384`, `--epochs`, `--backbone convnext_tiny.fb_in22k_ft_in1k --run_name convnext`.
Training resumes automatically after an interruption (finished folds are skipped).

## Model weights
Download[https://drive.google.com/drive/folders/1jKTrA_2qV7NN1Khl9IxmUgFCs5j7rTBy?usp=drive_link]

Download and Place the file (`model_effv2s_fold0..4.pt`) in `./outputs` and run `inference.py`.

## Submission format
`submission.csv` has exactly 2,000 rows plus the header `image_id,label`, labels in {0, 1}, no nulls (validated in `make_submission`).
