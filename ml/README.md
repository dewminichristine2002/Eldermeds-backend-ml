# ElderMeds Intake Monitoring ML

This folder contains the optional model-training pipeline for the medication intake monitoring component.

The runtime now prefers **pill object detection** for counting. It counts each
detected pill box in the palm photo, then falls back to the older tablet-count
classifier if detector weights are not available.

```text
palm photo -> trained pill detector -> count detected boxes
```

The Medicine Reminder **Tablet Identifier** can also use a trained medicine-name
classifier. If no identity model is available, the backend falls back to
color/shape appearance matching against the user's saved medicine list.

```text
tablet photo -> pill object crop -> metric visual retrieval -> close medicine matches
tablet photo -> trained tablet identity classifier -> medicine name
tablet photo -> color/shape fallback -> closest saved medicine
```

## Dataset Format

Use a CSV manifest:

```csv
image_path,count,split
data/intake-count/images/train/count_2/sample.jpg,2,train
data/intake-count/images/val/count_2/sample.jpg,2,val
```

`count` is the number of tablets visible in the user's palm for that intake.

## Tablet Identity Dataset Format

Use a separate CSV manifest for medicine-name classification:

```csv
image_path,medicine_name,split
data/tablet-identity/images/train/paracetamol_500/sample_001.jpg,Paracetamol,train
data/tablet-identity/images/val/paracetamol_500/sample_050.jpg,Paracetamol,val
data/tablet-identity/images/train/metformin_500/sample_001.jpg,Metformin,train
```

Guidance:

- Capture many photos per medicine name, ideally 30+ per class.
- Use the same phone/camera style as the app.
- Include different lighting, angles, and plain backgrounds.
- Use one clear tablet per photo for identity training.
- Use the exact `medicine_name` text you save in ElderMeds so model output can match the user's medicine list.

Train the identity model:

```powershell
cd backend\ml
python scripts\train_tablet_identity_model.py `
  --manifest data\tablet-identity\manifest.csv `
  --output models\tablet_identity_classifier.pt `
  --epochs 12 `
  --batch-size 24 `
  --architecture efficientnet_b0 `
  --class-weights `
  --pretrained `
  --cpu
```

Predict a medicine name from a photo:

```powershell
python scripts\predict_tablet_identity.py `
  --model models\tablet_identity_classifier.pt `
  --image data\tablet-identity\images\val\paracetamol_500\sample_050.jpg `
  --cpu
```

The backend automatically looks for:

```text
backend/ml/models/tablet_identity_classifier.pt
```

You can override the model path and acceptance threshold:

```powershell
$env:TABLET_IDENTITY_MODEL_PATH='D:\Document\SLIIT\Research\Project\ElderMeds\backend\ml\models\tablet_identity_classifier.pt'
$env:TABLET_IDENTITY_MODEL_CONFIDENCE='0.55'
```

## ePillID Dataset Option

The ePillID benchmark is useful for the **Tablet Identifier** model because it
was built for pill image recognition rather than just pill detection. The paper
describes 13k images across 9804 appearance classes, with reference images and
real-world consumer images.

Recommended extracted layout:

```text
backend/ml/data/public/ePillID_data/
  ...
```

If the dataset is already extracted in that folder, build the manifest with:

```powershell
cd backend\ml
python scripts\setup_epillid_dataset.py
```

If you have the dataset as a local archive:

```powershell
python scripts\setup_epillid_dataset.py `
  --archive data\public\_downloads\ePillID_data.zip
```

Build an ElderMeds identity manifest from the extracted ePillID files:

```powershell
cd backend\ml
python scripts\build_identity_manifest_from_epillid.py `
  --dataset-root data\public\ePillID_data `
  --output data\public\epillid-identity-manifest.csv `
  --relative-paths-from .
```

If the auto-detected metadata columns are not correct, pass them explicitly:

```powershell
python scripts\build_identity_manifest_from_epillid.py `
  --dataset-root data\public\ePillID_data `
  --metadata data\public\ePillID_data\metadata.csv `
  --image-column image_path `
  --label-column medicine_name `
  --output data\public\epillid-identity-manifest.csv `
  --relative-paths-from .
```

Then train a standard classifier if you want a direct medicine-name prediction:

```powershell
python scripts\train_tablet_identity_model.py `
  --manifest data\public\epillid-identity-manifest.csv `
  --output models\tablet_identity_classifier.pt `
  --epochs 12 `
  --batch-size 24 `
  --architecture efficientnet_b0 `
  --class-weights `
  --pretrained `
  --cpu
```

For better same-color/same-shape matching on low-shot datasets, train the
ePillID-style metric embedding model. This follows the benchmark idea of
learning fine-grained visual embeddings instead of relying only on color,
shape, or one softmax prediction:

```powershell
python scripts\train_tablet_metric_identity_model.py `
  --manifest data\public\epillid-identity-manifest.csv `
  --output models\tablet_metric_identity_model.pt `
  --architecture efficientnet_b0 `
  --embedding-dim 512 `
  --classes-per-batch 16 `
  --samples-per-class 4 `
  --epochs 20 `
  --pretrained `
  --cpu
```

Then build the visual retrieval index from that trained metric model:

```powershell
python scripts\build_tablet_identity_index.py `
  --manifest data\public\epillid-identity-manifest.csv `
  --output models\tablet_identity_index.npz `
  --metric-model models\tablet_metric_identity_model.pt `
  --cpu
```

If you have not trained the metric model yet, the index builder can still use a
pretrained image encoder as a weaker fallback:

```powershell
python scripts\build_tablet_identity_index.py `
  --manifest data\public\epillid-identity-manifest.csv `
  --output models\tablet_identity_index.npz `
  --architecture efficientnet_b0 `
  --pretrained `
  --cpu
```

Test the visual index:

```powershell
python scripts\predict_tablet_identity_index.py `
  --index models\tablet_identity_index.npz `
  --image data\public\ePillID_data\classification_data\fcn_mix_weight\dc_224\0.jpg `
  --cpu
```

The backend automatically uses this index first when it exists:

```text
backend/ml/models/tablet_identity_index.npz
```

You can override the path and threshold:

```powershell
$env:TABLET_IDENTITY_INDEX_PATH='D:\Document\SLIIT\Research\Project\ElderMeds\backend\ml\models\tablet_identity_index.npz'
$env:TABLET_IDENTITY_METRIC_MODEL_PATH='D:\Document\SLIIT\Research\Project\ElderMeds\backend\ml\models\tablet_metric_identity_model.pt'
$env:TABLET_IDENTITY_RETRIEVAL_CONFIDENCE='0.72'
$env:TABLET_IDENTITY_RETRIEVAL_MARGIN='0.045'
$env:TABLET_IDENTITY_MODEL_MARGIN='0.15'
```

Important: ePillID is a research benchmark with low-shot fine-grained labels.
For the app to match a model prediction to a user's saved medicine, the trained
label text should match the `medicine_name` saved in ElderMeds. If the ePillID
metadata uses appearance IDs instead of medicine names, use it as pretraining or
rename/map labels before training the final app model.

Same color and same shape tablets cannot be separated safely with color/shape
rules alone. The app will mark those cases as ambiguous unless the trained
identity model has enough confidence. Capture photos where imprint, engraving,
score lines, coating texture, or side details are visible.

For same-color, same-shape, same-size tablets, the backend requires a clear
fine-detail match before choosing one saved medicine. Visual retrieval must pass
both the confidence threshold and the top-vs-second similarity margin
(`TABLET_IDENTITY_RETRIEVAL_MARGIN`). The classifier must pass its confidence
threshold and top-vs-second probability margin (`TABLET_IDENTITY_MODEL_MARGIN`).
If the top matches are too close, the API returns `status: "ambiguous"` with
same-appearance close matches instead of forcing an unsafe single answer.

Runtime identity flow is object-first:

1. Detect/localize the pill object with the trained pill detector when available.
2. If the detector does not return a box, isolate the pill with contour, edge,
   saturation, and adaptive-threshold computer vision.
3. Compute color, shape, and visual details from the pill mask/crop only.
4. Run visual retrieval/classifier on the isolated pill crop, not the full
   background image.

## Pillbox Metadata Dataset Option

The Pillbox archive is better for same-color/same-shape/same-size tablets
because it includes structured metadata such as medicine name, NDC/product code,
size, shape, score, imprint, color, strength, and ingredients.

Expected local files:

```text
D:\Downloads\pillbox_production_images_full_202008.zip
D:\Downloads\Pillbox_-_Archived_Data_20260623.xlsx
```

Build a rich Pillbox identity manifest and extract the matched images:

```powershell
cd backend\ml
python scripts\setup_pillbox_dataset.py `
  --metadata "D:\Downloads\Pillbox_-_Archived_Data_20260623.xlsx" `
  --image-archive "D:\Downloads\pillbox_production_images_full_202008.zip" `
  --manifest-output data\public\pillbox-identity-manifest.csv `
  --image-output-dir data\public\pillbox\images `
  --relative-paths-from .
```

Build a Pillbox retrieval index using the trained metric model:

```powershell
python scripts\build_tablet_identity_index.py `
  --manifest data\public\pillbox-identity-manifest.csv `
  --output models\pillbox_tablet_identity_index.npz `
  --metric-model models\tablet_metric_identity_model.pt `
  --batch-size 64 `
  --cpu
```

The backend prefers this richer index automatically when it exists:

```text
backend/ml/models/pillbox_tablet_identity_index.npz
```

Pillbox predictions include metadata in `metadata`, for example:

```json
{
  "color": "GREEN",
  "shape": "PENTAGON (5 SIDED)",
  "size": "7",
  "imprint": "par;129",
  "product_code": "49884-129",
  "strength": "DEXAMETHASONE 6 mg"
}
```

## Actual Public Dataset Option

Use a public YOLO-format pill detection dataset, then convert bounding-box labels into tablet-count labels.

Recommended starting dataset:

```text
Pills detection dataset
https://www.kaggle.com/datasets/alexanderyyy/pills-detection-dataset
```

Why it fits:

- YOLO object detection format
- one object class: `pill`
- train/validation split
- each pill has its own bounding box
- count can be calculated from the number of labels per image

Expected extracted layout:

```text
backend/ml/data/public/pills-detection-dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Convert YOLO labels to the training manifest:

```powershell
cd backend\ml
python scripts\build_count_manifest_from_yolo.py `
  --dataset-root data\public\pills-detection-dataset `
  --output data\public\pills-detection-count-manifest.csv
```

Dose-focused count manifest for this app:

```powershell
python scripts\build_count_manifest_from_yolo.py `
  --dataset-root data\public\pills-detection-dataset `
  --output data\public\pills-detection-count-manifest-dose.csv `
  --max-count 6
```

Then train:

```powershell
python scripts\train_tablet_count_model.py `
  --manifest data\public\pills-detection-count-manifest-dose.csv `
  --output models\tablet_count_classifier.pt `
  --epochs 10 `
  --batch-size 32 `
  --pretrained `
  --cpu
```

For more accurate count results, train the detector model:

```powershell
cd backend\ml
python scripts\train_pill_detector_model.py `
  --data pill-detector-data.yaml `
  --epochs 3 `
  --image-size 416 `
  --batch-size 8 `
  --device cpu
```

To prove model selection, run the same experiment across multiple YOLO variants:

```powershell
cd backend\ml
python scripts\compare_yolo_detector_models.py `
  --data pill-detector-data.yaml `
  --models yolo11n.pt yolo11s.pt yolo11m.pt `
  --epochs 3 `
  --image-size 416 `
  --batch-size 8 `
  --device cpu `
  --exist-ok
```

This creates:

```text
backend/ml/runs/detect/model_comparison.csv
backend/ml/runs/detect/model_comparison.md
```

Use the generated table as evidence that several YOLO models were evaluated
under the same dataset, image size, epochs, batch size, and device. The script
selects the deployment model by validation mAP50-95, then chooses the smaller
weights file when models are within the configured mAP tolerance. This supports
selecting `yolo11n.pt` when it gives comparable validation accuracy with the
lowest deployment cost.

The backend looks for detector weights at:

```text
backend/ml/runs/detect/pill_detector/weights/best.pt
```

You can override this path with:

```powershell
$env:INTAKE_DETECTOR_MODEL_PATH='D:\Document\SLIIT\Research\Project\ElderMeds\backend\ml\runs\detect\pill_detector\weights\best.pt'
```

Predict with the detector:

```powershell
python scripts\predict_pill_detector_count.py `
  --model runs\detect\pill_detector\weights\best.pt `
  --image data\public\pills-detection-dataset\images\val\sample.jpg
```

Quick public-dataset smoke training:

```powershell
python scripts\train_tablet_count_model.py `
  --manifest data\public\pills-detection-count-manifest-dose.csv `
  --output models\tablet_count_classifier_public_smoke.pt `
  --epochs 1 `
  --batch-size 16 `
  --max-train-rows 600 `
  --max-val-rows 160 `
  --cpu
```

## Train

```powershell
cd backend\ml
python scripts\train_tablet_count_model.py --manifest data\public\pills-detection-count-manifest-dose.csv --output models\tablet_count_classifier.pt --epochs 10 --pretrained --cpu
```

## Predict

```powershell
cd backend\ml
python scripts\predict_tablet_count.py --model models\tablet_count_classifier.pt --image data\public\pills-detection-dataset\images\val\sample.jpg --cpu
```

## Use A Trained Model In The Backend

Set:

```powershell
$env:INTAKE_COUNT_MODEL_PATH='D:\Document\SLIIT\Research\Project\ElderMeds\backend\ml\models\tablet_count_classifier.pt'
```

Then the existing intake endpoint can use the trained model:

```text
POST /api/intake-monitoring/analyze-palm
```

The detector uses a default confidence threshold of `0.55`. Override with:

```powershell
$env:INTAKE_DETECTOR_CONFIDENCE='0.55'
```

The fallback classifier threshold can be overridden with:

```powershell
$env:INTAKE_COUNT_MODEL_CONFIDENCE='0.70'
```

## Current Runtime

Image processing remains available as fallback/explainability in:

```text
backend/scripts/analyze_palm_pills.py
```

The recommended production design is:

1. Use the trained detector for tablet count prediction.
2. Use the classifier only if detector weights are unavailable.
3. Use image processing as a fallback/explainability check.

## Motion Monitoring Without Dataset Training

Motion monitoring is rule-based, not trained:

```text
MediaPipe landmarks -> distance rules -> hand-to-mouth + mouth pause result
```

Rule analyzer:

```powershell
python ..\scripts\analyze_intake_motion_rules.py
```

Optional landmark extraction helper for a recorded video:

```powershell
python ..\scripts\extract_mediapipe_motion_landmarks.py `
  --video data\public\sample-intake.mp4 `
  --output data\public\sample-intake-landmarks.json
```

Install optional dependencies only if you want to extract landmarks from videos:

```powershell
pip install mediapipe opencv-python
```
