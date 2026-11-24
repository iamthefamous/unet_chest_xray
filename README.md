# U-Net on chest X-rays: lung segmentation and 3-class diagnosis

A U-Net with an ImageNet-pretrained ResNet-18 encoder, fine-tuned on a CPU through **transfer learning**. It takes one chest X-ray and returns two things:

1. a **lung mask**: which pixels are lung. This is segmentation, the task U-Net was designed for.
2. a **class**: `normal`, `lung_opacity` (non-COVID pneumonia) or `covid` (COVID-19).

## Dataset

The data is 6,000 chest X-rays from the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database). Each image comes with a lung mask. Images (299×299) and masks (256×256) are resized to 128×128.

| Class | Source | Used |
|---|---|---|
| normal | [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge), *Normal* | 2,000 of 8,851 |
| lung_opacity | RSNA, *Lung Opacity* | 2,000 of 6,012 |
| covid | PadChest/BIMCV, Eurorad, SIRM, GitHub collections | 2,000 of 3,616 |

Images are sampled with a fixed seed. The split is stratified, so every split has equal class sizes:

| Split | Images | Per class |
|---|---|---|
| Train | 4,200 | 1,400 |
| Validation | 900 | 300 |
| Test | 900 | 300 |

The lungs cover 24% of the pixels on average.

**Source caveat.** Normal and lung-opacity images both come from RSNA, but COVID images come from other hospitals. They often have burned-in text, different framing and different contrast. A model can partly recognise the source instead of the disease, so COVID scores are likely optimistic.

The data is not in the repository. `python data.py` downloads the 816 MB archive (no Kaggle login needed) into `data/raw/`. It then builds `data/lungs_3class_128.npz` and exports demo images to `samples/`. The build takes about 20 seconds.

## Model

```
                    input 1x128x128 (grayscale, 0..1)
                                 |
  ENCODER: ResNet-18, pretrained on ImageNet          DECODER: U-Net, trained from scratch
  stem  conv 7x7/2       64 x 64x64  ---- skip ---->  up + 2x conv   16 x 64x64  -> up + 2x conv  16 x 128x128 (skip: input)
  layer1 (maxpool+res)   64 x 32x32  ---- skip ---->  up + 2x conv   32 x 32x32                         |
  layer2 (res blocks)   128 x 16x16  ---- skip ---->  up + 2x conv   64 x 16x16                    conv 1x1
  layer3 (res blocks)   256 x  8x8   ---- skip ---->  up + 2x conv  128 x  8x8                          |
  layer4 (res blocks)   512 x  4x4   (bottleneck) ->  (input of the decoder)                     LUNG MASK 1x128x128
                                 |
              global average pool -> dropout -> dense 3 -> CLASS (normal / lung_opacity / covid)
```

Each decoder step works like the original U-Net (Ronneberger et al., 2015): a 2×2 up-convolution doubles the resolution, the matching encoder feature map is concatenated through the skip connection, and two 3×3 convolutions with BatchNorm and ReLU follow.

Changes from the 2015 paper:

- The contracting path is a pretrained ResNet-18 instead of plain conv blocks trained from scratch. This is the transfer learning part.
- Convolutions use padding, so the mask has the same size as the input. The original U-Net used unpadded convolutions and produced a smaller output.
- BatchNorm is added after every convolution.
- A classification head on the bottleneck adds the 3-class diagnosis.

The model has 12.3M parameters: 11.2M in the pretrained encoder and 1.1M in the new decoder and heads.

## Transfer learning

- The encoder starts from torchvision's `ResNet18_Weights.IMAGENET1K_V1`. The 45 MB file downloads automatically the first time you train.
- ImageNet images are RGB and X-rays are gray. The first convolution's weights are summed over the three colour channels, so a gray image gets the same response as the same image in colour. Inputs are normalised with the ImageNet mean and standard deviation.
- The whole network is fine-tuned, with two learning rates: **1e-4** for the pretrained encoder, so its features are adjusted gently instead of erased, and **1e-3** for the new layers.
- The rest of the recipe: AdamW, cosine learning-rate schedule, batch size 32, 12 epochs, and loss = cross-entropy (class) + binary cross-entropy (mask). Augmentation applies the same small rotation, shift and zoom to the image and its mask. There are no flips, because the heart sits on the left. After every epoch the model is checked on the validation set, and the epoch with the best validation accuracy is saved.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python data.py
```

## Usage

| Step | Command | Output in `outputs/` |
|---|---|---|
| Train and save | `python train.py` | `model.pt`, `history.json`, `learning_curves.png` |
| Test | `python evaluate.py` | `test_metrics.txt`, `confusion_matrix.png`, `segmentation_examples.png` |
| Predict | `python predict.py samples/*.png` | printed class and probabilities, plus `predictions/<name>_lungs.png` |

The trained model `outputs/model.pt` is in the repository, so `evaluate.py` and `predict.py` work without training first. `evaluate.py` needs the dataset (`python data.py`), but `predict.py` does not. `predict.py` accepts any PNG or JPG chest X-ray. It converts the image to grayscale, center-crops it and resizes it to 128×128.

## Files

| File | Purpose |
|---|---|
| `data.py` | Download, balanced 3-class dataset build with lung masks, loading, joint image and mask augmentation, preprocessing of new images |
| `model.py` | U-Net with ResNet-18 encoder, Dice score, save and load |
| `train.py` | Transfer-learning training loop and learning curves |
| `evaluate.py` | Accuracy, per-class precision and recall, confusion matrix, Dice and IoU on the test set |
| `predict.py` | Lung mask and class for new images |
