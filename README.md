# Card Trainer using Keras implentation of Mask-RCNN

## Install a virtual environments and dependencies
If you don't have a suitable pyto
hon environment:

```(sh)
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Downlaod training data and pre-trained model

```(sh)
gsutil -m cp gs://champop2/mrcnn/data/mask_rcnn_coco.h5 data/
gsutil -m cp -r gs://champop2/mrcnn/data/images data/
gsutil -m cp -r gs://champop2/mrcnn/data/masks data/

```

## Run training job

```(sh)
python3 -m trainer.champop --job-dir gs://champop2/mrcnn --weights data/mask_rcnn_coco.h5 --data-dir data
```
