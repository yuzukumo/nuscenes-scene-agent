# Dataset Downloads

Dataset archives, extracted datasets, map files, and generated artifacts are excluded from version control.

Download the dataset files yourself and place them under `archives/` before running the preparation commands.

## Download URLs

### Mini

- `v1.0-mini.tgz`
  - https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz

### Map Expansion

- `nuScenes-map-expansion-v1.3.zip`
  - https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/nuScenes-map-expansion-v1.3.zip

### Trainval

- `v1.0-trainval_meta.tgz`
  - https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval_meta.tgz
- `v1.0-trainval01_blobs.tgz`
  - https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval01_blobs.tgz
- `v1.0-trainval02_blobs.tgz`
  - https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval02_blobs.tgz
- `v1.0-trainval03_blobs.tgz`
  - https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval03_blobs.tgz
- `v1.0-trainval04_blobs.tgz`
  - https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval04_blobs.tgz
- `v1.0-trainval05_blobs.tgz`
  - https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval05_blobs.tgz
- `v1.0-trainval06_blobs.tgz`
  - https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval06_blobs.tgz
- `v1.0-trainval07_blobs.tgz`
  - https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval07_blobs.tgz
- `v1.0-trainval08_blobs.tgz`
  - https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval08_blobs.tgz
- `v1.0-trainval09_blobs.tgz`
  - https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval09_blobs.tgz
- `v1.0-trainval10_blobs.tgz`
  - https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval10_blobs.tgz

### Optional CAN Bus

- `can_bus.zip`
  - https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/can_bus.zip

## Expected Local Layout

```text
archives/
  mini/
    v1.0-mini.tgz
  maps/
    nuScenes-map-expansion-v1.3.zip
  trainval/
    v1.0-trainval_meta.tgz
    v1.0-trainval01_blobs.tgz
    ...
    v1.0-trainval10_blobs.tgz
```

## Optional nuPlan Data

Download `nuPlan` files from the official `nuPlan` dataset page. The replay-regression extension expects extracted files under:

```text
data/nuplan/
  archives/
    nuplan-maps-v1.0.zip
    nuplan-v1.1_mini.zip
    nuplan-v1.1_val.zip
    nuplan-v1.1_train_boston.zip
    nuplan-v1.1_train_pittsburgh.zip
    nuplan-v1.1_train_singapore.zip
  dataset/
    maps/
    data/cache/mini/
    data/cache/val/
    data/cache/train_boston/
    data/cache/train_pittsburgh/
    data/cache/train_singapore/
    nuplan-v1.1/splits/
```
