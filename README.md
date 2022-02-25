# Yolov5 for Fire Detection
In this repo, I trained yolov5 model to identify fire in images and videos. It borrows and modifies the script from [yolov5 repo](https://github.com/ultralytics/yolov5) to train the model on fire dataset. The fire detection task aims to identify fire from a video and put a bounding box around it.

<p align="center">
  <img src="results/result.gif" />
</p>

## Setup
Clone this repo and use the following script to install [yolov5](https://github.com/ultralytics/yolov5). Then download [fire dataset](https://mega.nz/file/MgVhQSoS#kOcuJFezOwU_9F46GZ1KJnX1STNny-tlD5oaJ9Hv0gY) and put it in datasets folder. The dataset contains subsamples from [Kaggle fire and Smoke](https://www.kaggle.com/dataclusterlabs/fire-and-smoke-dataset) and [Fire and Guns](https://www.kaggle.com/atulyakumar98/fire-and-gun-dataset) datasets. Note that I filtered out images only containing smokes and guns and then changed fire label in annotation files after removing their annotations.

```
# Install yolov5
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```
Use train.ipynb or the following commands for training & inference.
* Train
```
python train.py --img 640 --batch 16 --epochs 3 --data fire_config.yaml --weights yolov5s.pt --workers 0
```
* Inference
```
python detect.py --source input.mp4 --weights runs\train\exp\weights\best.pt
```

## Result

| P Curve | PR Curve | R Curve |
| :-: | :-: | :-: |
| ![](results/P_curve.png) | ![](results/PR_curve.png) | ![](results/R_curve.png) |

| Ground Truth | 
| :-: |
| ![](results/val_batch2_labels.jpg) |
| **Prediction** | 
| ![](results/val_batch2_pred.jpg) | 

## Reference

* https://github.com/robmarkcole/fire-detection-from-images
* https://github.com/ultralytics/yolov5 - for more information on training yolov5, please refer to its official homage page
