# Yolov5 for Fire Detection

## Dataset
The dataset can be download from [here](https://mega.nz/file/MgVhQSoS#kOcuJFezOwU_9F46GZ1KJnX1STNny-tlD5oaJ9Hv0gY). It contains subsamples from Kaggle fire dataset and FireNet datasets.


## Train
```
python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt --workers 1
```
## Inference
```
python detect.py --source E:/Research/Project2/fire/input/14.mp4 --weights E:\Programs\yolov5\runs\train\exp10\weights\best.pt
```

# Result
```
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/fire  # dataset root dir
train: train/images  # train images (relative to 'path') 128 images
val: val/images  # val images (relative to 'path') 128 images
test:  # test images (optional)

# Classes
nc: 1  # number of classes
names: ['fire']  # class names
```

| Ground Truth | **Prediction** | 
| :-: | :-: | 
| ![](results/val_batch2_labels.jpg) | ![](results/val_batch2_pred.jpg) | 



## Reference

* https://github.com/robmarkcole/fire-detection-from-images
* https://github.com/ultralytics/yolov5
