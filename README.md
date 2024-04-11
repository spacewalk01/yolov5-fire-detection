<h1 align="center"><span>YOLOv5/YOLOv9 for Fire Detection</span></h1>

Fire detection task aims to identify fire or flame in a video and put a bounding box around it. This repo includes a demo on how to build a fire detector using YOLOv5/YOLOv9. 

<p align="center">
  <img src="results/result.gif" />
</p>

## 🛠️ Installation
1. Clone this repo 
``` shell
# Clone
git clone https://github.com/spacewalk01/yolov5-fire-detection.git
cd Yolov5-Fire-Detection
```

2. Install [YOLOv5](https://github.com/ultralytics/yolov5). 
``` shell
git clone https://github.com/ultralytics/yolov5.git 
cd yolov5
pip install -r requirements.txt
```

3. Or install [YOLOv9](https://github.com/WongKinYiu/yolov9.git)
``` shell
git clone https://github.com/WongKinYiu/yolov9.git
cd yolov9
pip install -r requirements.txt
```

## 🏋️ Training
I set up ```train.ipynb``` script for training the model from scratch. To train the model, download [Fire-Dataset](https://drive.google.com/file/d/1TQKA9nzo0BVwtmojmSusDt5j02KWzIu9/view?usp=sharing) and put it in ```datasets``` folder. This dataset contains samples from both [Fire & Smoke](https://www.kaggle.com/dataclusterlabs/fire-and-smoke-dataset) and [Fire & Guns](https://www.kaggle.com/atulyakumar98/fire-and-gun-dataset) datasets on Kaggle. I filtered out images and annotations that contain smokes & guns as well as images with low resolution, and then changed fire annotation's label in annotation files.

- YOLOv5
```
python train.py --img 640 --batch 16 --epochs 10 --data ../fire.yaml --weights yolov5s.pt --workers 0
```

- YOLOv9
```
python train_dual.py --workers 4 --device 0 --batch 16 --data ../fire.yaml --img 640 --cfg models/detect/yolov9-c.yaml --weights '' --name yolov9-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 50 --close-mosaic 15
```

## 🌱 Inference

- YOLOv5
  
If you train your own model, use the following command for detection:
``` shell
python detect.py --source ../input.mp4 --weights runs/train/exp/weights/best.pt --conf 0.2
```
Or you can use the pretrained model located in ```models``` folder for detection as follows:
``` shell
python detect.py --source ../input.mp4 --weights ../models/yolov5s_best.pt --conf 0.2
```

- YOLOv9

``` shell
python detect.py --weights runs/train/yolov9-c2/weights/best.pt --source ../input.mp4
```

You can download the pretrained yolov9-c.pt model from [google drive](https://drive.google.com/file/d/1nV5C3dbc_Q3CoczHaERTojr78-SFPdMI/view?usp=sharing) for fire detection. Note that this model was trained on the fire dataset for 50 epochs. Refer to [link](https://github.com/WongKinYiu/yolov9/issues/162) to fix for detect.py runtime error when running yolov9.

## ⏱️ Results
The following charts were produced after training YOLOv5s with input size 640x640 on the fire dataset for 10 epochs.

| P Curve | PR Curve | R Curve |
| :-: | :-: | :-: |
| ![](results/P_curve.png) | ![](results/PR_curve.png) | ![](results/R_curve.png) |

#### Prediction Results
The fire detection results were fairly good even though the model was trained only for a few epochs. However, I observed that the trained model tends to predict red emergency light on top of police car as fire. It might be due to the fact that the training dataset contains only a few hundreds of negative samples. We may fix such problem and further improve the performance of the model by adding images with non-labeled fire objects as negative samples. The [authors](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results) who created YOLOv5 recommend using about 0-10% background images to help reduce false positives. 

| Ground Truth | Prediction | 
| :-: | :-: |
| ![](results/val_batch2_labels_1.jpg) | ![](results/val_batch2_pred_1.jpg) |
| ![](results/val_batch2_labels_2.jpg) | ![](results/val_batch2_pred_2.jpg) | 

#### Feature Visualization
It is desirable for AI engineers to know what happens under the hood of object detection models. Visualizing features in deep learning models can help us a little bit understand how they make predictions. In YOLOv5, we can visualize features using ```--visualize``` argument as follows:

```
python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.2 --source ../datasets/fire/val/images/0.jpg --visualize
```

| Input | Feature Maps | 
| :-: | :-: |
| ![](results/004dec94c5de631f.jpg) | ![](results/stage23_C3_features.png) |

## 🔗 Reference
I borrowed and modified [YOLOv5-Custom-Training.ipynb](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) script for training YOLOv5 model on the fire dataset. For more information on training YOLOv5, please refer to its homepage.
* https://github.com/robmarkcole/fire-detection-from-images
* https://github.com/ultralytics/yolov5
* https://github.com/AlexeyAB/darknet
