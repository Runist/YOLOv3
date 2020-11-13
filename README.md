# Yolov3

> A lightweight YOLOv3. <br>

## 1. Quick Start.

1. Clone the repositorie.

```python
$ git clone https://github.com/Runist/YOLOv3.git
```

2. You are supposed to install some dependencies before getting out hands with these codes.

```python
$ cd YOLOv3
$ pip install -r ./requirements.txt
```

3. Download VOC weights.

```Python
$ wget https://github.com/Runist/YOLOv3/releases/download/v1.0/voc_weight.h5
```

## 2. Train your dataset.

One files are required as follows:

- [`train.txt`](https://github.com/Runist/YOLOv3/config/trian.txt): 

```
xxx/xxx.jpg 174,101,349,351,14
xxx/xxx.jpg 104,78,375,183,0 133,88,197,123,0 195,180,213,229,14 26,189,44,238,14
# image_path x_min,y_min,x_max,y_max,class_id  x_min,y_min,...,class_id 
# make sure that x_max < width and y_max < height
```

And you need edit config.py:

```pYTHON
annotation_path = "./config/train.txt"
class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
```

Finally

```python
$ mkdir logs
$ cd logs
$ mkdir model
$ python train.py			
```

## 3. Show your predict result.

```python
$ python predict.py
```

If your want to change picture path, please edit predict.py:

```python
if __name__ == '__main__':
    img_path = "Your image path."
```
![result.jpg](https://i.loli.net/2020/11/13/Tr7IR2bW8Bk4gDN.jpg)
## 4. Reference.

- [bubbliiiing/yolo3-keras](https://github.com/bubbliiiing/yolo3-keras)

