# -*- coding: utf-8 -*-
# @File : dataReader.py
# @Author: Runist
# @Time : 2020/3/30 15:16
# @Software: PyCharm
# @Brief: YOLOv3数据读取 -- 用tf.data

import tensorflow as tf
import numpy as np
import config.config as cfg
from PIL import Image
import cv2 as cv


class DataReader:

    def __init__(self, data_path, input_shape, batch_size, max_boxes=100):
        self.data_path = data_path
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.max_boxes = max_boxes
        self.train_lines, self.validation_lines = self.read_data_and_split_data()

    def read_data_and_split_data(self):
        with open(self.data_path, "r", encoding='utf-8') as f:
            files = f.readlines()

        split = int(cfg.valid_rate * len(files))
        train = files[split:]
        valid = files[:split]

        return train, valid

    def get_data(self, annotation_line):
        line = annotation_line.split()
        image = Image.open(line[0])
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        image_width, image_height = image.size
        input_width, input_height = self.input_shape
        scale = min(input_width / image_width, input_height / image_height)

        new_width = int(image_width * scale)
        new_height = int(image_height * scale)

        image = image.resize((new_width, new_height), Image.BICUBIC)
        new_image = Image.new('RGB', self.input_shape, (128, 128, 128))
        new_image.paste(image, ((input_width - new_width)//2, (input_height - new_height)//2))

        image = np.asarray(new_image) / 255

        dx = (input_width - new_width) / 2
        dy = (input_height - new_height) / 2

        # 为填充过后的图片，矫正bbox坐标
        box_data = np.zeros((self.max_boxes, 5), dtype='float32')
        if len(box) > 0:
            if len(box) > self.max_boxes:
                box = box[:self.max_boxes]

            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
            box_data[:len(box)] = box

        return image, box_data

    def get_random_data(self, annotation_line, hue=.1, sat=1.5, val=1.5):
        """
        数据增强（改变长宽比例、大小、亮度、对比度、颜色饱和度）
        :param annotation_line: 一行数据
        :param hue: 色调抖动
        :param sat: 饱和度抖动
        :param val: 明度抖动
        :return: image, box_data
        """
        line = annotation_line.split()
        image = Image.open(line[0])

        image_width, image_height = image.size
        input_width, input_height = self.input_shape

        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        # 随机生成缩放比例，缩小或者放大
        scale = rand(0.5, 1.5)
        # 随机变换长宽比例
        new_ar = input_width / input_height * rand(0.7, 1.3)

        if new_ar < 1:
            new_height = int(scale * input_height)
            new_width = int(new_height * new_ar)
        else:
            new_width = int(scale * input_width)
            new_height = int(new_width / new_ar)

        image = image.resize((new_width, new_height), Image.BICUBIC)

        dx = rand(0, (input_width - new_width))
        dy = rand(0, (input_height - new_height))
        new_image = Image.new('RGB', (input_width, input_height), (128, 128, 128))
        new_image.paste(image, (int(dx), int(dy)))
        image = new_image

        # 翻转图片
        flip = rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 图像增强
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = cv.cvtColor(np.array(image, np.float32)/255, cv.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image = cv.cvtColor(x, cv.COLOR_HSV2RGB)

        box_data = np.zeros((self.max_boxes, 5))
        # 为填充过后的图片，矫正box坐标，如果没有box需要检测annotation文件
        if len(box) <= 0:
            raise Exception("{} doesn't have any bounding boxes.".format(image_path))

        box[:, [0, 2]] = box[:, [0, 2]] * new_width / image_width + dx
        box[:, [1, 3]] = box[:, [1, 3]] * new_height / image_height + dy
        # 若翻转了图像，框也需要翻转
        if flip:
            box[:, [0, 2]] = input_width - box[:, [2, 0]]

        # 定义边界
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > input_width] = input_width
        box[:, 3][box[:, 3] > input_height] = input_height

        # 计算新的长宽
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        # 去除无效数据
        box = box[np.logical_and(box_w > 1, box_h > 1)]
        if len(box) > self.max_boxes:
            box = box[:self.max_boxes]

        box_data[:len(box)] = box

        return image, box_data

    def process_true_bbox(self, box_data):
        """
        对真实框处理，首先会建立一个13x13，26x26，52x52的特征层，具体的shape是
        [b, n, n, 3, 25]的特征层，也就意味着，一个特征层最多可以存放n^2个数据
        :param box_data: 实际框的数据
        :return: 处理好后的 y_true
        """
        # [anchors[mask] for mask in anchor_masks]

        # 维度(b, max_boxes, 5)还是一样的，只是换一下类型，换成float32
        true_boxes = np.array(box_data, dtype='float32')
        input_shape = np.array(self.input_shape, dtype='int32')  # 416,416

        # “...”(ellipsis)操作符，表示其他维度不变，只操作最前或最后1维。读出xy轴，读出长宽
        # true_boxes[..., 0:2] 是左上角的点 true_boxes[..., 2:4] 是右上角的点
        # 计算中心点 和 宽高
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

        # 实际的宽高 / 416 转成比例
        true_boxes[..., 0:2] = boxes_xy / input_shape
        true_boxes[..., 2:4] = boxes_wh / input_shape

        # 生成3种特征大小的网格
        grid_shapes = [input_shape // [32, 16, 8][i] for i in range(cfg.num_bbox)]
        # 创建3个特征大小的全零矩阵，[(b, 13, 13, 3, 25), ... , ...]存在列表中
        y_true = [np.zeros((self.batch_size,
                            grid_shapes[i][0], grid_shapes[i][1], cfg.num_bbox, 5 + cfg.num_classes),
                           dtype='float32') for i in range(cfg.num_bbox)]

        # 计算哪个先验框比较符合 真实框的Gw,Gh 以最高的iou作为衡量标准
        # 因为先验框数据没有坐标，只有宽高，那么现在假设所有的框的中心在（0，0），宽高除2即可。（真实框也要做一样的处理才能匹配）
        anchors = np.expand_dims(cfg.anchors, 0)
        anchor_rightdown = anchors / 2.     # 网格中心为原点(即网格中心坐标为(0,0)),　计算出anchor 右下角坐标
        anchor_leftup = -anchor_rightdown     # 计算anchor 左上角坐标

        # 长宽要大于0才有效,也就是那些为了补齐到max_boxes大小的0数据无效
        # 返回一个列表，大于0的为True，小于等于0的为false
        # 选择具体一张图片，valid_mask存储的是true or false，然后只选择为true的行
        valid_mask = boxes_wh[..., 0] > 0

        for b in range(self.batch_size):
            # 只选择 > 0 的行
            wh = boxes_wh[b, valid_mask[b]]
            wh = np.expand_dims(wh, -2)      # 在第二维度插入1个维度 (框的数量, 2) -> (框的数量, 1, 2)
            box_rightdown = wh / 2.
            box_leftup = -box_rightdown

            # 将每个真实框 与 9个先验框对比，刚刚对数据插入的维度可以理解为 每次取一个框出来shape（1,1,2）和anchors 比最大最小值
            # 所以其实可以看到源代码是有将anchors也增加一个维度，但在不给anchors增加维度也行。
            # 计算真实框和哪个先验框最契合，计算最大的交并比 作为最契合的先验框
            intersect_leftup = np.maximum(box_leftup, anchor_leftup)
            intersect_rightdown = np.minimum(box_rightdown, anchor_rightdown)
            intersect_wh = np.maximum(intersect_rightdown - intersect_leftup, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

            # 计算真实框、先验框面积
            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = cfg.anchors[..., 0] * cfg.anchors[..., 1]
            # 计算最大的iou
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            best_anchors = np.argmax(iou, axis=-1)

            # best_anchor是个list，label中标了几个框，他就计算出几个。
            # enumerate对他进行遍历，所以每个框都要计算合适的先验框
            for key, value in enumerate(best_anchors):
                # 遍历三次（三种类型的框 对应 三个不同大小的特征层）
                for n in range(cfg.num_bbox):
                    # 如果key（最优先验框的下表）
                    if value in cfg.anchor_masks[n]:
                        # 真实框的x比例 * grid_shape的长度，一般np.array都是（y,x）的格式，floor向下取整
                        # i = x * 13, i = y * 13 -- 放进特征层对应的grid里
                        i = np.floor(true_boxes[b, key, 0] * grid_shapes[n][1]).astype('int32')
                        j = np.floor(true_boxes[b, key, 1] * grid_shapes[n][0]).astype('int32')

                        # 获取 先验框（二维列表）内索引
                        k = cfg.anchor_masks[n].index(value)
                        c = true_boxes[b, key, 4].astype('int32')

                        # 三个大小的特征层， 逐一赋值
                        y_true[n][b, j, i, k, 0:4] = true_boxes[b, key, 0:4]
                        y_true[n][b, j, i, k, 4] = 1    # 置信度是1 因为含有目标
                        y_true[n][b, j, i, k, 5+c] = 1  # 类别的one-hot编码，其他都为0

        return y_true

    def generate(self, mode):
        """
        数据生成器
        :return: image, rpn训练标签， 真实框数据
        """
        if mode == 'train':
            n = len(self.train_lines)
        else:
            n = len(self.validation_lines)

        i = 0
        while True:
            image_data = []
            box_data = []

            if i == 0:
                np.random.shuffle(self.train_lines)
            for b in range(self.batch_size):
                if mode == 'train':
                    image, bbox = self.get_random_data(self.train_lines[i])
                else:
                    image, bbox = self.get_data(self.validation_lines[i])

                image_data.append(image)
                box_data.append(bbox)

                i = (i + 1) % n

            image_data = np.array(image_data)
            box_data = np.array(box_data)

            box_data = self.process_true_bbox(box_data)

            yield image_data, box_data


def rand(small=0., big=1.):
    return np.random.rand() * (big - small) + small




