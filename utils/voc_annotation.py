# -*- coding: utf-8 -*-
# @File : voc_annotation.py
# @Author: Runist
# @Time : 2020/5/8 10:48
# @Software: PyCharm
# @Brief: voc转换为yolo3读取的格式


import xml.etree.ElementTree as ET
import config.config as cfg
import random
import os


def convert_annotation(xml_path, list_file):
    """
    把单个xml转换成annotation格式
    :param xml_path: xml的路径
    :param list_file: 写入的文件句柄
    :return: None
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.iter('object'):

        cls = obj.find('name').text
        cls_id = cfg.class_names.index(cls)
        xmlbox = obj.find('bndbox')

        b = (int(xmlbox.find('xmin').text),
             int(xmlbox.find('ymin').text),
             int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))

        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


if __name__ == '__main__':
    # VOC数据集的路径
    xml_file_path = 'C:/Software/Code/Work_Python/Dataset/VOCdevkit/VOC2012/Annotations'
    xml_list = os.listdir(xml_file_path)
    total_xml = []
    for xml in xml_list:
        if xml.endswith(".xml"):
            total_xml.append(xml)

    train_percent = 0.9
    test_percent = 1 - train_percent

    # 生成一段序列，然后在序列中随机选索引
    num = len(total_xml)
    image_range = range(num)
    tr = int(num * train_percent)
    te = int(num * test_percent)

    # 第一个参数序列，第二个参数去除的个数，从a中取出n个数字
    train_num = random.sample(image_range, tr)

    train_ids = []
    test_ids = []

    # 训练集和测试集分开
    for i in image_range:
        name = total_xml[i][:-4]
        if i in train_num:
            train_ids.append(name)
        else:
            test_ids.append(name)

    # 将信息写入train.txt和test.txt
    image_ids = {"train": train_ids, "test": test_ids}
    for key, value in image_ids.items():
        files = open('../config/{}.txt'.format(key), 'w')
        for image_id in value:
            img_path = 'C:/Software/Code/Work_Python/Dataset/VOCdevkit/VOC2012/JPEGImages/{}.jpg'.format(image_id)
            xml_path = '{}/{}.xml'.format(xml_file_path, image_id)
            files.write(img_path)
            convert_annotation(xml_path,  files)
            files.write('\n')
        files.close()
