# -*- coding: utf-8 -*-
# @File : train.py.py
# @Author: Runist
# @Time : 2020/4/7 12:37
# @Software: PyCharm
# @Brief: 训练脚本

import tensorflow as tf
import config.config as cfg
from core.dataReader import DataReader
from core.loss import YoloLoss
from nets.yolo import yolo_body

import os
from tensorflow.keras import optimizers, callbacks, metrics
from tensorflow.keras.optimizers.schedules import PolynomialDecay


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # 读取数据
    reader = DataReader(cfg.annotation_path, cfg.input_shape, cfg.batch_size)
    train_data = reader.generate('train')
    validation_data = reader.generate('validation')
    train_steps = len(reader.train_lines) // cfg.batch_size
    validation_steps = len(reader.validation_lines) // cfg.batch_size

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(reader.train_lines),
                                                                               len(reader.validation_lines),
                                                                               cfg.batch_size))

    optimizer = optimizers.Adam(learning_rate=cfg.learn_rating)
    yolo_loss = [YoloLoss(cfg.anchors[mask]) for mask in cfg.anchor_masks]

    train_by_fit(optimizer, yolo_loss, train_data, train_steps, validation_data, validation_steps)


def train_by_fit(optimizer, loss, train_data, train_steps, validation_data, validation_steps):
    """
    使用fit方式训练，可以知道训练完的时间，以及更规范的添加callbacks参数
    :param optimizer: 优化器
    :param loss: 自定义的loss function
    :param train_data: 以tf.data封装好的训练集数据
    :param validation_data: 验证集数据
    :param train_steps: 迭代一个epoch的轮次
    :param validation_steps: 同上
    :return: None
    """
    cbk = [
        callbacks.ReduceLROnPlateau(verbose=1),
        callbacks.EarlyStopping(patience=10, verbose=1),
        callbacks.ModelCheckpoint('./model/yolov3_{val_loss:.04f}.h5', save_best_only=True, save_weights_only=True)
    ]

    model = yolo_body()
    model.compile(optimizer=optimizer, loss=loss)

    # initial_epoch用于恢复之前的训练
    model.fit(train_data,
              steps_per_epoch=max(1, train_steps),
              validation_data=validation_data,
              validation_steps=max(1, validation_steps),
              epochs=cfg.epochs,
              callbacks=cbk)


if __name__ == '__main__':
    main()
