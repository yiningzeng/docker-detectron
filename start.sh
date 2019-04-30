#!/bin/bash
# writen by zengwei
nohup visdom &
convert2coco -d /Detectron/detectron/datasets/data
train -c /Detectron/detectron/datasets/data/train-config.yaml