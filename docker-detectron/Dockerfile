FROM registry.cn-hangzhou.aliyuncs.com/baymin/wfylozj:1.0-detectron

ADD start.sh /Detectron/start.sh
ADD train /usr/local/bin/train
ADD convert2coco /usr/local/bin/convert2coco
ADD PowerJson2COCO.py /Detectron/tools/PowerJson2COCO.py

WORKDIR /Detectron

CMD sh start.sh