FROM registry.cn-hangzhou.aliyuncs.com/baymin/ai-power:ai-power-v1.0
COPY detectron-2019-04-23 /Detectron
ADD train /usr/local/bin/train
ADD convert2coco /usr/local/bin/convert2coco
ADD README.txt /Detectron/README.txt
ADD PowerJson2COCO.py /Detectron/tools/PowerJson2COCO.py
RUN pip install visdom
ADD start.sh /Detectron/start.sh
WORKDIR /Detectron
RUN make
RUN make ops
EXPOSE 8097
CMD sh start.sh
