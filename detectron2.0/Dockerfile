FROM registry.cn-hangzhou.aliyuncs.com/baymin/ai-power:ai-power-test
ADD generate_proposals.py /Detectron/detectron/ops/generate_proposals.py
ADD argsort.pyx /Detectron/detectron/utils/argsort.pyx
ADD cython_nms.pyx /Detectron/detectron/utils/cython_nms.pyx
ADD setup.py /Detectron/setup.py
WORKDIR /Detectron
RUN make
RUN make ops
EXPOSE 8097
CMD sh start.sh