FROM registry.cn-hangzhou.aliyuncs.com/baymin/ai-power:ai-power-v2.0
RUN cat /Detectron/detectron/ops/generate_proposals.py
ADD generate_proposals.py /Detectron/detectron/ops/generate_proposals.py
RUN cat /Detectron/detectron/ops/generate_proposals.py
EXPOSE 8097
CMD sh start.sh