0.需要更改镜像初始化运行可以重新映射start.sh到/Detectron

1.powerAI软件的数据集转coco数据集使用
usage:	convert2coco	-d  需要转换的文件夹目录，powerai导出的数据，可以保护多个pjson文件，但是所有文件必须是在同级目录
			-o  output filepath "/home/baymin/demo.json" default <d>/coco/annotations/instances_train2014.json
			-t  default "pjson"
			-n  run in background
			-h  prints help screen

2.开始训练
usage:	train	-c  训练的cfg文件路径
		-d  训练结果保存路径 default:"detectron/datasets/data/result"
		-n  是否后台运行
		-h  帮助

3.验证数据
python tools/infer_simple_test.py --cfg detectron/datasets/data/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml --output-dir detectron/datasets/data/test --wts detectron/datasets/data/result/train/coco_2014_train/generalized_rcnn/model_iter59999.pkl detectron/datasets/data/coco/coco_val2014/
