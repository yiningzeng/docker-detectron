# -*- coding:utf-8 -*-
# !/usr/bin/env python


# from labelme import utils
import argparse
import json
import numpy as np
import glob
import PIL.Image
import PIL.ImageDraw
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a network with Detectron'
    )
    parser.add_argument(
        '--dir',
        dest='dir',
        help='input dir'
    )
    parser.add_argument(
        '--type',
        dest='type',
        default="pjson",
        help='filetype'
    )
    parser.add_argument(
        '--outfile',
        dest='outfile',
        help='set outfile'
    )
    if len(sys.argv) < 5:
        parser.print_help()
        sys.exit(1)
    print(len(sys.argv))
    return parser.parse_args()


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

class powerAi2coco(object):
    def __init__(self, power_json=[], save_json_path='./tran.json'):
        '''
        :param power_json: 所有labelme的json文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.power_json = power_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        # self.data_coco = {}
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0

        self.id = 0
        self.save_json()

    def data_transfer(self):

        for num, json_file in enumerate(self.power_json):
            print json_file
            with open(json_file, 'r') as fp:
                data = json.load(fp)  # 加载json文件
                for i, one in enumerate(data):
                    self.images.append(self.image(one))
                    for n, regions in enumerate(one['regions']):
                        label = regions['tags'][0]
                        if label not in self.label:
                            self.categories.append(self.categorie(label))
                            self.label.append(label)

                        points =[]
                        tempPoints=regions['points']#这里的point是用rectangle标注得到的，只有两个点，需要转成四个点
                        for j, p in enumerate(tempPoints):
                            points.append([int(round(tempPoints[j]['x'])), int(round(tempPoints[j]['y']))])
                        points.append([int(round(tempPoints[0]['x'])), int(round(tempPoints[1]['y']))])
                        points.append([int(round(tempPoints[1]['x'])), int(round(tempPoints[0]['y']))])
                        self.annotations.append(self.annotation(points, label))
                        self.annID += 1

    def image(self, data):
        image = {}
        # img = utils.img_b64_to_arr(data['imageData'])  # 解析原图片数据
        # img=io.imread(data['imagePath']) # 通过图片路径打开图片
        # img = cv2.imread(str(data['asset']['path']).replace("file:", ""), 0)
        # height, width = img.shape[:2]
        img = None
        image['height'] = data['asset']['size']['height']
        image['width'] = data['asset']['size']['width']
        self.id = self.id + 1
        image['id'] = self.id
        image['file_name'] = data['asset']['name'] # data['imagePath'].split('/')[-1]

        self.height = data['asset']['size']['height']
        self.width = data['asset']['size']['width']

        return image

    def categorie(self, label):
        categorie = {}
        categorie['supercategory'] = 'component'
        categorie['id'] = len(self.label) + 1  # 0 默认为背景
        categorie['name'] = label
        return categorie

    def annotation(self, points, label):
        annotation = {}
        annotation['segmentation'] = [list(np.asarray(points).flatten())]
        annotation['iscrowd'] = 0
        annotation['image_id'] = self.id
        # annotation['bbox'] = str(self.getbbox(points)) # 使用list保存json文件时报错（不知道为什么）
        # list(map(int,a[1:-1].split(','))) a=annotation['bbox'] 使用该方式转成list
        annotation['bbox'] = list(map(float, self.getbbox(points)))
        annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
        annotation['category_id'] = self.getcatid(label)
        # annotation['category_id'] = 1
        annotation['id'] = self.annID
        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return 1

    def getbbox(self, points):
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
        # cv2.fillPoly(img, [np.asarray(points)], 1)  # 画多边形 内部像素值为1
        polygons = points

        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        '''从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4, cls=MyEncoder)  # indent=4 更加美观显示

def main():
    args = parse_args()
    if args is not None:
        print 'runrunrun'
        if str(args.dir).endswith("/"):
            power_json = glob.glob(args.dir + "/*." + args.type)
        else:
            power_json = glob.glob(args.dir + "/*." + args.type)
        powerAi2coco(power_json, args.outfile)
    else:
        print 'nonono'

# power_json=glob.glob('/home/baymin/work/dockervolume/detectron_dengpao/dp/coco-json-export/*.pjson')
# power_json=['./1.json']
if __name__ == '__main__':
    main()
    # powerAi2coco(power_json, '/home/baymin/work/dockervolume/detectron_dengpao/dp/coco-json-export/final.json')
