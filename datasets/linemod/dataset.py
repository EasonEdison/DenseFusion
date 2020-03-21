import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json
import time
import random
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import yaml
import cv2


class PoseDataset(data.Dataset):
    def __init__(self, mode, num, add_noise, root, noise_trans, refine):
        self.objlist = [1]
        # self.objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        self.mode = mode

        self.list_rgb = []
        self.list_depth = []
        self.list_label = []
        self.list_obj = []
        self.list_rank = []
        self.meta = {}
        self.pt = {}
        self.root = root
        self.noise_trans = noise_trans
        self.refine = refine

        item_count = 0
        for item in self.objlist:
            if self.mode == 'train':
                input_file = open('{0}/data/{1}/train.txt'.format(self.root, '%02d' % item))
            else:
                input_file = open('{0}/data/{1}/test.txt'.format(self.root, '%02d' % item))
            while 1:
                item_count += 1
                input_line = input_file.readline()
                if self.mode == 'test' and item_count % 10 != 0:
                    continue
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                self.list_rgb.append('{0}/data/{1}/rgb/{2}.png'.format(self.root, '%02d' % item, input_line))
                self.list_depth.append('{0}/data/{1}/depth/{2}.png'.format(self.root, '%02d' % item, input_line))
                if self.mode == 'eval':
                    self.list_label.append('{0}/segnet_results/{1}_label/{2}_label.png'.format(self.root, '%02d' % item, input_line))
                else:
                    self.list_label.append('{0}/data/{1}/mask/{2}.png'.format(self.root, '%02d' % item, input_line))
                # use index to get object class
                # 使用index时能获取物体类别
                self.list_obj.append(item)
                # picture name, such 0000 0001
                # can use index to get name of the item(use list_obj to get class)
                # 存储图片的名字，同样使用index就能得到，配合着list_obj，就能知道详细信息
                self.list_rank.append(int(input_line))
            # ground truth R T bbox class
            # R，T bbox，类别的真实值
            meta_file = open('{0}/data/{1}/gt.yml'.format(self.root, '%02d' % item), 'r')
            # with list_rank get item ground truth infor
            # 配合着list_rank，就能知道具体信息
            self.meta[item] = yaml.load(meta_file)
            # ply infor
            # 3D模型点云文件
            self.pt[item] = ply_vtx('{0}/models/obj_{1}.ply'.format(self.root, '%02d' % item))
            
            print("Object {0} buffer loaded".format(item))

        self.length = len(self.list_rgb)

        self.cam_cx = 325.26110
        self.cam_cy = 242.04899
        self.cam_fx = 572.41140
        self.cam_fy = 573.57043
        # 做一个网格，用来提取mask的坐标
        # row is same, column from 0 to 479
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        # row from 0 to 639, column is same
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        
        self.num = num
        self.add_noise = add_noise
        # torchvision自动增强
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # 这个不懂
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.num_pt_mesh_large = 500
        self.num_pt_mesh_small = 500
        self.symmetry_obj_idx = [7, 8]

    def __getitem__(self, index):
        img = Image.open(self.list_rgb[index])
        img.save('/home/ouc/TXH/DenseFusion/visual_refine/ori.jpg')
        ori_img = np.array(img)
        depth = np.array(Image.open(self.list_depth[index]))
        label = np.array(Image.open(self.list_label[index]))
        obj = self.list_obj[index]
        rank = self.list_rank[index]        

        if obj == 2:
            for i in range(0, len(self.meta[obj][rank])):
                if self.meta[obj][rank][i]['obj_id'] == 2:
                    meta = self.meta[obj][rank][i]
                    break
        else:
            meta = self.meta[obj][rank][0]

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        if self.mode == 'eval':
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
        else:
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([255, 255, 255])))[:, :, 0]
        # 看来255是被选定的mask，就是白色
        # 只有True * True才是True
        # for what? for choose
        mask = mask_label * mask_depth

        if self.add_noise:
            img = self.trancolor(img)

        img = np.array(img)[:, :, :3]
        # channel in first
        img = np.transpose(img, (2, 0, 1))
        img_masked = img

        # get bbox是设置边界的函数？
        # 我觉得可能是把物体抠出来，但是borde_list有什么用我不清楚
        # NUS, about border_list?
        if self.mode == 'eval':
            rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label))
        else:
            rmin, rmax, cmin, cmax = get_bbox(meta['obj_bb'])

        img_masked = img_masked[:, rmin:rmax, cmin:cmax]
        #p_img = np.transpose(img_masked, (1, 2, 0))
        #scipy.misc.imsave('evaluation_result/{0}_input.png'.format(index), p_img)

        target_r = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
        target_t = np.array(meta['cam_t_m2c'])
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])
        # 只要限定范围内的物体
        # get infor in bbox
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) == 0:
            # 把narray转换成Long类型Tensor的操作
            cc = torch.LongTensor([0])
            return(cc, cc, cc, cc, cc, cc)

        # 采样的点数
        if len(choose) > self.num:
            # 先设置所有的0点，然后在其中选择num个1点，随机打乱顺序，再用nonzero转换成True，False，达到mask index的效果
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            # 右边拿0个添到左边，左边拿self.num - len(choose)个填到右边
            # 不知道为什么使用这种方法
            choose = np.pad(choose, (0, self.num - len(choose)), 'wrap')

        # 在限定的范围内做mask选择，就是在bbox中选mask了
        # 获取一一对应的x y depth信息，然后通过内参获得带你云
        # why flatten?
        # nice! get x y depth, one to one correspondence, so get point cloud
        # but not target?
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])
        d2d = np.concatenate((xmap_masked, ymap_masked),axis=1)
        np.savetxt('/home/ouc/TXH/DenseFusion/visual_refine/2d_rigid.txt', d2d)
        # get x y z
        cam_scale = 1.0
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        cloud = cloud / 1000.0

        if self.add_noise:
            # 加就完事了，为啥还得np.add?
            cloud = np.add(cloud, add_t)

        #fw = open('evaluation_result/{0}_cld.xyz'.format(index), 'w')
        #for it in cloud:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        #fw.close()

        # 这是模型的点云
        model_points = self.pt[obj] / 1000.0
        dellist = [j for j in range(0, len(model_points))]
        # 采样 稀疏一下 只要500个点
        # why not direct sample self.num_pt_mesh_small ?
        dellist = random.sample(dellist, len(model_points) - self.num_pt_mesh_small)
        model_points = np.delete(model_points, dellist, axis=0)

        #fw = open('evaluation_result/{0}_model_points.xyz'.format(index), 'w')
        #for it in model_points:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        #fw.close()

        # convert model point to camera frame
        # 将model的点云转到相机坐标系下
        # 注意这里的方法，先右乘，就是自身旋转，然后加上位移
        target = np.dot(model_points, target_r.T)
        if self.add_noise:
            target = np.add(target, target_t / 1000.0 + add_t)
            out_t = target_t / 1000.0 + add_t
        else:
            target = np.add(target, target_t / 1000.0)
            out_t = target_t / 1000.0

        #fw = open('evaluation_result/{0}_tar.xyz'.format(index), 'w')
        #for it in target:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        #fw.close()
        np.savetxt('/home/ouc/TXH/DenseFusion/visual_refine/cloud.txt', cloud.astype(np.float32))
        np.savetxt('/home/ouc/TXH/DenseFusion/visual_refine/target.txt', target.astype(np.float32))
        # reutrn point_cloud, mask, image_norm, target, model_point, obj_index
        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([self.objlist.index(obj)])

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small



border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640


def mask_to_bbox(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x = 0
    y = 0
    w = 0
    h = 0
    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x = tmp_x
            y = tmp_y
            w = tmp_w
            h = tmp_h
    return [x, y, w, h]


def get_bbox(bbox):
    # index of row is bbox's height(y)
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax


def ply_vtx(path):
    f = open(path)
    assert f.readline().strip() == "ply"
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)
