import argparse
import torch
import imageio
import skimage.transform
import torchvision
import numpy as np
import os
from torch.utils.data import DataLoader
import datetime
import cv2

import torch.optim
import ACNet_data_nyuv2_eval as ACNet_data
# import ACNet_models
import ACNet_models_V1
# import ACNet_models_V1_first as ACNet_models_V1
#import ACNet_models_V1_delA as ACNet_models_V1
from utils import utils
from utils.utils import load_ckpt, intersectionAndUnion, AverageMeter, accuracy, macc

parser = argparse.ArgumentParser(description='RGBD Sementic Segmentation')
parser.add_argument('--data-dir', default=None, metavar='DIR',
                    help='path to dataset')
parser.add_argument('-o', '--output', default='./result/', metavar='DIR',
                    help='path to output')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--num-class', default=40, type=int,
                    help='number of classes')
parser.add_argument('--visualize', default=False, action='store_true',
                    help='if output image')

args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
image_w = 640
image_h = 480
img_mean=[0.485, 0.456, 0.406]
img_std=[0.229, 0.224, 0.225]

# transform
class scaleNorm(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # Bi-linear
        image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                         mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        image = image.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype(np.float)
        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float(),
                'label': torch.from_numpy(label).float()}

class Normalize(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        origin_image = image.clone()
        origin_depth = depth.clone()
        image = image / 255
        image = torchvision.transforms.Normalize(mean=[0.4850042694973687, 0.41627756261047333, 0.3981809741523051],
                                                 std=[0.26415541082494515, 0.2728415392982039, 0.2831175140191598])(image)
        depth = torchvision.transforms.Normalize(mean=[2.8424503515351494],
                                                 std=[0.9932836506164299])(depth)
        sample['origin_image'] = origin_image
        sample['origin_depth'] = origin_depth
        sample['image'] = image
        sample['depth'] = depth

        return sample

# def visualize_result(img, label, preds, info, args):
#     # segmentation
#     img = img.squeeze(0).transpose(0, 2, 1)
#     seg_color = utils.color_label_eval(label)
#
#     # prediction
#     pred_color = utils.color_label_eval(preds)
#
#     # aggregate images and save
#     im_vis = np.concatenate((img, seg_color, pred_color),
#                             axis=1).astype(np.uint8)
#     im_vis = im_vis.transpose(2, 1, 0)
#
#     img_name = str(info)
#     # print('write check: ', im_vis.dtype)
#     cv2.imwrite(os.path.join(args.output,
#                 img_name+'.png'), im_vis)
def visualize_result(img, depth, label, preds, info, args):
    # segmentation
    img = img.squeeze(0).transpose(0, 2, 1)
    dep = depth.squeeze(0).squeeze(0)
    dep = (dep*255/dep.max()).astype(np.uint8)
    dep = cv2.applyColorMap(dep, cv2.COLORMAP_JET)
    dep = dep.transpose(2,1,0)
    seg_color = utils.color_label_eval(label)
    # prediction
    pred_color = utils.color_label_eval(preds)

    # aggregate images and save
    im_vis = np.concatenate((img, dep, seg_color, pred_color),
                            axis=1).astype(np.uint8)
    im_vis = im_vis.transpose(2, 1, 0)

    img_name = str(info)
    # print('write check: ', im_vis.dtype)
    cv2.imwrite(os.path.join(args.output,
                img_name+'.png'), im_vis)

def inference():

    model = ACNet_models_V1.ACNet(num_class=40, pretrained=False)
    load_ckpt(model, None, args.last_ckpt, device)
    model.eval()
    model.to(device)

    val_data = ACNet_data.SUNRGBD(transform=torchvision.transforms.Compose([scaleNorm(),
                                                                   ToTensor(),
                                                                   Normalize()]),
                                   phase_train=False,
                                   data_dir=args.data_dir
                                   )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False,num_workers=1, pin_memory=True)

    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    a_meter = AverageMeter()
    b_meter = AverageMeter()
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            #todo batch=1，这里要查看sample的size，决定怎么填装image depth label，估计要用到for循环
            origin_image = sample['origin_image'].numpy()
            origin_depth = sample['origin_depth'].numpy()
            image = sample['image'].to(device)
            depth = sample['depth'].to(device)
            label = sample['label'].numpy()

            with torch.no_grad():
                pred = model(image, depth)

            output = torch.max(pred, 1)[1] + 1
            output = output.squeeze(0).cpu().numpy()

            acc, pix = accuracy(output, label)
            intersection, union = intersectionAndUnion(output, label, args.num_class)
            acc_meter.update(acc, pix)
            a_m, b_m = macc(output, label, args.num_class)
            intersection_meter.update(intersection)
            union_meter.update(union)
            a_meter.update(a_m)
            b_meter.update(b_m)
            print('[{}] iter {}, accuracy: {}'
                  .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                          batch_idx, acc))

            # img = image.cpu().numpy()
            # print('origin iamge: ', type(origin_image))
            if args.visualize:
                visualize_result(origin_image, origin_depth, label-1, output-1, batch_idx, args)

    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {}'.format(i, _iou))

    mAcc = (a_meter.average() / (b_meter.average()+1e-10))
    print(mAcc.mean())
    print('[Eval Summary]:')
    print('Mean IoU: {:.4}, Accuracy: {:.2f}%'
          .format(iou.mean(), acc_meter.average() * 100))
        # imageio.imsave(args.output, output.cpu().numpy().transpose((1, 2, 0)))

if __name__ == '__main__':
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    inference()


