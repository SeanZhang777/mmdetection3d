import argparse
import time
import types
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from mmcv.runner import wrap_fp16_model
from tools.misc.fuse_conv_bn import fuse_module
from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--samples', default=2000, help='samples to benchmark')
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args

def extract_pts_feat(self, pts, img_feats, img_metas):
    """Extract features of points."""
    if not self.with_pts_bbox:
        return None
    torch.cuda.synchronize()
    start = time.perf_counter()
    voxels, num_points, coors = self.voxelize(pts)
    end = time.perf_counter()
    print(f'voxelize cost: {(end - start) * 1000} ms.')

    torch.cuda.synchronize()
    start = time.perf_counter()
    voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                            img_feats, img_metas)
    end = time.perf_counter()
    print(f'voxel_encoder cost: {(end - start) * 1000} ms.')

    batch_size = coors[-1, 0] + 1
    torch.cuda.synchronize()
    start = time.perf_counter()
    x = self.pts_middle_encoder(voxel_features, coors, batch_size)
    end = time.perf_counter()
    print(f'middle_encoder cost: {(end - start) * 1000} ms.')

    torch.cuda.synchronize()
    start = time.perf_counter()
    x = self.pts_backbone(x)
    end = time.perf_counter()
    print(f'backbone cost: {(end - start) * 1000} ms.')

    if self.with_pts_neck:
        torch.cuda.synchronize()
        start = time.perf_counter()
        x = self.pts_neck(x)
        end = time.perf_counter()
        print(f'neck cost: {(end - start) * 1000} ms.')
    return x

def simple_test_pts(self, x, img_metas, rescale=False):
    """Test function of point cloud branch."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    outs = self.pts_bbox_head(x)
    end = time.perf_counter()
    print(f'head cost: {(end - start) * 1000} ms.')

    torch.cuda.synchronize()
    start = time.perf_counter()
    bbox_list = self.pts_bbox_head.get_bboxes(
        *outs, img_metas, rescale=rescale)
    end = time.perf_counter()
    print(f'post cost: {(end - start) * 1000} ms.')

    bbox_results = [
        bbox3d2result(bboxes, scores, labels)
        for bboxes, scores, labels in bbox_list
    ]
    return bbox_results

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.extract_pts_feat = types.MethodType(extract_pts_feat, model)
    model.simple_test_pts = types.MethodType(simple_test_pts, model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_module(model)

    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0

    # benchmark with several samples and take the average
    for i, data in enumerate(data_loader):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Done image [{i + 1:<3}/ {args.samples}], '
                      f'fps: {fps:.1f} img / s')

        if (i + 1) == args.samples:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.1f} img / s')
            break


if __name__ == '__main__':
    main()
