import argparse
import os.path as osp
import types
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import DictAction
from mmcv.runner import force_fp32, auto_fp16
from mmdet.core import multi_apply
from mmdet3d.core import bbox3d2result
from mmdet3d.apis import inference_detector, init_detector
from mmdet3d.models.voxel_encoders.utils import get_paddings_indicator
import open3d as o3d

def draw_box_in_3d(boxes, labels):
    #assert len(boxes) == len(labels), "boxes size not equal to labels size"
    obj_color = [[0, 1, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 0, 0]]
    obbs = []
    for i in range(len(boxes)):
        box = boxes[i]
        label = labels[i]
        color = obj_color[label]
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, box[6]])
        obb = o3d.geometry.OrientedBoundingBox(center=box[:3], R=rot, extent=box[[3, 4, 5]])
        obb.color = color
        obbs.append(obb)
    return obbs

def draw_inference_result(points, boxes=None, labels=None,
                          window_name="Open3D", color=[0, 0, 0]):
    pxyz = points[:, :3]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pxyz))
    colors = np.array(color)
    colors = np.tile(colors, (pxyz.shape[0], 1))
    pc.colors = o3d.utility.Vector3dVector(colors)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    draw_list = [axis, pc]
    if boxes is not None:
        obbs = draw_box_in_3d(boxes, labels)
        draw_list.extend(obbs)
    o3d.visualization.draw_geometries(draw_list, window_name=window_name)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def gather_point_features(model, features, num_points, coors):
    vfe = model.pts_voxel_encoder
    features_ls = [features]
    # Find distance of x, y, and z from cluster center
    if vfe._with_cluster_center:
        points_mean = (
            features[:, :, :3].sum(dim=1, keepdim=True) /
            num_points.type_as(features).view(-1, 1, 1))
        # TODO: maybe also do cluster for reflectivity
        f_cluster = features[:, :, :3] - points_mean
        features_ls.append(f_cluster)

    # Find distance of x, y, and z from pillar center
    if vfe._with_voxel_center:
        f_center = features.new_zeros(
            size=(features.size(0), features.size(1), 3))
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 3].type_as(features).unsqueeze(1) * vfe.vx +
            vfe.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].type_as(features).unsqueeze(1) * vfe.vy +
            vfe.y_offset)
        f_center[:, :, 2] = features[:, :, 2] - (
            coors[:, 1].type_as(features).unsqueeze(1) * vfe.vz +
            vfe.z_offset)
        features_ls.append(f_center)

    if vfe._with_distance:
        points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
        features_ls.append(points_dist)

    # Combine together feature decorations
    voxel_feats = torch.cat(features_ls, dim=-1)
    # The feature decorations were calculated without regard to whether
    # pillar was empty.
    # Need to ensure that empty voxels remain set to zeros.
    voxel_count = voxel_feats.shape[1]
    mask = get_paddings_indicator(num_points, voxel_count, axis=0)
    voxel_feats *= mask.unsqueeze(-1).type_as(voxel_feats)

    return voxel_feats

@force_fp32(out_fp16=True)
def vfe_forward(self,
            voxel_feats,
            img_feats=None,
            img_metas=None):
    """Forward functions.

    Args:
        voxel_feats (torch.Tensor): Features of voxels, shape is MxNxC.
        img_feats (list[torch.Tensor], optional): Image fetures used for
            multi-modality fusion. Defaults to None.
        img_metas (dict, optional): [description]. Defaults to None.

    Returns:
        tuple: If `return_point_feats` is False, returns voxel features and
            its coordinates. If `return_point_feats` is True, returns
            feature of each points inside voxels.
    """
    voxel_feats = torch.squeeze(voxel_feats, 1)
    for i, vfe in enumerate(self.vfe_layers):
        voxel_feats = vfe(voxel_feats)

    return voxel_feats

def anchor_head_forward(self, feats):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple[list[torch.Tensor]]: Multi-level class score, bbox \
                and direction predictions.
        """
        conv_cls_list = []
        conv_box_list = []
        conv_dir_list = []
        for feat in feats:
            conv_cls, conv_box, conv_dir = self.forward_single(feat)
            # conv_cls_list.append(conv_cls.permute(0, 2, 3, 1).reshape(-1, 4))
            # conv_box_list.append(conv_box.permute(0, 2, 3, 1).reshape(-1, 7))
            # conv_dir_list.append(conv_dir.permute(0, 2, 3, 1).reshape(-1, 2))
            conv_cls_list.append(conv_cls.reshape(1, 32, -1).permute(0, 2, 1))
            conv_box_list.append(conv_box.reshape(1, 56, -1).permute(0, 2, 1))
            conv_dir_list.append(conv_dir.reshape(1, 16, -1).permute(0, 2, 1))
        conv_cls = torch.cat(conv_cls_list, dim=1)
        conv_box = torch.cat(conv_box_list, dim=1)
        conv_dir = torch.cat(conv_dir_list, dim=1)

        return (conv_cls, conv_box, conv_dir)

def simple_test(self, points, img_metas, img=None, rescale=False):
    """Extract features of images."""
    img_feats = self.extract_img_feat(img, img_metas)

    """Extract features of points."""
    if not self.with_pts_bbox:
        return None
    voxels, num_points, coors = self.voxelize(points)
    gathered_features = gather_point_features(self, voxels, num_points, coors)

    num_voxels = voxels.shape[0]
    pad_gather_features = torch.zeros((16000 - num_voxels, 100, 10), dtype=torch.float32).cuda()
    gathered_features = torch.cat([gathered_features, pad_gather_features], dim=0)
    pad_coors = torch.zeros((16000 - num_voxels, 4), dtype=torch.int32).cuda()
    coors = torch.cat([coors, pad_coors], dim=0)

    gathered_features = torch.unsqueeze(gathered_features, 1)
    pfe_input = (gathered_features,)
    torch.onnx.export(
        self.pts_voxel_encoder,
        pfe_input,
        "pfe.onnx",
        input_names=['gathered_features'],
        output_names=['pfe_features'],
        # dynamic_axes={'voxels': {0: 'num_voxels'},
        #               'num_points': {0: 'num_voxels'},
        #               'coors': {0: 'num_voxels'}},
        export_params=True,
        verbose=True,
        opset_version=11,
        do_constant_folding=True)

    # check pfe onnx model
    pfe_model = onnx.load("pfe.onnx")
    onnx.checker.check_model(pfe_model)

    pfe_ort_session = ort.InferenceSession("pfe.onnx")
    pfe_ort_inputs = {pfe_ort_session.get_inputs()[0].name: to_numpy(gathered_features)}
    pfe_ort_outs = pfe_ort_session.run(None, pfe_ort_inputs)

    voxel_features = self.pts_voxel_encoder(gathered_features,
                                            img_feats, img_metas)
    np.testing.assert_allclose(to_numpy(voxel_features), pfe_ort_outs[0], rtol=1e-01, atol=1e-03)
    print('pfe model check OK!')

    batch_size = coors[-1, 0] + 1
    scatter_features = self.pts_middle_encoder(voxel_features, coors, batch_size)
    #scatter_features = np.loadtxt("/home/maxus/test/tensor_check/scatter.txt").reshape(1, 64, 320, 600)
    #scatter_features = torch.Tensor(scatter_features).cuda()
    backbone_features = self.pts_backbone(scatter_features)
    neck_features = self.pts_neck(backbone_features)
    outs = self.pts_bbox_head(neck_features)
    rpn = nn.Sequential(self.pts_backbone, self.pts_neck, self.pts_bbox_head)
    #scatter_features = torch.zeros([1, 64, 400, 400], device=scatter_features.device)
    rpn_input = (scatter_features, )
    torch.onnx.export(
        rpn,
        rpn_input,
        "rpn.onnx",
        input_names=['scatter_features'],
        output_names=['conv_cls', 'conv_box', 'conv_dir'],
        export_params=True,
        verbose=True,
        opset_version=11,
        do_constant_folding=True)

    # check rpn onnx model
    rpn_model = onnx.load("rpn.onnx")
    onnx.checker.check_model(rpn_model)

    rpn_ort_session = ort.InferenceSession("rpn.onnx")
    rpn_ort_inputs = {rpn_ort_session.get_inputs()[0].name: to_numpy(scatter_features)}
    rpn_ort_outs = rpn_ort_session.run(None, rpn_ort_inputs)

    np.testing.assert_allclose(to_numpy(outs[0]), rpn_ort_outs[0], rtol=1e-01, atol=1e-03)
    np.testing.assert_allclose(to_numpy(outs[1]), rpn_ort_outs[1], rtol=1e-01, atol=1e-03)
    np.testing.assert_allclose(to_numpy(outs[2]), rpn_ort_outs[2], rtol=1e-01, atol=1e-03)
    print('rpn model check OK!')

    # conv_cls_all = np.loadtxt("/home/maxus/test/tensor_check/conv_cls_ort.txt", dtype=np.float32).reshape(-1, 32)
    # conv_box_all = np.loadtxt("/home/maxus/test/tensor_check/conv_box_ort.txt", dtype=np.float32).reshape(-1, 56)
    # conv_dir_all = np.loadtxt("/home/maxus/test/tensor_check/conv_dir_ort.txt", dtype=np.float32).reshape(-1, 16)
    # conv_cls_all = outs[0].cpu().numpy().reshape(-1, 32)
    # conv_box_all = outs[1].cpu().numpy().reshape(-1, 56)
    # conv_dir_all = outs[2].cpu().numpy().reshape(-1, 16)
    conv_cls_all = rpn_ort_outs[0].reshape(-1, 32)
    conv_box_all = rpn_ort_outs[1].reshape(-1, 56)
    conv_dir_all = rpn_ort_outs[2].reshape(-1, 16)
    np.savetxt("/home/maxus/test/tensor_check/scatter.txt", scatter_features.cpu().numpy().reshape(64, -1), fmt='%.10f')
    np.savetxt("/home/maxus/test/tensor_check/conv_cls_ort.txt", conv_cls_all, fmt='%.10f')
    np.savetxt("/home/maxus/test/tensor_check/conv_box_ort.txt", conv_box_all, fmt='%.10f')
    np.savetxt("/home/maxus/test/tensor_check/conv_dir_ort.txt", conv_dir_all, fmt='%.10f')
    conv_cls1 = torch.Tensor(conv_cls_all[:160*300, :].reshape(1, 160, 300, 32)).permute(0, 3, 1, 2)
    conv_cls2 = torch.Tensor(conv_cls_all[160 * 300:160 * 300+80*150, :].reshape(1, 80, 150, 32)).permute(0, 3, 1, 2)
    conv_cls3 = torch.Tensor(conv_cls_all[160 * 300+80*150:, :].reshape(1, 40, 75, 32)).permute(0, 3, 1, 2)
    conv_box1 = torch.Tensor(conv_box_all[:160*300, :].reshape(1, 160, 300, 56)).permute(0, 3, 1, 2)
    conv_box2 = torch.Tensor(conv_box_all[160 * 300:160 * 300+80*150, :].reshape(1, 80, 150, 56)).permute(0, 3, 1, 2)
    conv_box3 = torch.Tensor(conv_box_all[160 * 300+80*150:, :].reshape(1, 40, 75, 56)).permute(0, 3, 1, 2)
    conv_dir1 = torch.Tensor(conv_dir_all[:160*300, :].reshape(1, 160, 300, 16)).permute(0, 3, 1, 2)
    conv_dir2 = torch.Tensor(conv_dir_all[160 * 300:160 * 300+80*150, :].reshape(1, 80, 150, 16)).permute(0, 3, 1, 2)
    conv_dir3 = torch.Tensor(conv_dir_all[160 * 300+80*150:, :].reshape(1, 40, 75, 16)).permute(0, 3, 1, 2)

    outs = ([conv_cls1.cuda(), conv_cls2.cuda(), conv_cls3.cuda()],
            [conv_box1.cuda(), conv_box2.cuda(), conv_box3.cuda()],
            [conv_dir1.cuda(), conv_dir2.cuda(), conv_dir3.cuda()])
    bbox_list = [dict() for i in range(len(img_metas))]
    bboxes = self.pts_bbox_head.get_bboxes(
        *outs, img_metas, rescale=rescale)
    bbox_results = [
        bbox3d2result(bboxes, scores, labels)
        for bboxes, scores, labels in bboxes
    ]
    for result_dict, pts_bbox in zip(bbox_list, bbox_results):
        result_dict['pts_bbox'] = pts_bbox
    if img_feats and self.with_img_bbox:
        bbox_img = self.simple_test_img(
            img_feats, img_metas, rescale=rescale)
        for result_dict, img_bbox in zip(bbox_list, bbox_img):
            result_dict['img_bbox'] = img_bbox
    return bbox_list

def pytorch2onnx(config_path,
                 checkpoint_path,
                 input_pts):
    # build the model from a config file and a checkpoint file
    model = init_detector(config_path, checkpoint_path, device='cuda:0')
    model.simple_test = types.MethodType(simple_test, model)
    model.pts_voxel_encoder.forward = types.MethodType(vfe_forward, model.pts_voxel_encoder)
    model.pts_bbox_head.forward = types.MethodType(anchor_head_forward, model.pts_bbox_head)
    result, data = inference_detector(model, input_pts)
    # show the results in open3d
    boxes = result[0]['pts_bbox']['boxes_3d'].tensor
    labels = result[0]['pts_bbox']['labels_3d']
    boxes = boxes[:, [0, 1, 2, 4, 3, 5, 6]]
    boxes[:, 6] = -boxes[:, 6] - np.pi / 2.0
    boxes[:, 2] += boxes[:, 5] / 2.0
    points = data['points'][0][0].cpu().numpy()
    draw_inference_result(points, boxes, labels)
    pass

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection3d models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_pts', type=str, help='Point Cloud for input')
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--test-pts', type=str, default=None, help='Point Cloud for test')
    parser.add_argument(
        '--dataset', type=str, default='maxus', help='Dataset name')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Whether to simplify onnx model.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    assert args.opset_version == 11, 'MMDet only support opset 11 now'

    # convert model to onnx file
    pytorch2onnx(
        args.config,
        args.checkpoint,
        args.input_pts)
