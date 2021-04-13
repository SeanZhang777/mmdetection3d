import os
import numpy as np
import open3d as o3d
from pathlib import Path
from argparse import ArgumentParser
from mmdet3d.apis import inference_detector, init_detector

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

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('point_dir', default=None, help='Point cloud file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.35, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    point_files = os.listdir(args.point_dir)
    for file in point_files:
        # test a single image
        abs_dir = Path(args.point_dir) / file
        result, data = inference_detector(model, abs_dir)

        # show the results in open3d
        boxes = result[0]['pts_bbox']['boxes_3d'].tensor
        labels = result[0]['pts_bbox']['labels_3d']
        boxes = boxes[:, [0, 1, 2, 4, 3, 5, 6]]
        boxes[:, 6] = -boxes[:, 6] - np.pi / 2.0
        boxes[:, 2] += boxes[:, 5] / 2.0
        points = data['points'][0][0].cpu().numpy()
        draw_inference_result(points, boxes, labels, window_name=file)


if __name__ == '__main__':
    main()
