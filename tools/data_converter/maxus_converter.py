import mmcv
import numpy as np
from pathlib import Path

from mmdet3d.core.bbox import box_np_ops
from .maxus_data_utils import get_maxus_image_info

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def _calculate_num_points_in_gt(data_path,
                                infos,
                                relative_path,
                                remove_outside=True,
                                num_features=4):
    for info in mmcv.track_iter_progress(infos):
        pc_info = info['point_cloud']
        image_info = info['image']
        calib = info['calib']
        if relative_path:
            v_path = str(Path(data_path) / pc_info['pointcloud_path'])
        else:
            v_path = pc_info['pointcloud_path']
        points_v = np.fromfile(
            v_path, dtype=np.float32, count=-1).reshape([-1, num_features])

        # points_v = points_v[points_v[:, 0] > 0]
        annos = info['annos']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        # annos = kitti.filter_kitti_anno(annos, ['DontCare'])
        gt_boxes_lidar = annos['box3d_lidar']
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos['box3d_lidar']) - num_obj
        num_points_in_gt = np.concatenate(
            [num_points_in_gt, -np.ones([num_ignored])])
        annos['num_points_in_gt'] = num_points_in_gt.astype(np.int32)


def create_maxus_info_file(data_path,
                           pkl_prefix='maxus',
                           save_path=None,
                           relative_path=True):
    """Create info file of MAXUS dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str): Prefix of the info file to be generated.
        save_path (str): Path to save the info file.
        relative_path (bool): Whether to use relative path.
    """
    imageset_folder = Path(data_path) / 'ImageSets'
    train_img_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
    val_img_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    maxus_infos_train = get_maxus_image_info(
        data_path,
        training=True,
        pointcloud=True,
        calib=True,
        image_ids=train_img_ids,
        relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, maxus_infos_train, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'Maxus info train file is saved to {filename}')
    mmcv.dump(maxus_infos_train, filename)
    maxus_infos_val = get_maxus_image_info(
        data_path,
        training=True,
        pointcloud=True,
        calib=True,
        image_ids=val_img_ids,
        relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, maxus_infos_val, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'Maxus info val file is saved to {filename}')
    mmcv.dump(maxus_infos_val, filename)
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'Maxus info trainval file is saved to {filename}')
    mmcv.dump(maxus_infos_train + maxus_infos_val, filename)
