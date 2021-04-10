import numpy as np
from collections import OrderedDict
from concurrent import futures as futures
from os import path as osp
from pathlib import Path
from skimage import io
from mmdet3d.core.bbox import box_np_ops

def get_image_index_str(img_idx, use_prefix_id=False):
    if use_prefix_id:
        return img_idx
    else:
        return img_idx

def get_maxus_info_path(idx,
                        prefix,
                        info_type='image',
                        file_tail='.jpg',
                        training=True,
                        relative_path=True,
                        exist_check=True,
                        use_prefix_id=False):
    img_idx_str = get_image_index_str(idx, use_prefix_id)
    img_idx_str += file_tail
    prefix = Path(prefix)
    if training:
        file_path = Path('training') / info_type / img_idx_str
    else:
        file_path = Path('testing') / info_type / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_image_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='image',
                   use_prefix_id=False):
    return get_maxus_info_path(idx, prefix, info_type, '.jpg', training,
                               relative_path, exist_check, use_prefix_id)


def get_label_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='label',
                   use_prefix_id=False):
    return get_maxus_info_path(idx, prefix, info_type, '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_pointcloud_path(idx,
                        prefix,
                        training=True,
                        relative_path=True,
                        exist_check=True,
                        use_prefix_id=False):
    return get_maxus_info_path(idx, prefix, 'pointcloud', '.bin', training,
                               relative_path, exist_check, use_prefix_id)


def get_calib_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   use_prefix_id=False):
    return get_maxus_info_path(idx, prefix, 'calib', '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_pose_path(idx,
                  prefix,
                  training=True,
                  relative_path=True,
                  exist_check=True,
                  use_prefix_id=False):
    return get_maxus_info_path(idx, prefix, 'pose', '.txt', training,
                               relative_path, exist_check, use_prefix_id)

def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'occluded': [],
        'bbox': [],
        'box3d_camera': [],
        'box3d_lidar': [],
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['occluded'] = np.array([int(x[1]) for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[2:6]]
                                    for x in content]).reshape(-1, 4)
    # normal lidar coordinate: x, y, z, l, w, h, yaw
    gt_box3d_lidar = np.array([[float(info) for info in x[6:13]]
                              for x in content]).reshape(-1, 7)
    annotations['box3d_camera'] = box_np_ops.pcdet_box3d_lidar2camera(
                                     gt_box3d_lidar)
    # very strange coordinate defination: x, y, z, w, l, h, rotation_y(cameara)
    gt_box3d_lidar = gt_box3d_lidar[:, [0, 1, 2, 4, 3, 5, 6]]
    gt_box3d_lidar[:, 2] = gt_box3d_lidar[:, 2] - gt_box3d_lidar[:, 5] / 2.0  # bottom center
    gt_box3d_lidar[:, 6] = annotations['box3d_camera'][:, 6]
    annotations['box3d_lidar'] = gt_box3d_lidar

    if len(content) != 0 and len(content[0]) == 14:  # have score
        annotations['score'] = np.array([float(x[13]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def get_maxus_image_info(path,
                         training=True,
                         label_info=True,
                         pointcloud=False,
                         calib=False,
                         image_ids=7481,
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=True):
    """
    MAXUS annotation format version 2:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for maxus]image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {
            num_features: 4
            pointcloud_path: ...
        }
        [optional, for maxus]calib: {

        }
        annos: {
            gt_box3d_camera:
            gt_box3d_lidar:
            bbox:
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    root_path = Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        info = {}
        pc_info = {'num_features': 4}
        calib_info = {}

        image_info = {'image_idx': idx}
        annotations = None
        if pointcloud:
            pc_info['pointcloud_path'] = get_pointcloud_path(
                idx, path, training, relative_path)
        # image_info['image_path'] = get_image_path(idx, path, training, relative_path)

        if with_imageshape:
            img_path = image_info['image_path']
            if relative_path:
                img_path = str(root_path / img_path)
            image_info['image_shape'] = np.array(
                io.imread(img_path).shape[:2], dtype=np.int32)
        else:
            image_info['image_path'] = None
            image_info['image_shape'] = np.array([1280, 720])
        if label_info:
            label_path = get_label_path(idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        info['image'] = image_info
        info['point_cloud'] = pc_info
        if calib:
            info['calib'] = calib_info

        if annotations is not None:
            info['annos'] = annotations
            add_difficulty_to_annos(info)
        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)

    return list(image_infos)


def get_waymo_image_info(path,
                         training=True,
                         label_info=True,
                         velodyne=False,
                         calib=False,
                         pose=False,
                         image_ids=7481,
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=True,
                         max_sweeps=5):
    """
    Waymo annotation format version like KITTI:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for kitti]image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {
            num_features: 6
            velodyne_path: ...
        }
        [optional, for kitti]calib: {
            R0_rect: ...
            Tr_velo_to_cam0: ...
            P0: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    root_path = Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        info = {}
        pc_info = {'num_features': 6}
        calib_info = {}

        image_info = {'image_idx': idx}
        annotations = None
        if velodyne:
            pc_info['velodyne_path'] = get_velodyne_path(
                idx, path, training, relative_path, use_prefix_id=True)
            points = np.fromfile(
                Path(path) / pc_info['velodyne_path'], dtype=np.float32)
            points = np.copy(points).reshape(-1, pc_info['num_features'])
            info['timestamp'] = np.int64(points[0, -1])
            # values of the last dim are all the timestamp
        image_info['image_path'] = get_image_path(
            idx,
            path,
            training,
            relative_path,
            info_type='image_0',
            use_prefix_id=True)
        if with_imageshape:
            img_path = image_info['image_path']
            if relative_path:
                img_path = str(root_path / img_path)
            image_info['image_shape'] = np.array(
                io.imread(img_path).shape[:2], dtype=np.int32)
        if label_info:
            label_path = get_label_path(
                idx,
                path,
                training,
                relative_path,
                info_type='label_all',
                use_prefix_id=True)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        info['image'] = image_info
        info['point_cloud'] = pc_info
        if calib:
            calib_path = get_calib_path(
                idx, path, training, relative_path=False, use_prefix_id=True)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                           ]).reshape([3, 4])
            P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
                           ]).reshape([3, 4])
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                           ]).reshape([3, 4])
            P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                           ]).reshape([3, 4])
            P4 = np.array([float(info) for info in lines[4].split(' ')[1:13]
                           ]).reshape([3, 4])
            if extend_matrix:
                P0 = _extend_matrix(P0)
                P1 = _extend_matrix(P1)
                P2 = _extend_matrix(P2)
                P3 = _extend_matrix(P3)
                P4 = _extend_matrix(P4)
            R0_rect = np.array([
                float(info) for info in lines[5].split(' ')[1:10]
            ]).reshape([3, 3])
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect

            Tr_velo_to_cam = np.array([
                float(info) for info in lines[6].split(' ')[1:13]
            ]).reshape([3, 4])
            if extend_matrix:
                Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
            calib_info['P0'] = P0
            calib_info['P1'] = P1
            calib_info['P2'] = P2
            calib_info['P3'] = P3
            calib_info['P4'] = P4
            calib_info['R0_rect'] = rect_4x4
            calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
            info['calib'] = calib_info
        if pose:
            pose_path = get_pose_path(
                idx, path, training, relative_path=False, use_prefix_id=True)
            info['pose'] = np.loadtxt(pose_path)

        if annotations is not None:
            info['annos'] = annotations
            info['annos']['camera_id'] = info['annos'].pop('score')
            add_difficulty_to_annos(info)

        sweeps = []
        prev_idx = idx
        while len(sweeps) < max_sweeps:
            prev_info = {}
            prev_idx -= 1
            prev_info['velodyne_path'] = get_velodyne_path(
                prev_idx,
                path,
                training,
                relative_path,
                exist_check=False,
                use_prefix_id=True)
            if_prev_exists = osp.exists(
                Path(path) / prev_info['velodyne_path'])
            if if_prev_exists:
                prev_points = np.fromfile(
                    Path(path) / prev_info['velodyne_path'], dtype=np.float32)
                prev_points = np.copy(prev_points).reshape(
                    -1, pc_info['num_features'])
                prev_info['timestamp'] = np.int64(prev_points[0, -1])
                prev_pose_path = get_pose_path(
                    prev_idx,
                    path,
                    training,
                    relative_path=False,
                    use_prefix_id=True)
                prev_info['pose'] = np.loadtxt(prev_pose_path)
                sweeps.append(prev_info)
            else:
                break
        info['sweeps'] = sweeps

        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)

    return list(image_infos)


def kitti_anno_to_label_file(annos, folder):
    folder = Path(folder)
    for anno in annos:
        image_idx = anno['metadata']['image_idx']
        label_lines = []
        for j in range(anno['bbox'].shape[0]):
            label_dict = {
                'name': anno['name'][j],
                'alpha': anno['alpha'][j],
                'bbox': anno['bbox'][j],
                'location': anno['location'][j],
                'dimensions': anno['dimensions'][j],
                'rotation_y': anno['rotation_y'][j],
                'score': anno['score'][j],
            }
            label_line = kitti_result_line(label_dict)
            label_lines.append(label_line)
        label_file = folder / f'{get_image_index_str(image_idx)}.txt'
        label_str = '\n'.join(label_lines)
        with open(label_file, 'w') as f:
            f.write(label_str)


def add_difficulty_to_annos(info):
    annos = info['annos']
    occlusion = annos['occluded']
    diff = occlusion
    annos['difficulty'] = np.array(diff, np.int32)
    return diff


def kitti_result_line(result_dict, precision=4):
    prec_float = '{' + ':.{}f'.format(precision) + '}'
    res_line = []
    all_field_default = OrderedDict([
        ('name', None),
        ('truncated', -1),
        ('occluded', -1),
        ('alpha', -10),
        ('bbox', None),
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_y', -10),
        ('score', 0.0),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError('you must specify a value for {}'.format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == 'name':
            res_line.append(val)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == 'occluded':
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append('{}'.format(val))
        elif key in ['bbox', 'dimensions', 'location']:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError('unknown key. supported key:{}'.format(
                res_dict.keys()))
    return ' '.join(res_line)