import numpy as np
from pathlib import Path
from .read_write_model import *

def get_poses(images):
    poses = []
    for i in images:
        R = images[i].qvec2rotmat()
        t = images[i].tvec.reshape([3, 1])
        bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(w2c)
        poses.append(c2w)
    return np.array(poses)

def load_colmap_data(realdir):
    
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_cameras_binary(camerasfile)
    
    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print( 'Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h,w,f]).reshape([3,1])
    
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_images_binary(imagesfile)
    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    names = [imdata[k].name for k in imdata]
    print( 'Images #', len(names))
    perm = np.argsort(names)
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    
    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    
    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = read_points3d_binary(points3dfile)
    
    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    
    return poses, pts3d, perm

def save_poses(poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            print(ind)
            if len(cams) < ind - 1:
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            cams[ind-1] = 1
        vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print( 'Points', pts_arr.shape, 'Visibility', vis_arr.shape )
    
    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr==1]
    print( 'Depth stats', valid_z.min(), valid_z.max(), valid_z.mean() )
    
    save_arr = []
    for i in perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis==1]
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
        
        save_arr.append(np.array([close_depth, inf_depth]))
    save_arr = np.array(save_arr)

    return save_arr

def load_colmap_depth(basedir, sc=1.0):
    images = read_images_binary(Path(basedir) / 'sparse' / '0' / 'images.bin')
    points = read_points3d_binary(Path(basedir) / 'sparse' / '0' / 'points3D.bin')

    poses = get_poses(images)

    r1, r2, r3 = load_colmap_data(basedir)
    poses_arr = save_poses(r1, r2, r3)
    bds_raw = poses_arr.transpose([1, 0])
    bds_raw = np.moveaxis(bds_raw, -1, 0).astype(np.float32)

    data_list = []
    for id_im in range(1, len(images) + 1):
        depth_list = []
        coord_list = []
        for i in range(len(images[id_im].xys)):
            point2D = images[id_im].xys[i]
            id_3D = images[id_im].point3D_ids[i]
            if id_3D == -1:
                continue
            point3D = points[id_3D].xyz
            depth = (poses[id_im - 1, :3, 2].T @
                     (point3D - poses[id_im - 1, :3, 3])) * sc
            if depth < bds_raw[id_im - 1, 0] * sc or depth > bds_raw[id_im - 1, 1] * sc:
                continue
            depth_list.append(depth)
            coord_list.append(point2D)
        data_list.append({"depth": np.array(depth_list), "coord": np.array(coord_list)})

    return data_list

def load_object_points(basedir, masks):

    images = read_images_binary(Path(basedir) / 'sparse' / '0' / 'images.bin')
    points = read_points3d_binary(Path(basedir) / 'sparse' / '0' / 'points3D.bin')

    H, W = masks[0].shape[:2]
    object_points = []
    for point3d_id in points:
        point3d = points[point3d_id]
        is_object_point = True
        for im_id, point2d_id in zip(point3d.image_ids, point3d.point2D_idxs):
            point2d = images[im_id].xys[point2d_id]
            x, y = point2d.astype(np.int32)
            x, y = min(x, W-1), min(y, H-1)
            if masks[im_id-1][y, x] == 0:
                is_object_point = False
        if is_object_point:
            object_points.append(point3d.xyz)
    object_points = np.stack(object_points, 0)

    return object_points