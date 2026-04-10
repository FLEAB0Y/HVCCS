import numpy as np
import json
import cv2
import pdb
import os
import mapping as mp
import vis
import hashlib


def get_cam_param(cam_param_path, json_path):
    with open(cam_param_path, "r") as f:
        cam_param = json.load(f)
    with open(json_path, "r") as f:
        data = json.load(f)
    cam_param = cam_param[data['cam']]

    return cam_param

def get_marker_labels(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data['keypoint_name']

def project_2d(keypoint_data, cam_param, frame, undistort=False, adj_cam=False,rotate=False):
    """
    Projects 3D keypoints onto a 2D image plane based on camera parameters.

    Args:
        keypoint_data (list): A list of 3D keypoints (N x 3 array) to be projected.
        cam_param (dict): Camera parameters, including intrinsic, extrinsic matrices, and distortion coefficients.
        frame (ndarray): The image frame on which to draw the projected keypoints.
        undistort (bool): Flag indicating whether to undistort the points. Default is False.
        adj_cam (bool): Flag to adjust the camera position (scale it to meters). Default is False.

    Returns:
        ndarray: The frame with the projected keypoints drawn on it.
    """
    fu = cam_param["affine_intrinsics_matrix"][0][0]
    fv = cam_param["affine_intrinsics_matrix"][1][1]
    cu = cam_param["affine_intrinsics_matrix"][0][2]
    cv = cam_param["affine_intrinsics_matrix"][1][2]
    affine_intrinsics_matrix = np.array(cam_param["affine_intrinsics_matrix"])

    rot_mat = np.array(cam_param["extrinsic_matrix"])
    rot_mat[1:, :] *= -1

    camera_position = np.array(cam_param["xyz"])
    #divided by 1000 to convert to meters
    camera_position = camera_position/1000 if adj_cam else camera_position

    distortion = np.array(cam_param["distortion"])
    
    for i in range(len(keypoint_data)):
        keypoints = keypoint_data[i]

        # World to cam
        translated = keypoints[0:3] - camera_position
        kpts_camera = (rot_mat @ translated.T).T

        # Cam to pixel
        Xc = kpts_camera[0]
        Yc = kpts_camera[1]
        Zc = kpts_camera[2]
        u = fu * (Xc / Zc) + cu
        v = fv * (Yc / Zc) + cv
        uv = np.stack([u, v], axis=-1)

        if undistort:
            # Undistort the points
            uv = cv2.undistortPoints(np.expand_dims(uv, axis=0), affine_intrinsics_matrix, distortion, None, affine_intrinsics_matrix)

            # Convert from normalized coordinates to pixel coordinates
            uv = uv.squeeze(axis=0)
            u, v = uv[0][0], uv[0][1]

        if rotate:
            #rotate the points
            u,v = v,u
            #flip the y
            v = frame.shape[0] - v

        #check if u and v are is nan
        if np.isnan(u) or np.isnan(v):
            continue

        # Draw the keypoints on the frame
        print(u,v)
        frame = cv2.circle(frame, (int(u), int(v)), 5, get_color(i), -1)
        
        # Add text at the keypoints
        frame = cv2.putText(frame, str(i), (int(u), int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return frame

def get_color(i):
    """
    Generate a unique color based on the input number. (Same color for the same number)

    Args:
        i (int): The input number to generate a color for.

    Returns:
        tuple: A tuple representing the color in BGR format (blue, green, red).
    """
    hash_value = int(hashlib.md5(str(i).encode()).hexdigest(), 16)  # Hash the number
    r = (hash_value % 256)  
    g = ((hash_value // 256) % 256)
    b = ((hash_value // 256 // 256) % 256)
    return (r, g, b)  # OpenCV uses BGR format

def get_project_2d(keypoint_data, cam_param, undistort=False, adj_cam=False,rotate=False,img_height=None):
    """
    Projects 3D keypoints onto a 2D image plane based on camera parameters.

    Args:
        keypoint_data (list): A list of 3D keypoints (N x 3 array) to be projected.
        cam_param (dict): Camera parameters, including intrinsic, extrinsic matrices, and distortion coefficients.
        undistort (bool): Flag indicating whether to undistort the points. Default is False.
        adj_cam (bool): Flag to adjust the camera position (scale it to meters). Default is False.
        rotate (bool): Flag to rotate the points. Default is False.
        img_height (int): The height of the image. Required if rotate is True.

    Returns:
        ndarray: The frame with the projected keypoints drawn on it.
    """
    fu = cam_param["affine_intrinsics_matrix"][0][0]
    fv = cam_param["affine_intrinsics_matrix"][1][1]
    cu = cam_param["affine_intrinsics_matrix"][0][2]
    cv = cam_param["affine_intrinsics_matrix"][1][2]
    affine_intrinsics_matrix = np.array(cam_param["affine_intrinsics_matrix"])

    rot_mat = np.array(cam_param["extrinsic_matrix"])
    rot_mat[1:, :] *= -1

    camera_position = np.array(cam_param["xyz"])
    #divided by 1000 to convert to meters
    camera_position = camera_position/1000 if adj_cam else camera_position

    distortion = np.array(cam_param["distortion"])
    
    #create an array to store the projected keypoints
    projected_keypoints = np.zeros((len(keypoint_data),2))
    for i in range(len(keypoint_data)):
        keypoints = keypoint_data[i]

        # World to cam
        translated = keypoints[0:3] - camera_position
        kpts_camera = (rot_mat @ translated.T).T

        # Cam to pixel
        Xc = kpts_camera[0]
        Yc = kpts_camera[1]
        Zc = kpts_camera[2]
        u = fu * (Xc / Zc) + cu
        v = fv * (Yc / Zc) + cv
        uv = np.stack([u, v], axis=-1)

        if undistort:
            # Undistort the points
            uv = cv2.undistortPoints(np.expand_dims(uv, axis=0), affine_intrinsics_matrix, distortion, None, affine_intrinsics_matrix)

            # Convert from normalized coordinates to pixel coordinates
            uv = uv.squeeze(axis=0)
            u, v = uv[0][0], uv[0][1]
        
        if rotate:
            if img_height is None:
                print("Error: Image height is required for rotating the points")
                return None
            
            #rotate the points
            u,v = v,u
            #flip the y
            v = img_height - v
        
        #store the projected keypoints
        projected_keypoints[i] = [u,v]
    return projected_keypoints


def load_pose_frames(npy_file, rig_path, rig_name, marker_labels):
    np_keypoint_data = np.load(npy_file)
    if np_keypoint_data.ndim == 4 and np_keypoint_data.shape[0] == 1:
        np_keypoint_data = np_keypoint_data[0]

    if np_keypoint_data.ndim != 3:
        raise ValueError(
            f"Unexpected keypoint shape: {np_keypoint_data.shape}, expected (num_frames, num_markers, 3)"
        )

    joint_names, marker_idxs = mp.load_rig_mapping(rig_path, rig_name, marker_labels)
    keypoint_data = np.array([
        mp.apply_rig_format(frame_kpts, joint_names, marker_idxs)
        for frame_kpts in np_keypoint_data
    ])
    return keypoint_data


def visualize_single_video(vid_path, npy_file, json_path, cam_param_path, rig_path, output_path,
                           rig_name="Human3.6M_RM", dataset="Human3.6M"):
    marker_labels = get_marker_labels(json_path)
    keypoint_data = load_pose_frames(npy_file, rig_path, rig_name, marker_labels)
    cam_param = get_cam_param(cam_param_path, json_path)

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {vid_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30 if fps <= 0 else fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", cv2.VideoWriter.fourcc)

    writer = cv2.VideoWriter(
        output_path,
        fourcc_fn(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create output video: {output_path}")

    frame_idx = 0
    while frame_idx < len(keypoint_data):
        ret, frame = cap.read()
        if not ret:
            break

        projected_keypoints = get_project_2d(keypoint_data[frame_idx], cam_param)
        frame_2d = vis.show2Dpose(projected_keypoints, frame, unique_color=False, dataset=dataset)
        writer.write(frame_2d)
        frame_idx += 1

    if frame_idx < len(keypoint_data):
        print(f"Warning: video ended early. rendered {frame_idx}/{len(keypoint_data)} frames for {os.path.basename(vid_path)}")

    extra_video_frames = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        extra_video_frames += 1

    if extra_video_frames > 0:
        print(f"Warning: pose data shorter than video by {extra_video_frames} frame(s) for {os.path.basename(vid_path)}")

    writer.release()
    cap.release()


def collect_video_files(input_dir, sort_by_name=True, limit=None):
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"}
    video_files = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and os.path.splitext(f)[1] in video_exts
    ]

    if sort_by_name:
        video_files.sort()

    if limit is not None and limit > 0:
        video_files = video_files[:limit]

    return video_files

if __name__ == "__main__":
    # -----------------------------
    # Direct config (edit here)
    # -----------------------------
    input_dir = "/home/data/ztw/AtheletePose3D/data/train_set/S3"
    cam_param_path = "/home/data/ztw/AtheletePose3D/cam_param.json"
    output_dir = "/home/ztw/HVCCS/res/Athelete3D120fpsoriginal"
    limit = 10              # e.g. 10, None means all videos
    sort_by_name = True       # True: sort by filename, False: filesystem order
    rig_name = "Human3.6M_RM"
    dataset = "Human3.6M"

    rig_path = os.path.join(os.path.dirname(__file__), "rig.json")

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not os.path.isfile(cam_param_path):
        raise FileNotFoundError(f"Camera parameter file not found: {cam_param_path}")
    if not os.path.isfile(rig_path):
        raise FileNotFoundError(f"Rig file not found: {rig_path}")

    os.makedirs(output_dir, exist_ok=True)
    video_files = collect_video_files(input_dir, sort_by_name=sort_by_name, limit=limit)

    print(f"Found {len(video_files)} video(s) to process in: {input_dir}")

    processed = 0
    skipped = 0
    failed = 0

    for idx, video_name in enumerate(video_files, start=1):
        base_name, _ = os.path.splitext(video_name)
        vid_path = os.path.join(input_dir, video_name)
        npy_file = os.path.join(input_dir, f"{base_name}.npy")
        json_path = os.path.join(input_dir, f"{base_name}.json")
        output_path = os.path.join(output_dir, f"{base_name}_vis.mp4")

        if not os.path.isfile(npy_file) or not os.path.isfile(json_path):
            print(f"[{idx}/{len(video_files)}] Skip {video_name}: missing matched npy/json")
            skipped += 1
            continue

        try:
            print(f"[{idx}/{len(video_files)}] Processing {video_name} ...")
            visualize_single_video(
                vid_path=vid_path,
                npy_file=npy_file,
                json_path=json_path,
                cam_param_path=cam_param_path,
                rig_path=rig_path,
                output_path=output_path,
                rig_name=rig_name,
                dataset=dataset,
            )
            print(f"Saved: {output_path}")
            processed += 1
        except Exception as e:
            print(f"[{idx}/{len(video_files)}] Failed {video_name}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Done. processed={processed}, skipped={skipped}, failed={failed}, total_selected={len(video_files)}")
