import cv2
import numpy as np
import zarr
from pathlib import Path
from tqdm import tqdm
import os

def crop_and_resize(frame, camera_idx, target_size=96):
    """裁剪图像中心的正方形区域并调整大小
    
    Args:
        frame: 原始帧 (720, 1280, 3)
        camera_idx: 相机索引 (0 或 1)
        target_size: 目标尺寸
    Returns:
        处理后的帧 (96, 96, 3)
    """
    h, w = frame.shape[:2]
    
    # 根据相机索引设置不同的起始x坐标
    crop_size = 672
    start_x = 150 if camera_idx == 0 else 400
    start_y = h//2 - crop_size//2

    # 颜色空间转换
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 裁剪
    cropped = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]
    
    # 缩放到目标尺寸
    resized = cv2.resize(cropped, (target_size, target_size), 
                        interpolation=cv2.INTER_AREA)
    
    return resized

def process_video(video_path, start_frame, num_frames, camera_idx, max_frame_diff=30):
    """处理单个视频文件
    
    Args:
        video_path: 视频文件路径
        start_frame: 在总帧序列中的起始位置
        num_frames: 需要处理的帧数
        camera_idx: 相机索引 (0 或 1)
        max_frame_diff: 允许的最大帧数差异
    Returns:
        处理后的帧序列 (num_frames, 96, 96, 3)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理帧时传入camera_idx
        processed_frame = crop_and_resize(frame, camera_idx)
        frames.append(processed_frame)
    
    cap.release()
    
    actual_frames = len(frames)
    frame_diff = num_frames - actual_frames
    
    if frame_diff > 0:
        if frame_diff <= max_frame_diff and actual_frames > 0:
            # 通过复制最后一帧来补全缺失的帧
            print(f"警告: {video_path} 缺少 {frame_diff} 帧，将通过复制最后一帧进行补全")
            last_frame = frames[-1]
            for _ in range(frame_diff):
                frames.append(last_frame)
        else:
            raise ValueError(f"视频帧数差异过大: {video_path}, 期望 {num_frames} 帧，实际获得 {actual_frames} 帧")
    
    return np.stack(frames, axis=0)

def main():
    # 配置路径
    base_dir = Path("./")  # 确保这是正确的基础目录
    zarr_path = base_dir / "replay_buffer.zarr"
    videos_dir = base_dir / "videos"
    
    # 打开zarr文件
    store = zarr.open(str(zarr_path), 'r+')
    episode_ends = store['meta/episode_ends'][:]
    
    # 计算每个episode的长度
    episode_lengths = np.diff(np.concatenate([[0], episode_ends]))
    total_frames = episode_ends[-1]
    
    # 创建或获取两个img数据集
    for camera_idx in range(2):
        dataset_name = f'data/img{camera_idx+1}'
        if dataset_name in store:
            print(f"警告：{dataset_name}已存在，将被覆盖")
            del store[dataset_name]
        
        # 为每个摄像头创建新的img数据集
        store.create_dataset(dataset_name, 
                           shape=(total_frames, 96, 96, 3),
                           chunks=(100, 96, 96, 3),
                           dtype=np.uint8)
    
    # 处理每个episode的视频
    for episode_idx in tqdm(range(len(episode_lengths)), desc="处理视频"):
        start_idx = 0 if episode_idx == 0 else episode_ends[episode_idx-1]
        end_idx = episode_ends[episode_idx]
        num_frames = end_idx - start_idx
        
        # 处理两个视频
        for camera_idx in range(2):
            video_path = videos_dir / str(episode_idx) / f"{camera_idx}.mp4"
            if not video_path.exists():
                raise FileNotFoundError(f"找不到视频文件: {video_path}")
            
            frames = process_video(video_path, start_idx, num_frames, camera_idx, max_frame_diff=30)
            
            # 将每个摄像头的帧存储到对应的数据集中
            dataset_name = f'data/img{camera_idx+1}'
            store[dataset_name][start_idx:end_idx] = frames
    
    print(f"处理完成！总共处理了 {total_frames} 帧")
    for camera_idx in range(2):
        dataset_name = f'data/img{camera_idx+1}'
        print(f"{dataset_name}数据集形状: {store[dataset_name].shape}")

if __name__ == "__main__":
    main()