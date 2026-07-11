"""
Preprocess action-recognition videos into frame folders.

Expected raw layout:
    raw_videos/
        ClassName/
            video1.avi
            ...

Output layout:
    frame_dir/
        ClassName/
            video1/
                frame_0000.jpg
                ...
"""

import argparse
import os
from pathlib import Path

from tqdm import tqdm

from utils import get_frames, store_frames


VIDEO_EXTENSIONS = {'.avi', '.mp4', '.mpeg', '.mpg', '.mov', '.mkv'}


def parse_args():
    parser = argparse.ArgumentParser(description='Extract uniformly sampled frames from action videos.')
    parser.add_argument('--input_dir', required=True, help='Directory containing class folders of raw videos.')
    parser.add_argument('--output_dir', required=True, help='Directory where extracted frame folders will be written.')
    parser.add_argument('--frames_per_video', type=int, default=16, help='Number of frames to sample per video.')
    parser.add_argument('--overwrite', action='store_true', help='Re-extract videos that already have frame files.')
    return parser.parse_args()


def iter_videos(input_dir):
    input_path = Path(input_dir)
    for class_dir in sorted(input_path.iterdir()):
        if not class_dir.is_dir():
            continue
        for video_path in sorted(class_dir.iterdir()):
            if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
                yield class_dir.name, video_path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    videos = list(iter_videos(args.input_dir))
    if not videos:
        raise ValueError('No video files found under {}'.format(args.input_dir))

    skipped = 0
    failed = 0
    for class_name, video_path in tqdm(videos, desc='Extracting frames'):
        output_path = Path(args.output_dir) / class_name / video_path.stem
        if output_path.exists() and any(output_path.glob('*.jpg')) and not args.overwrite:
            skipped += 1
            continue

        frames, _ = get_frames(str(video_path), args.frames_per_video)
        if not frames:
            failed += 1
            continue
        store_frames(frames, str(output_path))

    print('Processed {} videos, skipped {}, failed {}.'.format(len(videos) - skipped - failed, skipped, failed))


if __name__ == '__main__':
    main()
