from __future__ import annotations

import csv
import cv2
import yaml
import os
import numpy as np
from pathlib import Path
from urllib.parse import urljoin
from typing import Final, Any
from collections.abc import Iterable
from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile
from path_constants import Retention, BENCHMARK_RETENTION, VSLAMLAB_VIDEOS


class VIDEOS_dataset(DatasetVSLAMLab):
    """VIDEOS dataset helper for VSLAM-LAB benchmark."""
    
    def __init__(self, benchmark_path: str | Path, dataset_name: str = "videos") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get videso path
        self.videos_path = VSLAMLAB_VIDEOS

        # Create sequence_nicknames
        self.sequence_nicknames = self.sequence_names

        # Get resolution size
        self.resolution_size = cfg['resolution_size']

    def download_sequence_data(self, sequence_name: str) -> None:
        pass

    def create_rgb_folder(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / 'rgb_0'
        for p in self.videos_path.iterdir():
            if p.is_file() and sequence_name in p.name:
                video_path = p
                break

        if not rgb_path.exists():
            self.extract_png_frames(video_path=video_path, output_dir=rgb_path)  # extract at 30Hz

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_path = sequence_path / 'rgb_0'
        rgb_csv = sequence_path / 'rgb.csv'

        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        rgb_files.sort()

        tmp = rgb_csv.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts_rgb_0 (ns)", "path_rgb_0"])
            idx = 0
            for filename in rgb_files:
                path_r0 = "rgb_0/" + filename
                ts_r0_ns = int(float(idx) / 30 * 1e9)                
                w.writerow([ts_r0_ns, path_r0])
                idx += 1
        tmp.replace(rgb_csv)

    def create_calibration_yaml(self, sequence_name: str) -> None:
       
        rgb: dict[str, Any] = {"cam_name": "rgb_0", "cam_type": "rgb", 
                    "cam_model": "unknown", "focal_length": [0, 0], "principal_point": [0, 0],
                    "fps": float(self.rgb_hz),
                    "T_BS": np.eye(4)}
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        groundtruth_csv = sequence_path / "groundtruth.csv"
        tmp = groundtruth_csv.with_suffix(".csv.tmp")

        with open(tmp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts (ns)","tx (m)","ty (m)","tz (m)","qx","qy","qz","qw"])
        tmp.replace(groundtruth_csv)
             
    def extract_png_frames(self, video_path: Path, output_dir: Path):
        """
        Extract frames from a video based on a frequency in Hertz (frames per second) and save as PNG images.
        Also creates an rgb.txt file with timestamps and image paths.

        Args:
            video_path (str): Path to the input video file.
            output_dir (str): Directory to save the PNG files.
            frequency_hz (int or float): How many frames to save per second of video.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError("Failed to get FPS from video.")

        frame_interval = int(round(fps / self.rgb_hz))

        print(f"Video opened: {video_path}")
        print(f"Video FPS: {fps:.2f}")
        print(f"Extracting {self.rgb_hz} frames per second (every {frame_interval} frames).")

        frame_idx = 0
        saved_idx = 0
        timestamp_list = []

        estimate_new_resolution = True
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # Compute timestamp
                timestamp_nsec = int(1e9 * frame_idx / fps)

                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if estimate_new_resolution:
                    rgb_frame_height, rgb_frame_width = rgb_frame.shape[:2]
                    scaled_height = np.sqrt(self.resolution_size[0] * self.resolution_size[1] * rgb_frame_height / rgb_frame_width)
                    scaled_width = self.resolution_size[0] * self.resolution_size[1] / scaled_height
                    scaled_height = int(scaled_height)
                    scaled_width = int(scaled_width)
                    estimate_new_resolution = False
                
                resized_img = cv2.resize(rgb_frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LANCZOS4)

                # Save as PNG with 5-digit padded integer filename
                filename = output_dir / f"{saved_idx:05d}.png"
                cv2.imwrite(str(filename), cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))

                # Save timestamp and image path
                image_relative_path = output_dir / f"{saved_idx:05d}.png"
                timestamp_list.append((timestamp_nsec, str(image_relative_path)))

                saved_idx += 1

            frame_idx += 1

        cap.release()