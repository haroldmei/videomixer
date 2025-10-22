#!/usr/bin/env python3
"""
Optimized Video GPS Overlay Tool
Overlays GPS track data from GPX file onto video with time alignment
Features: Progress display, Google Maps integration, RAM processing, multiprocessing
"""

import cv2
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import argparse
import os
from typing import List, Tuple, Optional
import math
import multiprocessing as mp
from multiprocessing import Pool, Manager
import folium
from PIL import Image
import io
import base64
from tqdm import tqdm
import tempfile


class GPSPoint:
    """Represents a single GPS point with timestamp"""
    def __init__(self, lat: float, lon: float, ele: float, timestamp: datetime):
        self.lat = lat
        self.lon = lon
        self.ele = ele
        self.timestamp = timestamp


class GPXParser:
    """Parses GPX files and extracts GPS track points"""
    
    @staticmethod
    def parse_gpx(gpx_file: str) -> List[GPSPoint]:
        """Parse GPX file and return list of GPS points"""
        tree = ET.parse(gpx_file)
        root = tree.getroot()
        
        points = []
        
        namespace = root.tag.split('}')[0] + '}' if '}' in root.tag else ''
        track_points = root.findall(f'.//{namespace}trkpt')
        
        if not track_points:
            track_points = root.findall('.//trkpt')
        
        print(f"Found {len(track_points)} track points")
        
        for trkpt in track_points:
            lat = float(trkpt.get('lat'))
            lon = float(trkpt.get('lon'))
            
            ele_elem = trkpt.find(f'{namespace}ele') if namespace else trkpt.find('ele')
            ele = float(ele_elem.text) if ele_elem is not None else 0.0
            
            time_elem = trkpt.find(f'{namespace}time') if namespace else trkpt.find('time')
            if time_elem is not None:
                timestamp = datetime.fromisoformat(time_elem.text.replace('Z', '+00:00'))
                points.append(GPSPoint(lat, lon, ele, timestamp))
        
        return points


class MapGenerator:
    """Generates Google Maps visualization for GPS track"""
    
    @staticmethod
    def create_map_html(gps_points: List[GPSPoint], output_file: str = "gps_map.html"):
        """Create interactive map HTML file"""
        if not gps_points:
            return None
        
        lats = [p.lat for p in gps_points]
        lons = [p.lon for p in gps_points]
        
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
        
        folium.PolyLine(
            locations=[[p.lat, p.lon] for p in gps_points],
            color='red',
            weight=3,
            opacity=0.8
        ).add_to(m)
        
        folium.Marker(
            [gps_points[0].lat, gps_points[0].lon],
            popup='Start',
            icon=folium.Icon(color='green')
        ).add_to(m)
        
        folium.Marker(
            [gps_points[-1].lat, gps_points[-1].lon],
            popup='End',
            icon=folium.Icon(color='red')
        ).add_to(m)
        
        m.save(output_file)
        print(f"Map saved to: {output_file}")
        return output_file


class VideoProcessor:
    """Handles video processing and GPS overlay with optimizations"""
    
    def __init__(self, video_path: str, gps_points: List[GPSPoint]):
        self.video_path = video_path
        self.gps_points = gps_points
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.video_duration = 0
        self.processed_frames = []
        
    def get_video_duration(self) -> float:
        """Get video duration in seconds"""
        if not self.cap:
            self.cap = cv2.VideoCapture(self.video_path)
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        self.fps = fps
        self.frame_count = frame_count
        self.video_duration = duration
        
        return duration
    
    def get_gps_duration(self) -> float:
        """Get GPS track duration in seconds"""
        if not self.gps_points:
            return 0
        
        start_time = self.gps_points[0].timestamp
        end_time = self.gps_points[-1].timestamp
        
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=datetime.timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=datetime.timezone.utc)
        
        duration = (end_time - start_time).total_seconds()
        return duration
    
    def align_times(self) -> Tuple[float, float]:
        """Align video and GPS durations, return (video_duration, gps_duration)"""
        video_duration = self.get_video_duration()
        gps_duration = self.get_gps_duration()
        
        print(f"Video duration: {video_duration:.2f} seconds")
        print(f"GPS duration: {gps_duration:.2f} seconds")
        
        target_duration = max(video_duration, gps_duration)
        
        if abs(video_duration - gps_duration) > 1.0:
            print(f"Aligning to target duration: {target_duration:.2f} seconds")
        
        return video_duration, gps_duration
    
    def get_gps_position_at_time(self, time_seconds: float, gps_duration: float, video_duration: float) -> Optional[GPSPoint]:
        """Get GPS position at specific time, with time alignment"""
        if not self.gps_points:
            return None
        
        target_duration = max(video_duration, gps_duration)
        
        if video_duration != target_duration:
            time_ratio = time_seconds / video_duration
            gps_time = time_ratio * target_duration
        else:
            gps_time = time_seconds
        
        if gps_duration != target_duration:
            gps_time = (gps_time / target_duration) * gps_duration
        
        start_time = self.gps_points[0].timestamp
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=datetime.timezone.utc)
        
        target_time = start_time + timedelta(seconds=gps_time)
        
        closest_point = None
        min_diff = float('inf')
        
        for point in self.gps_points:
            point_time = point.timestamp
            if point_time.tzinfo is None:
                point_time = point_time.replace(tzinfo=datetime.timezone.utc)
            
            time_diff = abs((point_time - target_time).total_seconds())
            if time_diff < min_diff:
                min_diff = time_diff
                closest_point = point
        
        return closest_point
    
    def project_gps_to_screen(self, gps_point: GPSPoint, frame_width: int, frame_height: int, 
                            track_bounds: Optional[Tuple[float, float, float, float]] = None) -> Tuple[int, int]:
        """Project GPS coordinates to screen coordinates"""
        if not gps_point:
            return frame_width // 2, frame_height // 2
        
        if track_bounds is None:
            lats = [p.lat for p in self.gps_points]
            lons = [p.lon for p in self.gps_points]
            track_bounds = (min(lats), max(lats), min(lons), max(lons))
        
        min_lat, max_lat, min_lon, max_lon = track_bounds
        
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        padding = 0.1
        
        if lat_range == 0:
            lat_range = 0.001
        if lon_range == 0:
            lon_range = 0.001
        
        x = int((gps_point.lon - (min_lon - lon_range * padding)) / 
                ((max_lon + lon_range * padding) - (min_lon - lon_range * padding)) * frame_width)
        y = int((gps_point.lat - (min_lat - lat_range * padding)) / 
                ((max_lat + lat_range * padding) - (min_lat - lat_range * padding)) * frame_height)
        
        y = frame_height - y
        
        x = max(0, min(frame_width - 1, x))
        y = max(0, min(frame_height - 1, y))
        
        return x, y
    
    def draw_gps_overlay(self, frame: np.ndarray, current_time: float, 
                        video_duration: float, gps_duration: float) -> np.ndarray:
        """Draw GPS overlay on video frame"""
        current_gps = self.get_gps_position_at_time(current_time, gps_duration, video_duration)
        
        if not current_gps:
            return frame
        
        lats = [p.lat for p in self.gps_points]
        lons = [p.lon for p in self.gps_points]
        track_bounds = (min(lats), max(lats), min(lons), max(lons))
        
        frame_height, frame_width = frame.shape[:2]
        
        for i, gps_point in enumerate(self.gps_points):
            x, y = self.project_gps_to_screen(gps_point, frame_width, frame_height, track_bounds)
            
            point_time = (gps_point.timestamp - self.gps_points[0].timestamp).total_seconds()
            if gps_duration != video_duration:
                point_time = (point_time / gps_duration) * max(gps_duration, video_duration)
            
            if point_time <= current_time:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            else:
                cv2.circle(frame, (x, y), 1, (100, 100, 100), -1)
        
        current_x, current_y = self.project_gps_to_screen(current_gps, frame_width, frame_height, track_bounds)
        cv2.circle(frame, (current_x, current_y), 8, (0, 0, 255), -1)
        cv2.circle(frame, (current_x, current_y), 12, (255, 255, 255), 2)
        
        info_text = f"GPS: {current_gps.lat:.6f}, {current_gps.lon:.6f}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        time_text = f"Time: {current_time:.1f}s / {max(video_duration, gps_duration):.1f}s"
        cv2.putText(frame, time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    @staticmethod
    def process_frame_batch(args: Tuple[int, np.ndarray, float, float, float, List[GPSPoint]]) -> Tuple[int, np.ndarray]:
        """Process a single frame - used for multiprocessing"""
        frame_number, frame, current_time, video_duration, gps_duration, gps_points = args
        
        # Create a temporary processor instance for this frame
        temp_processor = VideoProcessor("", gps_points)
        processed_frame = temp_processor.draw_gps_overlay(frame, current_time, video_duration, gps_duration)
        return frame_number, processed_frame
    
    def process_video_optimized(self, output_path: str, num_processes: int = None):
        """Process video with optimizations: progress bar, RAM processing, multiprocessing"""
        if not self.cap:
            self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")
        
        video_duration, gps_duration = self.align_times()
        
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing {total_frames} frames with {num_processes or mp.cpu_count()} processes...")
        
        # Read all frames into RAM
        print("Loading frames into RAM...")
        frames_data = []
        with tqdm(total=total_frames, desc="Loading frames") as pbar:
            frame_number = 0
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                current_time = frame_number / fps
                frames_data.append((frame_number, frame.copy(), current_time, video_duration, gps_duration, self.gps_points))
                frame_number += 1
                pbar.update(1)
        
        self.cap.release()
        
        # Process frames in parallel
        print("Processing frames...")
        if num_processes is None:
            num_processes = min(mp.cpu_count(), len(frames_data))
        
        processed_frames = {}
        
        try:
            with Pool(processes=num_processes) as pool:
                with tqdm(total=len(frames_data), desc="Processing frames") as pbar:
                    for frame_number, processed_frame in pool.imap(VideoProcessor.process_frame_batch, frames_data):
                        processed_frames[frame_number] = processed_frame
                        pbar.update(1)
        except Exception as e:
            print(f"Multiprocessing failed: {e}")
            print("Falling back to single-threaded processing...")
            with tqdm(total=len(frames_data), desc="Processing frames") as pbar:
                for frame_data in frames_data:
                    frame_number, processed_frame = VideoProcessor.process_frame_batch(frame_data)
                    processed_frames[frame_number] = processed_frame
                    pbar.update(1)
        
        # Write all processed frames to video file
        print("Writing video file...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        with tqdm(total=total_frames, desc="Writing video") as pbar:
            for i in range(total_frames):
                if i in processed_frames:
                    out.write(processed_frames[i])
                pbar.update(1)
        
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Video processing complete! Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Overlay GPS track on video (Optimized)')
    parser.add_argument('video', help='Input video file path')
    parser.add_argument('gpx', help='Input GPX file path')
    parser.add_argument('-o', '--output', help='Output video file path', 
                       default='output_with_gps_overlay.mp4')
    parser.add_argument('-m', '--map', help='Generate map HTML file', 
                       default='gps_map.html')
    parser.add_argument('-p', '--processes', type=int, help='Number of processes for multiprocessing')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    if not os.path.exists(args.gpx):
        print(f"Error: GPX file not found: {args.gpx}")
        return
    
    try:
        print("Parsing GPX file...")
        gps_points = GPXParser.parse_gpx(args.gpx)
        print(f"Found {len(gps_points)} GPS points")
        
        if not gps_points:
            print("Error: No GPS points found in GPX file")
            return
        
        print("Generating GPS map...")
        MapGenerator.create_map_html(gps_points, args.map)
        
        print("Processing video...")
        processor = VideoProcessor(args.video, gps_points)
        processor.process_video_optimized(args.output, args.processes)
        
    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.freeze_support()
    main()