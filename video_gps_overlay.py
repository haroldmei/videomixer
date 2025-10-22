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
import requests
import json
from urllib.parse import urlencode


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
    
    @staticmethod
    def create_static_map_image(gps_points: List[GPSPoint], width: int = 300, height: int = 200) -> np.ndarray:
        """Create a static map image for video overlay"""
        if not gps_points:
            return None
        
        lats = [p.lat for p in gps_points]
        lons = [p.lon for p in gps_points]
        
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        
        # Create map with specific dimensions
        m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=13,
            width=width,
            height=height,
            tiles='OpenStreetMap'
        )
        
        # Add track line
        folium.PolyLine(
            locations=[[p.lat, p.lon] for p in gps_points],
            color='red',
            weight=2,
            opacity=0.8
        ).add_to(m)
        
        # Add start marker
        folium.CircleMarker(
            [gps_points[0].lat, gps_points[0].lon],
            radius=3,
            color='green',
            fill=True,
            fillColor='green'
        ).add_to(m)
        
        # Add end marker
        folium.CircleMarker(
            [gps_points[-1].lat, gps_points[-1].lon],
            radius=3,
            color='red',
            fill=True,
            fillColor='red'
        ).add_to(m)
        
        # Convert map to image
        map_data = m._repr_html_()
        
        # Create a temporary HTML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(map_data)
            temp_html = f.name
        
        try:
            # Use selenium or similar to render HTML to image
            # For now, create a simple colored map representation
            map_image = np.zeros((height, width, 3), dtype=np.uint8)
            map_image[:] = [50, 50, 50]  # Dark gray background
            
            # Draw track bounds
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)
            
            # Draw track points
            for i, point in enumerate(gps_points):
                x = int((point.lon - min_lon) / (max_lon - min_lon) * width)
                y = int((point.lat - min_lat) / (max_lat - min_lat) * height)
                y = height - y  # Flip Y coordinate
                
                if 0 <= x < width and 0 <= y < height:
                    if i == 0:
                        cv2.circle(map_image, (x, y), 3, (0, 255, 0), -1)  # Green start
                    elif i == len(gps_points) - 1:
                        cv2.circle(map_image, (x, y), 3, (0, 0, 255), -1)  # Red end
                    else:
                        cv2.circle(map_image, (x, y), 1, (255, 255, 255), -1)  # White track points
            
            # Draw track line
            for i in range(1, len(gps_points)):
                x1 = int((gps_points[i-1].lon - min_lon) / (max_lon - min_lon) * width)
                y1 = int((gps_points[i-1].lat - min_lat) / (max_lat - min_lat) * height)
                y1 = height - y1
                
                x2 = int((gps_points[i].lon - min_lon) / (max_lon - min_lon) * width)
                y2 = int((gps_points[i].lat - min_lat) / (max_lat - min_lat) * height)
                y2 = height - y2
                
                if (0 <= x1 < width and 0 <= y1 < height and 
                    0 <= x2 < width and 0 <= y2 < height):
                    cv2.line(map_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
            
            return map_image
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_html)
            except:
                pass


class GoogleMapsAPI:
    """High-definition map generation using Google Maps API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/staticmap"
    
    def create_high_res_map(self, gps_points: List[GPSPoint], current_gps: GPSPoint, 
                           width: int = 400, height: int = 300, zoom: int = 15) -> np.ndarray:
        """Create high-resolution map using Google Maps API"""
        if not gps_points or not current_gps:
            return self._create_fallback_map(gps_points, current_gps, width, height)
        
        # Calculate map bounds from GPS data
        lats = [p.lat for p in gps_points]
        lons = [p.lon for p in gps_points]
        
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        
        # Calculate center point
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        
        # Calculate optimal zoom level based on GPS bounds
        optimal_zoom = self._calculate_optimal_zoom(min_lat, max_lat, min_lon, max_lon, width, height)
        
        # Reduce GPS points if URL would be too long
        reduced_points = self._reduce_gps_points_for_url(gps_points)
        
        # Create path string for the track
        path_points = []
        for point in reduced_points:
            path_points.append(f"{point.lat},{point.lon}")
        path_string = "|".join(path_points)
        
        # Create markers
        markers = []
        
        # Start marker
        markers.append(f"color:green|label:S|{gps_points[0].lat},{gps_points[0].lon}")
        
        # End marker
        markers.append(f"color:red|label:E|{gps_points[-1].lat},{gps_points[-1].lon}")
        
        # Current position marker
        markers.append(f"color:blue|label:C|{current_gps.lat},{current_gps.lon}")
        
        # Build URL parameters
        params = {
            'center': f"{center_lat},{center_lon}",
            'zoom': str(optimal_zoom),
            'size': f"{width}x{height}",
            'maptype': 'roadmap',
            'path': f"color:0xff0000ff|weight:3|{path_string}",
            'markers': markers,
            'format': 'png'
        }
        
        if self.api_key:
            params['key'] = self.api_key
        
        # Create URL
        url = f"{self.base_url}?{urlencode(params)}"
        
        # Check URL length (Google Maps API has ~8192 character limit)
        if len(url) > 8000:
            print(f"URL too long ({len(url)} chars), reducing points further...")
            # Further reduce points
            reduced_points = self._reduce_gps_points_for_url(gps_points, max_points=50)
            path_points = [f"{point.lat},{point.lon}" for point in reduced_points]
            path_string = "|".join(path_points)
            params['path'] = f"color:0xff0000ff|weight:3|{path_string}"
            url = f"{self.base_url}?{urlencode(params)}"
            print(f"Reduced URL length to {len(url)} chars")
        
        try:
            # Download map image
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                # Convert to OpenCV format
                image = Image.open(io.BytesIO(response.content))
                image = image.convert('RGB')
                map_array = np.array(image)
                map_array = cv2.cvtColor(map_array, cv2.COLOR_RGB2BGR)
                return map_array
            else:
                print(f"Google Maps API error: {response.status_code}")
                if response.status_code == 400:
                    print("API request error - check your API key and parameters")
                    print(f"Request URL: {url}")
                    print("Response:", response.text[:200])
                return self._create_fallback_map(gps_points, current_gps, width, height)
                
        except Exception as e:
            print(f"Error fetching Google Maps: {e}")
            return self._create_fallback_map(gps_points, current_gps, width, height)
    
    def _reduce_gps_points_for_url(self, gps_points: List[GPSPoint], max_points: int = 200) -> List[GPSPoint]:
        """Reduce GPS points to fit within URL length limits while preserving track shape"""
        if len(gps_points) <= max_points:
            return gps_points
        
        # Always include first and last points
        reduced = [gps_points[0]]
        
        # Calculate step size for uniform sampling
        step = len(gps_points) / (max_points - 2)
        
        for i in range(1, max_points - 1):
            idx = int(i * step)
            if idx < len(gps_points) - 1:
                reduced.append(gps_points[idx])
        
        # Always include the last point
        if len(gps_points) > 1:
            reduced.append(gps_points[-1])
        
        return reduced
    
    def _calculate_optimal_zoom(self, min_lat: float, max_lat: float, min_lon: float, max_lon: float, 
                              width: int, height: int) -> int:
        """Calculate optimal zoom level to fit GPS bounds within map dimensions"""
        # Calculate the span of the GPS track
        lat_span = max_lat - min_lat
        lon_span = max_lon - min_lon
        
        # Add padding (10% on each side)
        lat_span *= 1.2
        lon_span *= 1.2
        
        # Calculate zoom level based on the larger span
        # Google Maps zoom levels: each level doubles the resolution
        # Level 0 shows the whole world, level 20 shows individual buildings
        
        # Calculate zoom based on latitude span
        lat_zoom = int(np.log2(360 / lat_span))
        
        # Calculate zoom based on longitude span (adjusted for latitude)
        # Longitude degrees get smaller as you move away from equator
        avg_lat = (min_lat + max_lat) / 2
        lon_degrees_per_pixel = lon_span / (width / 256)  # 256 is base tile size
        lon_zoom = int(np.log2(360 / lon_degrees_per_pixel))
        
        # Use the smaller zoom level to ensure everything fits
        optimal_zoom = min(lat_zoom, lon_zoom)
        
        # Clamp zoom level to valid range (0-20)
        optimal_zoom = max(0, min(20, optimal_zoom))
        
        # For very small tracks, ensure minimum zoom level
        if optimal_zoom > 18:
            optimal_zoom = 18
        
        print(f"GPS bounds: lat={lat_span:.6f}, lon={lon_span:.6f}, optimal zoom: {optimal_zoom}")
        
        return optimal_zoom
    
    def _create_fallback_map(self, gps_points: List[GPSPoint], current_gps: GPSPoint, 
                            width: int, height: int) -> np.ndarray:
        """Create fallback map when API is not available"""
        if not gps_points:
            return np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create map image
        map_image = np.zeros((height, width, 3), dtype=np.uint8)
        map_image[:] = [40, 40, 40]  # Dark gray background
        
        # Calculate bounds with same logic as Google Maps
        lats = [p.lat for p in gps_points]
        lons = [p.lon for p in gps_points]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        
        # Add padding (20% total, same as Google Maps)
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        padding = 0.1
        
        if lat_range == 0:
            lat_range = 0.001
        if lon_range == 0:
            lon_range = 0.001
        
        min_lat -= lat_range * padding
        max_lat += lat_range * padding
        min_lon -= lon_range * padding
        max_lon += lon_range * padding
        
        # Draw complete track
        for i in range(1, len(gps_points)):
            x1 = int((gps_points[i-1].lon - min_lon) / (max_lon - min_lon) * width)
            y1 = int((gps_points[i-1].lat - min_lat) / (max_lat - min_lat) * height)
            y1 = height - y1
            
            x2 = int((gps_points[i].lon - min_lon) / (max_lon - min_lon) * width)
            y2 = int((gps_points[i].lat - min_lat) / (max_lat - min_lat) * height)
            y2 = height - y2
            
            if (0 <= x1 < width and 0 <= y1 < height and 
                0 <= x2 < width and 0 <= y2 < height):
                cv2.line(map_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw current position
        if current_gps:
            current_x = int((current_gps.lon - min_lon) / (max_lon - min_lon) * width)
            current_y = int((current_gps.lat - min_lat) / (max_lat - min_lat) * height)
            current_y = height - current_y
            
            if 0 <= current_x < width and 0 <= current_y < height:
                cv2.circle(map_image, (current_x, current_y), 5, (0, 0, 255), -1)
                cv2.circle(map_image, (current_x, current_y), 7, (255, 255, 255), 2)
        
        # Draw start and end points
        start_x = int((gps_points[0].lon - min_lon) / (max_lon - min_lon) * width)
        start_y = int((gps_points[0].lat - min_lat) / (max_lat - min_lat) * height)
        start_y = height - start_y
        
        end_x = int((gps_points[-1].lon - min_lon) / (max_lon - min_lon) * width)
        end_y = int((gps_points[-1].lat - min_lat) / (max_lat - min_lat) * height)
        end_y = height - end_y
        
        if 0 <= start_x < width and 0 <= start_y < height:
            cv2.circle(map_image, (start_x, start_y), 4, (0, 255, 0), -1)
            cv2.putText(map_image, "S", (start_x-5, start_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if 0 <= end_x < width and 0 <= end_y < height:
            cv2.circle(map_image, (end_x, end_y), 4, (255, 0, 0), -1)
            cv2.putText(map_image, "E", (end_x-5, end_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return map_image


class VideoProcessor:
    """Handles video processing and GPS overlay with optimizations"""
    
    def __init__(self, video_path: str, gps_points: List[GPSPoint], google_api_key: str = None):
        self.video_path = video_path
        self.gps_points = gps_points
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.video_duration = 0
        self.processed_frames = []
        self.map_image = None
        self.google_maps = GoogleMapsAPI(google_api_key)
        
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
                        video_duration: float, gps_duration: float, 
                        track_thickness: int = 3, contrast_factor: float = 1.5,
                        show_map: bool = True, map_only: bool = False) -> np.ndarray:
        """Draw GPS overlay on video frame"""
        current_gps = self.get_gps_position_at_time(current_time, gps_duration, video_duration)
        
        if not current_gps:
            return frame
        
        frame_height, frame_width = frame.shape[:2]
        
        # If map-only mode, create a black background
        if map_only:
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            frame[:] = [20, 20, 20]  # Dark background
        
        lats = [p.lat for p in self.gps_points]
        lons = [p.lon for p in self.gps_points]
        track_bounds = (min(lats), max(lats), min(lons), max(lons))
        
        # Create a semi-transparent overlay for better visibility
        overlay = frame.copy()
        
        # Draw track path with lines for better visibility
        past_points = []
        future_points = []
        
        for i, gps_point in enumerate(self.gps_points):
            x, y = self.project_gps_to_screen(gps_point, frame_width, frame_height, track_bounds)
            
            point_time = (gps_point.timestamp - self.gps_points[0].timestamp).total_seconds()
            if gps_duration != video_duration:
                point_time = (point_time / gps_duration) * max(gps_duration, video_duration)
            
            if point_time <= current_time:
                past_points.append((x, y))
            else:
                future_points.append((x, y))
        
        # Draw past track as connected line
        if len(past_points) > 1:
            for i in range(1, len(past_points)):
                cv2.line(overlay, past_points[i-1], past_points[i], (0, 255, 0), track_thickness)
        
        # Draw future track as connected line
        if len(future_points) > 1:
            for i in range(1, len(future_points)):
                cv2.line(overlay, future_points[i-1], future_points[i], (100, 100, 100), track_thickness-1)
        
        # Draw individual points for better visibility
        for point in past_points:
            cv2.circle(overlay, point, track_thickness+1, (0, 255, 0), -1)
            cv2.circle(overlay, point, track_thickness+3, (255, 255, 255), 1)
        
        for point in future_points:
            cv2.circle(overlay, point, track_thickness, (100, 100, 100), -1)
        
        # Draw current position with enhanced visibility
        current_x, current_y = self.project_gps_to_screen(current_gps, frame_width, frame_height, track_bounds)
        cv2.circle(overlay, (current_x, current_y), 12, (0, 0, 255), -1)  # Red center
        cv2.circle(overlay, (current_x, current_y), 16, (255, 255, 255), 3)  # White border
        cv2.circle(overlay, (current_x, current_y), 20, (0, 0, 0), 2)  # Black outer border
        
        # Apply contrast enhancement
        if contrast_factor != 1.0:
            overlay = cv2.convertScaleAbs(overlay, alpha=contrast_factor, beta=0)
        
        # Blend overlay with original frame
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        
        # Enhanced text visibility with background
        info_text = f"GPS: {current_gps.lat:.6f}, {current_gps.lon:.6f}"
        time_text = f"Time: {current_time:.1f}s / {max(video_duration, gps_duration):.1f}s"
        
        # Draw background rectangles for text
        cv2.rectangle(frame, (5, 5), (500, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (500, 80), (255, 255, 255), 2)
        
        # Draw text with better visibility
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add map overlay if enabled
        if show_map:
            self._add_map_overlay(frame, current_gps, current_time, gps_duration, video_duration, map_only)
        
        return frame
    
    def _add_map_overlay(self, frame: np.ndarray, current_gps: GPSPoint, current_time: float, 
                        gps_duration: float, video_duration: float, map_only: bool = False):
        """Add high-definition Google Maps overlay to the frame"""
        frame_height, frame_width = frame.shape[:2]
        
        # Map dimensions - larger for map-only mode
        if map_only:
            map_width = min(frame_width - 40, 800)
            map_height = min(frame_height - 100, 600)
        else:
            map_width = 400
            map_height = 300
        
        # Position map with bounds checking
        if map_only:
            # Center the map in map-only mode
            map_x = max(0, (frame_width - map_width) // 2)
            map_y = max(0, (frame_height - map_height) // 2)
        else:
            # Top-right corner for normal mode
            map_x = max(0, frame_width - map_width - 20)
            map_y = 20
        
        # Ensure map fits within frame bounds
        map_total_width = map_width + 30
        map_total_height = map_height + 50
        
        if map_x + map_total_width > frame_width:
            map_x = max(0, frame_width - map_total_width)
        if map_y + map_total_height > frame_height:
            map_y = max(0, frame_height - map_total_height)
        
        # Create map background with border
        map_bg = np.zeros((map_height + 50, map_width + 30, 3), dtype=np.uint8)
        map_bg[:] = [20, 20, 20]  # Dark background
        
        # Add border
        cv2.rectangle(map_bg, (0, 0), (map_width + 29, map_height + 49), (255, 255, 255), 3)
        
        # Add map title
        title_text = "GPS Track Map" if not map_only else "GPS Track Map (Map Only Mode)"
        cv2.putText(map_bg, title_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Create high-definition map using Google Maps API
        map_image = self.google_maps.create_high_res_map(
            self.gps_points, current_gps, map_width, map_height, zoom=15
        )
        
        if map_image is not None:
            # Ensure map_image has correct dimensions
            if map_image.shape[:2] != (map_height, map_width):
                map_image = cv2.resize(map_image, (map_width, map_height))
            
            # Place map on background
            map_bg[40:40+map_height, 15:15+map_width] = map_image
            
            # Overlay on frame with bounds checking
            roi_height = map_height + 50
            roi_width = map_width + 30
            
            # Ensure ROI is within frame bounds
            if (map_y + roi_height <= frame_height and 
                map_x + roi_width <= frame_width and
                map_y >= 0 and map_x >= 0):
                
                roi = frame[map_y:map_y+roi_height, map_x:map_x+roi_width]
                
                if roi.shape == map_bg.shape:
                    if map_only:
                        # In map-only mode, make the map more prominent
                        cv2.addWeighted(roi, 0.1, map_bg, 0.9, 0, roi)
                    else:
                        # Normal blending for video overlay mode
                        cv2.addWeighted(roi, 0.2, map_bg, 0.8, 0, roi)
                else:
                    # Direct copy if sizes don't match
                    frame[map_y:map_y+roi_height, map_x:map_x+roi_width] = map_bg
    
    def _create_dynamic_map(self, current_gps: GPSPoint, current_time: float, 
                           gps_duration: float, video_duration: float, 
                           width: int, height: int) -> np.ndarray:
        """Create a dynamic map showing current position"""
        if not self.gps_points:
            return None
        
        # Create map image
        map_image = np.zeros((height, width, 3), dtype=np.uint8)
        map_image[:] = [30, 30, 30]  # Dark gray background
        
        # Calculate bounds
        lats = [p.lat for p in self.gps_points]
        lons = [p.lon for p in self.gps_points]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        
        # Add padding
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        padding = 0.1
        
        if lat_range == 0:
            lat_range = 0.001
        if lon_range == 0:
            lon_range = 0.001
        
        min_lat -= lat_range * padding
        max_lat += lat_range * padding
        min_lon -= lon_range * padding
        max_lon += lon_range * padding
        
        # Draw complete track
        for i in range(1, len(self.gps_points)):
            x1 = int((self.gps_points[i-1].lon - min_lon) / (max_lon - min_lon) * width)
            y1 = int((self.gps_points[i-1].lat - min_lat) / (max_lat - min_lat) * height)
            y1 = height - y1
            
            x2 = int((self.gps_points[i].lon - min_lon) / (max_lon - min_lon) * width)
            y2 = int((self.gps_points[i].lat - min_lat) / (max_lat - min_lat) * height)
            y2 = height - y2
            
            if (0 <= x1 < width and 0 <= y1 < height and 
                0 <= x2 < width and 0 <= y2 < height):
                cv2.line(map_image, (x1, y1), (x2, y2), (100, 100, 100), 1)
        
        # Draw current position
        current_x = int((current_gps.lon - min_lon) / (max_lon - min_lon) * width)
        current_y = int((current_gps.lat - min_lat) / (max_lat - min_lat) * height)
        current_y = height - current_y
        
        if 0 <= current_x < width and 0 <= current_y < height:
            cv2.circle(map_image, (current_x, current_y), 4, (0, 0, 255), -1)  # Red current position
            cv2.circle(map_image, (current_x, current_y), 6, (255, 255, 255), 1)  # White border
        
        # Draw start and end points
        start_x = int((self.gps_points[0].lon - min_lon) / (max_lon - min_lon) * width)
        start_y = int((self.gps_points[0].lat - min_lat) / (max_lat - min_lat) * height)
        start_y = height - start_y
        
        end_x = int((self.gps_points[-1].lon - min_lon) / (max_lon - min_lon) * width)
        end_y = int((self.gps_points[-1].lat - min_lat) / (max_lat - min_lat) * height)
        end_y = height - end_y
        
        if 0 <= start_x < width and 0 <= start_y < height:
            cv2.circle(map_image, (start_x, start_y), 3, (0, 255, 0), -1)  # Green start
        
        if 0 <= end_x < width and 0 <= end_y < height:
            cv2.circle(map_image, (end_x, end_y), 3, (255, 0, 0), -1)  # Blue end
        
        return map_image
    
    @staticmethod
    def process_frame_batch(args: Tuple[int, np.ndarray, float, float, float, List[GPSPoint], int, float, bool, str, bool]) -> Tuple[int, np.ndarray]:
        """Process a single frame - used for multiprocessing"""
        frame_number, frame, current_time, video_duration, gps_duration, gps_points, track_thickness, contrast_factor, show_map, google_api_key, map_only = args
        
        # Create a temporary processor instance for this frame
        temp_processor = VideoProcessor("", gps_points, google_api_key)
        processed_frame = temp_processor.draw_gps_overlay(frame, current_time, video_duration, gps_duration, track_thickness, contrast_factor, show_map, map_only)
        return frame_number, processed_frame
    
    def process_video_optimized(self, output_path: str, num_processes: int = None, 
                              track_thickness: int = 3, contrast_factor: float = 1.5, show_map: bool = True, map_only: bool = False):
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
                frames_data.append((frame_number, frame.copy(), current_time, video_duration, gps_duration, self.gps_points, track_thickness, contrast_factor, show_map, self.google_maps.api_key, map_only))
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
    parser.add_argument('-t', '--thickness', type=int, default=3, help='Track line thickness (default: 3)')
    parser.add_argument('-c', '--contrast', type=float, default=1.5, help='Contrast factor for overlay visibility (default: 1.5)')
    parser.add_argument('--no-map', action='store_true', help='Disable map overlay in video')
    parser.add_argument('--google-api-key', help='Google Maps API key for high-definition maps')
    parser.add_argument('--map-only', action='store_true', help='Show only GPS track and map (no video)')
    
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
        processor = VideoProcessor(args.video, gps_points, args.google_api_key)
        show_map = not args.no_map
        processor.process_video_optimized(args.output, args.processes, args.thickness, args.contrast, show_map, args.map_only)
        
    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.freeze_support()
    main()