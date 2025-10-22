# Video GPS Overlay Tool

A Python tool that overlays GPS track data from GPX files onto video files with automatic time alignment.

## Features

- Parses GPX files to extract GPS track points with timestamps
- Automatically aligns video and GPS durations by extending the shorter one
- Overlays GPS track visualization on video frames
- Shows current position, track path, and timing information
- Handles different GPX namespace formats

## Installation

1. Install Python 3.7+
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```bash
python video_gps_overlay.py <video_file> <gpx_file> [-o output_file]
```

### Example

```bash
python video_gps_overlay.py DJI_20251022093950_0003_D.MP4 It_is_actually_a_short_run.gpx -o output_with_gps.mp4
```

## Parameters

- `video`: Input video file path (MP4, AVI, etc.)
- `gpx`: Input GPX file path
- `-o, --output`: Output video file path (default: output_with_gps_overlay.mp4)

## How it works

1. **GPX Parsing**: Extracts GPS track points with timestamps from the GPX file
2. **Time Alignment**: Compares video and GPS durations, aligns them by stretching the shorter one
3. **GPS Overlay**: Projects GPS coordinates to screen coordinates and overlays the track
4. **Real-time Position**: Shows current GPS position as a red dot with white border
5. **Track Visualization**: Displays the complete GPS track with past points in green and future points in gray

## Output

The tool creates a new video file with:
- GPS track overlay showing the complete path
- Current position indicator (red dot with white border)
- GPS coordinates and timing information
- Progress visualization with past/future track segments
