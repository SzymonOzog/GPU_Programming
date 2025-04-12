import argparse
import os
from typing import List, Tuple
import subprocess
import ffmpeg


def extract_video_segment(video_path: str, start_time: float, end_time: float, output_path: str) -> None:
    """
    Extract a segment from a video file based on start and end timestamps using ffmpeg.
    
    Args:
        video_path: Path to the input video file
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Path to save the output video segment
    """
    try:
        # Calculate duration
        duration = end_time - start_time
        
        # Use ffmpeg-python
        (
            ffmpeg
            .input(video_path, ss=start_time, t=duration)
            .output(output_path, c='copy')
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        print(f"Successfully created segment: {output_path}")
    except ffmpeg.Error as e:
        print(f"Error creating segment {output_path}: {e.stderr.decode()}")
    except Exception as e:
        print(f"Error creating segment {output_path}: {str(e)}")


def split_video(video_path: str, timestamps: List[Tuple[float, float]]) -> None:
    """
    Split a video into multiple segments based on a list of timestamp tuples.
    
    Args:
        video_path: Path to the input video file
        timestamps: List of (start_time, end_time) tuples in seconds
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found")
        return
    
    # Get video filename without extension
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.dirname(os.path.abspath(video_path))
    
    # Process each timestamp tuple
    for i, (start_time, end_time) in enumerate(timestamps):
        # Validate timestamps
        if start_time >= end_time:
            print(f"Warning: Skipping segment {i+1} - start time must be less than end time")
            continue
        
        # Create output filename
        output_filename = f"{base_name}_segment_{i+1}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        if os.path.exists(output_path):
            print("removing ", output_path)
            os.unlink(output_path)
        
        # Extract the segment
        extract_video_segment(video_path, start_time, end_time, output_path)


def parse_timestamp_tuple(timestamp_str: str) -> Tuple[float, float]:
    """
    Parse a string representation of a timestamp tuple into a tuple of floats.
    Format: "start_time,end_time" in seconds
    
    Args:
        timestamp_str: String representation of timestamp tuple (e.g., "10.5,25.0")
        
    Returns:
        Tuple of (start_time, end_time) as floats
    """
    out = []
    try:
        for x in timestamp_str.split(";"):
            start, end = x.split(',')
            out.append((float(start.strip()), float(end.strip())))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid timestamp format: {timestamp_str}. Use 'start,end' format (e.g., '10.5,25.0')"
        )
    print(out)
    return tuple(out)


def check_ffmpeg_installed():
    """Check if ffmpeg is installed on the system."""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description='Split a video into segments based on timestamps')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument(
        'timestamps', 
        type=str, 
        help='List of timestamp tuples (start,end) in seconds. Example: 10,30 60,90'
    )
    
    args = parser.parse_args()
    
    # Check if ffmpeg is installed
    if not check_ffmpeg_installed():
        print("Error: ffmpeg is not installed or not in the PATH.")
        print("Please install ffmpeg before running this script.")
        return
    
    # Check if ffmpeg-python is installed
    try:
        import ffmpeg
    except ImportError:
        print("Error: ffmpeg-python package is not installed.")
        print("Please install it with: pip install ffmpeg-python")
        return
    
    split_video(args.video_path, parse_timestamp_tuple(args.timestamps))


if __name__ == "__main__":
    main()
