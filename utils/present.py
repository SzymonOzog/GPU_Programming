import cv2
import sys
import subprocess
import ffmpeg
import os

def play_videos(video_paths):
    cv2.namedWindow("Video Presentation", cv2.WINDOW_NORMAL)
    while len(video_paths):
        video_path = video_paths.pop(0)
        print(f"\nPlaying video: {video_path}")
        
        is_video = ".mp4" in video_path

        if is_video:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                continue
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_delay = int(1000 / fps) if fps > 0 else 33

        else:
            img = cv2.imread(video_path)
        
        while cv2.waitKey(frame_delay) != 32:
            (x, y, w, h) = cv2.getWindowImageRect("Video Presentation")
            if is_video:
                ret, frame = cap.read() 
            else:
                ret, frame = True, img

            if ret:
                if h > 0 and w > 0:
                    frame = cv2.resize(frame, (w,h))
                cv2.imshow('Video Presentation', frame)

        if is_video:
            cap.release()
        
    cv2.destroyAllWindows()
    print("All videos have been played.")

def convert_videos_fps(video_paths, input_fps=60, output_fps=30, output_dir=None):
    """
    Convert videos from one framerate to another using ffmpeg.
    
    Args:
        video_paths (list): List of paths to video files
        input_fps (int): Input framerate (default: 60)
        output_fps (int): Output framerate (default: 30)
        output_dir (str, optional): Directory to save output videos. If None, videos are saved in the same directory with '_30fps' suffix
        
    Returns:
        list: Paths to converted video files
    """
    # Check if ffmpeg is installed
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg before running this function.")
    
    converted_paths = []
    
    for video_path in video_paths:
        if not os.path.exists(video_path):
            print(f"Warning: File not found - {video_path}")
            continue
            
        # Get file details
        file_dir, file_name = os.path.split(video_path)
        file_name_no_ext, file_ext = os.path.splitext(file_name)
        
        # Determine output path
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{file_name_no_ext}_{output_fps}fps{file_ext}")
        else:
            output_path = os.path.join(file_dir, f"{file_name_no_ext}_{output_fps}fps{file_ext}")
        
        try:
            # Process the video
            print(f"Converting {video_path} to {output_fps} fps...")
            
            # Use ffmpeg-python to convert the video
            (
                ffmpeg
                .input(video_path)
                .filter('fps', fps=output_fps)
                .output(output_path, crf=18)
                .run(overwrite_output=True, quiet=True)
            )
            
            print(f"Successfully converted! Output saved to: {output_path}")
            converted_paths.append(output_path)
            
        except ffmpeg.Error as e:
            print(f"Error converting {video_path}: {e.stderr.decode() if e.stderr else str(e)}")
        except Exception as e:
            print(f"Unexpected error converting {video_path}: {str(e)}")
    
    return converted_paths


if __name__ == "__main__":
    video_paths = [
            "./videos/Speed_segment_1.mp4",
            "./videos/Speed_segment_2.mp4",
            "./videos/Speed_segment_3.mp4",
            "./videos/Speed_segment_4.mp4",

            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_2_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_3_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_4_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_5_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_6_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_7_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_8_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_9_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_10_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_11_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_12_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_13_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_14_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_15_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_16_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_17_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_18_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_19_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_20_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_21_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_22_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_23_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_24_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_25_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_26_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_27_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_28_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_29_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_30_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_31_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_32_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_33_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_34_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_35_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_36_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_37_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_38_30fps.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_39_30fps.mp4",

            "/Users/szymon.ozog/Downloads/front.jpg",
            "/Users/szymon.ozog/Downloads/memory.jpg",

            "./media/videos/11_Occupancy/2160p30/Occupancy_segment_1_30fps.mp4",
            "./media/videos/11_Occupancy/2160p30/Occupancy_segment_2_30fps.mp4",
            "./media/videos/11_Occupancy/2160p30/Occupancy_segment_3_30fps.mp4",
            "./media/videos/11_Occupancy/2160p30/Occupancy_segment_4_30fps.mp4",

            "./videos/TensorCores_segment_4_30fps.mp4",

            "./videos/HierarchicalTiling_segment_9_30fps.mp4",
            "./videos/HierarchicalTiling_segment_10_30fps.mp4",
            "./videos/HierarchicalTiling_segment_11_30fps.mp4",
            "./videos/HierarchicalTiling_segment_12_30fps.mp4",
            "./videos/HierarchicalTiling_segment_13_30fps.mp4",
            "./videos/HierarchicalTiling_segment_14_30fps.mp4",
            "./videos/HierarchicalTiling_segment_15_30fps.mp4",
            "./videos/HierarchicalTiling_segment_16_30fps.mp4",
            "./videos/HierarchicalTiling_segment_17_30fps.mp4",
            "./videos/HierarchicalTiling_segment_18_30fps.mp4",


            "./videos/Quantization_segment_1_30fps.mp4",
            "./videos/Quantization_segment_2_30fps.mp4",
            "./videos/Quantization_segment_3_30fps.mp4",
            "./videos/Quantization_segment_4_30fps.mp4",
            "./videos/Quantization_segment_5_30fps.mp4",
            "./videos/Quantization_segment_6_30fps.mp4",
            "./videos/Quantization_segment_7_30fps.mp4",
            "./videos/Quantization_segment_8_30fps.mp4",
            "./videos/Quantization_segment_9_30fps.mp4",
            "./videos/Quantization_segment_10_30fps.mp4",
            "./videos/Quantization_segment_11_30fps.mp4",
            "./videos/Quantization_segment_12_30fps.mp4",
            "./videos/Quantization_segment_13_30fps.mp4",
            "./videos/Quantization_segment_14_30fps.mp4",

            "./videos/MoE_segment_2_30fps.mp4",
            "./videos/MoE_segment_3_30fps.mp4",
            "./videos/MoE_segment_4_30fps.mp4",

    ]
    # convert_videos_fps(video_paths)
    play_videos(video_paths)

