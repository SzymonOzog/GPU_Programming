import cv2
import sys

def play_videos(video_paths):
    while len(video_paths):
        video_path = video_paths.pop(0)
        print(f"\nPlaying video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            continue
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = int(1000 / fps) if fps > 0 else 33
        
        while cv2.waitKey(frame_delay) != 32:
            ret, frame = cap.read()
            if ret:
                cv2.imshow('Video Presentation', frame)
        cap.release()
        
    cv2.destroyAllWindows()
    print("All videos have been played.")

if __name__ == "__main__":
    video_paths = [
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_2.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_3.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_4.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_5.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_6.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_7.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_8.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_9.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_10.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_11.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_12.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_13.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_14.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_15.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_16.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_17.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_18.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_19.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_20.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_21.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_22.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_23.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_24.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_25.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_26.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_27.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_28.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_29.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_30.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_31.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_32.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_33.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_34.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_35.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_36.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_37.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_38.mp4",
            "./media/videos/10_Memory_Coalescing/2160p30/Coalescing_segment_39.mp4",

            "./media/videos/11_Occupancy/2160p30/Occupancy_segment_1.mp4",
            "./media/videos/11_Occupancy/2160p30/Occupancy_segment_2.mp4",
            "./media/videos/11_Occupancy/2160p30/Occupancy_segment_3.mp4",
            "./media/videos/11_Occupancy/2160p30/Occupancy_segment_4.mp4",

            "./videos/TensorCores_segment_4.mp4",

            "./videos/HierarchicalTiling_segment_10.mp4",
            "./videos/HierarchicalTiling_segment_11.mp4",
            "./videos/HierarchicalTiling_segment_12.mp4",
            "./videos/HierarchicalTiling_segment_13.mp4",
            "./videos/HierarchicalTiling_segment_14.mp4",
            "./videos/HierarchicalTiling_segment_15.mp4",
            "./videos/HierarchicalTiling_segment_16.mp4",
            "./videos/HierarchicalTiling_segment_17.mp4",
            "./videos/HierarchicalTiling_segment_18.mp4",


            "./videos/Quantization_segment_1.mp4",
            "./videos/Quantization_segment_2.mp4",
            "./videos/Quantization_segment_3.mp4",
            "./videos/Quantization_segment_4.mp4",
            "./videos/Quantization_segment_5.mp4",
            "./videos/Quantization_segment_6.mp4",
            "./videos/Quantization_segment_7.mp4",
            "./videos/Quantization_segment_8.mp4",
            "./videos/Quantization_segment_9.mp4",
            "./videos/Quantization_segment_10.mp4",
            "./videos/Quantization_segment_11.mp4",
            "./videos/Quantization_segment_12.mp4",
            "./videos/Quantization_segment_13.mp4",
            "./videos/Quantization_segment_14.mp4",

            "./videos/MoE_segment_1.mp4",
            "./videos/MoE_segment_2.mp4",
            "./videos/MoE_segment_3.mp4",
            "./videos/MoE_segment_4.mp4",

    ]
    play_videos(video_paths)
