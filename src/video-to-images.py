import cv2
import sys
#
#
#
#README  to utilize this script video must be in mp4 format,, from terminal run
#     python video-to-images.py <input video path> <output path>
#
def video_to_frames(video_path, output_path):

    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        cv2.imwrite(f"{output_path}/frame_{frame_count}.jpg", frame)
        frame_count += 1

    video.release()

def main():
    if len(sys.argv) != 3:
        print("Usage: python videotoframes.py <input_video_path> <output_frames_directory>")
        sys.exit(1)

    input_video_path = sys.argv[1]
    output_frames_directory = sys.argv[2]

    video_to_frames(input_video_path, output_frames_directory)

if __name__ == "__main__":
    main()
