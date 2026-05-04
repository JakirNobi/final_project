from utils.tracker import VideoTracker

def main():
    # Initialize the tracker with the model weights
    tracker = VideoTracker(model_path="models/best.pt")
    
    # Process the video
    tracker.process_video(
        input_path="input/215475.mp4",
        output_path="output/tracked_video.mp4"
    )

if __name__ == "__main__":
    main()
