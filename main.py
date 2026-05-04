from utils.tracker import VideoTracker

def main():
    # Initialize the tracker with model weights and an adaptive noise threshold
    # A threshold of 15.0 is typically a good balance for detecting noise in low-light/grainy scenes
    tracker = VideoTracker(model_path="models/best.pt", noise_threshold=15.0)
    
    # Process the video through the frame-adaptive pipeline
    tracker.process_video(
        input_path="input/noisy.mp4",
        output_path="output/tracked_video.mp4"
    )

if __name__ == "__main__":
    main()
