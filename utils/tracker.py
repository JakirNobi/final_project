import os
import cv2
from ultralytics import YOLO

class VideoTracker:
    def __init__(self, model_path):
        """
        Initializes the tracker with the specified YOLO model.
        """
        self.model = YOLO(model_path)

    def process_video(self, input_path, output_path, crop_dir="tracking_pics"):
        """
        Reads a video, applies tracking on each frame, and saves the output.
        Also crops and saves detected bounding boxes into crop_dir.
        """
        # Create the crop directory if it doesn't exist
        os.makedirs(crop_dir, exist_ok=True)

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print(f"Error: Could not open video at {input_path}")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        print(f"Starting tracking on {input_path}...")
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1

            # Run YOLO inference with tracking on the current frame
            results = self.model.track(frame, persist=True, verbose=False)

            # Crop and save the detected frames according to the bounding boxes
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Ensure coordinates are within frame bounds
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        cropped_img = frame[y1:y2, x1:x2]
                        crop_filename = os.path.join(crop_dir, f"id_{track_id}_frame_{frame_count}.jpg")
                        cv2.imwrite(crop_filename, cropped_img)

            # Plot the tracking results on the frame
            annotated_frame = results[0].plot()

            # Write the annotated frame to the output video
            out.write(annotated_frame)

            # Display the frame (Comment these 3 lines out if running on a headless server without GUI)
            # cv2.imshow("Tracking", annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Tracking completed. Processed {frame_count} frames. Output saved to {output_path}")
