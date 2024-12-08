import cv2
import argparse
import os

def create_tracker(tracker_type):
    if tracker_type == "BOOSTING":
        return cv2.legacy.TrackerBoosting_create()
    elif tracker_type == "CSRT":
        if hasattr(cv2, 'legacy'):
            return cv2.legacy.TrackerCSRT_create()
        else:
            return cv2.TrackerCSRT_create()
    elif tracker_type == "MIL":
        if hasattr(cv2, 'legacy'):
            return cv2.legacy.TrackerMIL_create()
        else:
            return cv2.TrackerMIL_create()
    else:
        raise ValueError(f"Unsupported tracker type: {tracker_type}")

def main(source, tracker_type):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open video source '{source}'.")
        exit()

    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read from video source.")
        cap.release()
        exit()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    if source in [0, 1, 2, 3]:
        out = cv2.VideoWriter(f'output/{tracker_type}_webcam_output.mp4', fourcc, fps, (width, height))
    else:
        out = cv2.VideoWriter(f'output/{tracker_type}_{os.path.basename(source)}_output.mp4', fourcc, fps, (width, height)) 

    bbox = cv2.selectROI("Select target to track", frame, False, False)
    cv2.destroyWindow("Select target to track")

    tracker = create_tracker(tracker_type)
    success = tracker.init(frame, bbox)

    start_time = cv2.getTickCount()  # Start time for overall FPS calculation

    frame_count = 0
    fps_display_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot capture frame.")
            break

        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Tracking failed!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        frame_count += 1
        current_time = cv2.getTickCount()
        elapsed_time = (current_time - start_time) / cv2.getTickFrequency()  # Convert to seconds

        if elapsed_time - fps_display_time >= 1.0:
            fps = frame_count / elapsed_time
            fps_display_time = elapsed_time
            print(f"FPS: {fps:.2f}")
            print(f"Frame count: {frame_count}")
            print(f"Elapsed time: {elapsed_time:.2f} seconds")

        # Change the color of the FPS text to blue (BGR format)
        cv2.putText(frame, f"FPS: {fps:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        out.write(frame)
        cv2.imshow(f"{tracker_type} Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = cv2.getTickCount()  # End time for overall FPS calculation
    overall_elapsed_time = (end_time - start_time) / cv2.getTickFrequency()  # Convert to seconds
    overall_fps = frame_count / overall_elapsed_time
    print(f"Overall FPS: {overall_fps:.2f}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Tracking using OpenCV")
    parser.add_argument("--source", type=str, required=True, help="Video source (file path or webcam index)")
    parser.add_argument("--tracker", type=str, required=True, choices=["BOOSTING", "CSRT", "MIL"], help="Tracking algorithm to use")
    args = parser.parse_args()  # Corrected method name

    source = args.source
    if source.isdigit():
        source = int(source)

    main(source, args.tracker)
