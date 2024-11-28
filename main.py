import cv2
import argparse

def main(source):
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
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    bbox = cv2.selectROI("Select target to track", frame, False, False)
    cv2.destroyWindow("Select target to track")

    if hasattr(cv2, 'legacy'):
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = cv2.TrackerCSRT_create()

    success = tracker.init(frame, bbox)

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

        out.write(frame)
        cv2.imshow("Object Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Tracking using OpenCV")
    parser.add_argument("--source", type=str, required=True, help="Video source (file path or webcam index)")
    args = parser.parse_args()
    source = args.source
    if source.isdigit():
        source = int(source)
    main(source)
