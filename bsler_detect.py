import cv2
from ultralytics import YOLO
import time
import threading
import numpy as np
import torch
import urllib.parse
from collections import Counter
from pypylon import pylon
from pypylon import pylon


# Import the stream server components directly
from stream_server import DualStreamHandler, ThreadedHTTPServer
from frame_manager import FrameManager

def open_basler_camera(index=0):
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()
    if len(devices) <= index:
        raise RuntimeError("Camera index {} not found".format(index))

    camera = pylon.InstantCamera(factory.CreateDevice(devices[index]))
    camera.Open()

    # Disable buffering and always grab the latest frame
    camera.MaxNumBuffer = 15
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    return camera


# Create a converter once (outside the function for efficiency)
rgb_converter = pylon.ImageFormatConverter()
rgb_converter.InputPixelFormat = pylon.PixelType_Data8 
rgb_converter.OutputPixelFormat = pylon.PixelType_BGR8packed
rgb_converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned


def grab_latest_frame(camera):
    if not camera.IsGrabbing():
        return None

    grab_result = camera.RetrieveResult(50, pylon.TimeoutHandling_Return)  # ↓ reduced timeout
    if not grab_result:
        return None

    try:
        if grab_result.GrabSucceeded():
            converted = rgb_converter.Convert(grab_result)
            return converted.GetArray()  # (H, W, 3) uint8 — ready for YOLO
        return None
    finally:
        grab_result.Release()  # Always released


def start_stream_server(frame_manager):
    """Start the stream server in a separate thread"""
    print("[INFO] Starting stream server...")

    try:
        # Create the server with our frame manager
        server = ThreadedHTTPServer(('0.0.0.0', 8000),
                                    lambda *args, **kwargs: DualStreamHandler(*args, frame_manager=frame_manager, **kwargs))

        # Start server in a thread
        def serve_forever():
            print("🚀 Stream Server Started")
            print("📍 Main Page: http://localhost:8000")
            print("📹 Stream 1:  http://localhost:8000/stream1")
            print("📹 Stream 2:  http://localhost:8000/stream2")
            print("🎥 Combined:  http://localhost:8000/both")
            server.serve_forever()

        server_thread = threading.Thread(target=serve_forever, daemon=True)
        server_thread.start()

        # Wait for server to initialize
        time.sleep(2)
        print("[INFO] Stream server started successfully")
        return server, server_thread
    except Exception as e:
        print(f"[ERROR] Failed to start stream server: {e}")
        return None, None

def build_rtsp_url():
    print("=== IP Camera Login ===")
    username = "admin"
    password = "FT787814"
    ip = "192.168.1.65"
    port = 554
    path = "stream1"

    username_enc = urllib.parse.quote(username)
    password_enc = urllib.parse.quote(password)

    url = f"rtsp://{username_enc}:{password_enc}@{ip}:{port}/{path}"
    print(f"[INFO] Using RTSP URL: {url}")
    return url

def resize_frame(frame, target_width=640, target_height=480):
    """Resize frame to target dimensions while maintaining aspect ratio"""
    if frame is None:
        return None

    h, w = frame.shape[:2]

    # If already target size, return as is
    if w == target_width and h == target_height:
        return frame

    # Calculate scaling factors
    scale_x = target_width / w
    scale_y = target_height / h
    scale = min(scale_x, scale_y)

    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize
    print(frame.ndim)
    resized = cv2.resize(frame, (new_w, new_h))

    # Pad to target size if needed
    if new_w != target_width or new_h != target_height:
        pad_x = (target_width - new_w) // 2
        pad_y = (target_height - new_h) // 2

        # Create black background
        padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)

        # Place resized image in center
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        return padded

    return resized

def main():
    # Initialize frame manager for sharing frames
    frame_manager = FrameManager()

    # Start the stream server with our frame manager
    server, server_thread = start_stream_server(frame_manager)
    if server is None:
        print("[ERROR] Could not start stream server. Exiting.")
        return


    print("[INFO] Loading YOLOv8 model (yolov8n.pt)...")
#    model = YOLO("models/yolov8n_fish_trained.pt")
    model = YOLO("models/yolov8_side_fish_detector_augmented_251030.pt")
    if torch.cuda.is_available():
        model.model.to("cuda:0")
        print("[INFO] Using CUDA acceleration")
    else:
        print("[INFO] Using CPU")

    #print(f"[INFO] Opening camera stream: {camera_url}")

    # Open camera without changing resolution settings
    cam = open_basler_camera(0)



    # Get actual camera resolution (native resolution)
    #actual_width = int(.get(cv2.CAP_PROP_FRAME_WIDTH))
    #actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #actual_fps = cap.get(cv2.CAP_PROP_FPS)

    #print(f"[INFO] Camera native resolution: {actual_width}x{actual_height}")
    #print(f"[INFO] Camera FPS: {actual_fps}")
    #print(f"[INFO] Output resolution: 640x480")

    prev_time = 0
    frame_count = 0
    fps_history = []

    print("[INFO] Starting YOLO detection and frame sharing...")
    print("[INFO] Press Ctrl+C to stop")
    count_index = 0
    try:
        while True:
            frame = grab_latest_frame(cam)
            if frame is None:
                print("[WARN] Failed to grab frame")
                continue

            # Run YOLO detection on original high-resolution frame
            results = model(frame, verbose=False)
            #print("results: ", results)

            # Count detected object names
            names = [det.cls for det in results[0].boxes]
            class_names = [results[0].names[int(cls)] for cls in names]
            
            counts = Counter(class_names)

            # Draw annotations on ORIGINAL high-resolution frame first
            annotated_high_res = results[0].plot()

            # Then RESIZE to 640x480 for streaming
            try:
                annotated = resize_frame(annotated_high_res, 640, 480)
            except:
                annotated = annotated_high_res
            # FPS calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # Add overlays to resized annotated frame
            if annotated is not None:
                # Display object counts
                y0 = 60
                for obj, count in counts.items():
                    cv2.putText(annotated, f"{obj}: {count}", (20, y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    y0 += 25

                # SHARE THE RESIZED FRAME with the streaming server
                frame_manager.update_frame1(annotated)

            frame_count += 1

            # Log every 30 frames
            if frame_count % 10 == 0:
                
                print(f"Frame {frame_count}: FPS: {avg_fps:.1f}, Objects: {dict(counts)}")

            # Control frame rate
            #time.sleep(0.01)  # ~100 FPS max

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
    finally:
        #cam.release()
        cv2.destroyAllWindows()
        print("[INFO] Application stopped")

if __name__ == "__main__":
    main()
