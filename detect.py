import cv2
from ultralytics import YOLO

#This function extracts the frame at the middle of the vid
def extract_keyframe(video_path):
    cap = cv2.VideoCapture(video_path)
    #check for any errors
    if not cap.isOpened():
        print("Error: cannot open video")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #getting total count of frames to find middle frame
    if total_frames == 0:
        print("Error: video has no frames")
        return None

    frame_idx = total_frames // 2  # Middle frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx) #setting and reading the middle frae
    ok, frame = cap.read()
    cap.release() #releasing to free resources

    if not ok:
        print("Error: cannot read frame")
        return None
    return frame

#checks for the objects weve labelled as prohibited
def detect_objects(frame, model, labels, conf_thresh=0.35):
    found = set()
    results = model(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            conf = float(box.conf) #checks confidence of every detection and only considers it when above 0.35
            cls = int(box.cls) #get class of detected objects
            name = r.names.get(cls, "unknown") #gets the objects that have been detected
            if conf > conf_thresh and name in labels:
                found.add(name) #returns detections that have passed the threshold
    return found

#Main code snippet that calls other functions
def check_video(video_path):
    print(f"Checking video: {video_path}")

    # Load models
    weapons_model = YOLO("./runs/detect/Normal_Compressed/weights/best.pt")  # yolo model trained extensively on weapons dataset
    # Model: https://github.com/JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8
    general_model = YOLO("yolov8n.pt")  # general-purpose YOLOv8 model

    # list of prohibited items
    prohibited = {
        "weapons": ["knife", "knives", "gun", "guns", "pistol", "rifle", "sword"],
        "violence": ["fight", "punch", "kick"],
        "explicit": ["nude", "sexual", "porn", "sex"],
        "explosives": ["bomb", "grenade", "explosive", "tnt", "dynamite"]
    }

    # Step 1: Extract keyframe
    frame = extract_keyframe(video_path)
    if frame is None:
        return {"status": "error", "items": [], "message": "Failed to get keyframe"}

    # Step 2: Detect weapons with weapons model
    detected_weapons = detect_objects(frame, weapons_model, prohibited["weapons"])

    # Step 3: Detect other categories with general model
    detected_other = set()
    for cat in ["violence", "explicit", "explosives"]:
        items = detect_objects(frame, general_model, prohibited[cat])
        detected_other.update(items)

    # Combine all detections
    all_found = detected_weapons.union(detected_other)

    # Step 4: Report
    if all_found:
        print("Prohibited items found:")
        for item in all_found:
            print("-", item)
        return {"status": "flagged", "items": list(all_found)}
    else:
        print("No prohibited items detected.")
        return {"status": "safe", "items": []}

video_file = "3.mp4"
result = check_video(video_file)
print("Result:", result)
