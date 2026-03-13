import cv2
import mediapipe as mp
from ultralytics import YOLO

MODEL_PATH = "runs/classify/train3/weights/best.pt"
WEBCAM_INDEX = 0

LABELS = {
    0: "alapadma", 1: "arala", 2: "kapitha", 3: "kartarimukha",
    4: "katakamukha-1", 5: "katakamukha-2", 6: "katakamukha-3",
    7: "mayura", 8: "mrighashisha", 9: "mukula", 10: "mushthi",
    11: "padmakosha", 12: "ardhachandra", 13: "pataka", 14: "santamsha",
    15: "sarpashisha", 16: "shikhara", 17: "shukatunda", 18: "singhamukha",
    19: "suchi", 20: "tamrachuda", 21: "tripataka", 22: "trishula",
    23: "ardhapataka", 24: "bhramhara", 25: "chandrakala", 26: "chatura",
    27: "hamsapaksha", 28: "hamsasya", 29: "kangula",
}


def get_hand_box(hand_landmarks, h, w):
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    x1, x2 = int(min(xs) * w), int(max(xs) * w)
    y1, y2 = int(min(ys) * h), int(max(ys) * h)

    side = max(x2 - x1, y2 - y1)
    pad = 20
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = side // 2 + pad

    sq_x1 = max(cx - half, 0)
    sq_y1 = max(cy - half, 0)
    sq_x2 = min(cx + half, w)
    sq_y2 = min(cy + half, h)

    return (x1, y1, x2, y2), (sq_x1, sq_y1, sq_x2, sq_y2)


def predict(model, crop):
    cv2.imwrite("saved_image.jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    results = model("saved_image.jpg")
    return LABELS[results[0].probs.top1]


def main():
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    print("Loading MediaPipe hand detector...")
    hands = mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    print(f"Opening webcam (index {WEBCAM_INDEX})...")
    cap = cv2.VideoCapture(WEBCAM_INDEX)

    if not cap.isOpened():
        print(f"Could not open webcam at index {WEBCAM_INDEX}. Try changing WEBCAM_INDEX at the top of this file.")
        return

    print("Ready! A window should appear — check your Dock if you don't see it.")
    print("Hold up a hand gesture to see predictions. Press Esc to quit.")

    cv2.namedWindow("Mudra Recognition", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("Mudra Recognition", 100, 100)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks[:2]:
                (x1, y1, x2, y2), (sx1, sy1, sx2, sy2) = get_hand_box(hand_lm, h, w)
                crop = rgb[sy1:sy2, sx1:sx2]
                if crop.size == 0:
                    continue
                name = predict(model, crop)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 3)

        cv2.imshow("Mudra Recognition", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
