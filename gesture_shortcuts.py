
from __future__ import annotations
import argparse
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from pynput.keyboard import Controller, Key

print("Program started")
try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit(
        "Failed to import mediapipe. Install with `pip install mediapipe`.\n"
        f"Original error: {e}"
    )
    

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

keyboard = Controller()

# ---------------------------- CONFIG ----------------------------
# Map normalized gestures to hotkeys. You can change these.
# Supported key names: regular characters, or Key.<name> from pynput
GESTURE_BINDINGS: Dict[str, Tuple] = {
    "FIST": ("ctrl`,"),              # Toggle integrated terminal (VS Code default: Ctrl+`)
    "ONE": ("ctrl+b",),               # Toggle Side Bar
    "TWO": ("ctrl+shift+m",),        # Problems panel
    "THREE": ("ctrl+k", "ctrl+s"),  # Keyboard Shortcuts (chord)
    "FOUR": ("ctrl+shift+p",),       # Command Palette
    "FIVE": ("alt+shift+f",),        # Format Document
    "PINCH": ("ctrl+s",),            # Save
}

DEBOUNCE_FRAMES = 5           # how many consecutive frames to confirm a gesture
COOLDOWN_SEC = 1.0            # min seconds between triggering the same gesture

# Pinch threshold: smaller is stricter. Value is distance (in % of frame diag)
PINCH_THRESHOLD = 0.045

# ---------------------------- UTILS ----------------------------

def parse_hotkey(ch: str):
    """Return a list of keys to press for a single hotkey like 'ctrl+shift+m'.
    Supports ctrl, shift, alt, cmd, win. Letters/symbols go as literals.
    """
    special = {
        "ctrl": Key.ctrl,
        "shift": Key.shift,
        "alt": Key.alt,
        "cmd": Key.cmd,
        "win": Key.cmd,
        "enter": Key.enter,
        "tab": Key.tab,
        "space": Key.space,
        "esc": Key.esc,
        "up": Key.up,
        "down": Key.down,
        "left": Key.left,
        "right": Key.right,
        "backspace": Key.backspace,
        "delete": Key.delete,
        "home": Key.home,
        "end": Key.end,
        "pageup": Key.page_up,
        "pagedown": Key.page_down,
        "f1": Key.f1, "f2": Key.f2, "f3": Key.f3, "f4": Key.f4, "f5": Key.f5,
        "f6": Key.f6, "f7": Key.f7, "f8": Key.f8, "f9": Key.f9, "f10": Key.f10,
        "f11": Key.f11, "f12": Key.f12,
        "`": '`',
    }
    parts = [p.strip().lower() for p in ch.split("+")]
    keys = []
    for p in parts:
        if p in special:
            keys.append(special[p])
        elif len(p) == 1:
            keys.append(p)
        else:
            # try single char from word (e.g., 'm')
            if len(p) > 1:
                keys.append(p[0])
    return keys


def press_hotkey_sequence(seq: Tuple[str, ...]):
    """Press a sequence of hotkeys. For chords like ("ctrl+k", "ctrl+s")."""
    for hotkey in seq:
        keys = parse_hotkey(hotkey)
        # Press modifiers first, then chars; release in reverse
        modifiers = [k for k in keys if isinstance(k, Key) or str(k) in {"\x11", "\x10", "\x12"}]
        normals = [k for k in keys if k not in modifiers]
        try:
            for k in modifiers:
                keyboard.press(k)
            for k in normals:
                keyboard.press(k)
            # brief hold
            time.sleep(0.05)
        finally:
            for k in reversed(normals):
                keyboard.release(k)
            for k in reversed(modifiers):
                keyboard.release(k)
        time.sleep(0.08)


@dataclass
class GestureState:
    current: Optional[str] = None
    last_confirmed: Optional[str] = None
    frames_held: int = 0
    last_trigger_time: float = 0.0


# ---------------------------- GESTURE LOGIC ----------------------------

def finger_states_from_landmarks(landmarks, handedness: str, img_w: int, img_h: int):
    """Return dict {thumb,index,middle,ring,pinky: bool_up} using landmark geometry.
    Based on MCP vs TIP positions (y for non-thumb, x for thumb, adjusted by handedness).
    """
    # Landmark indices
    TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
    MCP = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}

    pts = np.array([(int(lm.x * img_w), int(lm.y * img_h)) for lm in landmarks])

    states = {}
    # Thumb: compare x relative to MCP depending on hand side
    is_right = handedness.lower().startswith("right")
    thumb_tip_x = pts[TIP["thumb"]][0]
    thumb_mcp_x = pts[MCP["thumb"]][0]
    if is_right:
        states["thumb"] = thumb_tip_x < thumb_mcp_x  # right hand: left means extended
    else:
        states["thumb"] = thumb_tip_x > thumb_mcp_x  # left hand: right means extended

    # Other fingers: tip above MCP (smaller y) => extended
    for name in ["index", "middle", "ring", "pinky"]:
        tip_y = pts[TIP[name]][1]
        mcp_y = pts[MCP[name]][1]
        states[name] = tip_y < mcp_y - 4  # small bias

    return states, pts


def count_fingers_up(states: Dict[str, bool]) -> int:
    return sum(1 for v in states.values() if v)


def compute_pinch(pts: np.ndarray, img_w: int, img_h: int) -> float:
    """Return normalized pinch distance between thumb tip (4) and index tip (8)
    relative to frame diagonal (0..~1). Smaller means closer => pinch.
    """
    t = pts[4]
    i = pts[8]
    dist = np.linalg.norm(t - i)
    diag = np.sqrt(img_w ** 2 + img_h ** 2)
    return float(dist / diag)


def recognize_gesture(states: Dict[str, bool], pinch_norm: float) -> str:
    fingers = count_fingers_up(states)
    if pinch_norm < PINCH_THRESHOLD:
        return "PINCH"
    mapping = {
        0: "FIST",
        1: "ONE",
        2: "TWO",
        3: "THREE",
        4: "FOUR",
        5: "FIVE",
    }
    return mapping.get(fingers, "UNKNOWN")


# ---------------------------- MAIN LOOP ----------------------------

def run(preview: bool = True, camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Try a different --camera index.")

    # Improve camera performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_FPS, 30)

    gesture_state = GestureState()

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        prev_time = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            img_h, img_w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            gesture_name = "NONE"

            if results.multi_hand_landmarks and results.multi_handedness:
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0].classification[0].label
                states, pts = finger_states_from_landmarks(
                    hand_landmarks.landmark, handedness, img_w, img_h
                )
                pinch_norm = compute_pinch(pts, img_w, img_h)
                gesture_name = recognize_gesture(states, pinch_norm)

                # Draw
                if preview:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                # Debounce + cooldown
                now = time.time()
                if gesture_name == gesture_state.current:
                    gesture_state.frames_held += 1
                else:
                    gesture_state.current = gesture_name
                    gesture_state.frames_held = 1

                if (
                    gesture_name in GESTURE_BINDINGS
                    and gesture_state.frames_held >= DEBOUNCE_FRAMES
                    and (now - gesture_state.last_trigger_time) >= COOLDOWN_SEC
                ):
                    # Trigger
                    seq = GESTURE_BINDINGS[gesture_name]
                    press_hotkey_sequence(seq)
                    gesture_state.last_trigger_time = now
                    gesture_state.last_confirmed = gesture_name

                if preview:
                    # HUD
                    cv2.rectangle(frame, (0, 0), (img_w, 60), (0, 0, 0), -1)
                    cv2.putText(
                        frame,
                        f"Gesture: {gesture_name}",
                        (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Held: {gesture_state.frames_held}  Last: {gesture_state.last_confirmed}",
                        (10, 46),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
            else:
                gesture_state.current = None
                gesture_state.frames_held = 0

            if preview:
                # FPS
                now = time.time()
                fps = 1.0 / max(1e-6, (now - prev_time))
                prev_time = now
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (img_w - 120, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Gesture Shortcuts", frame)
                if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):  # q or ESC
                    break

        cap.release()
        if preview:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Gesture Shortcuts Controller")
    parser.add_argument("--no-preview", action="store_true", help="Run headless (no window)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    args = parser.parse_args()

    try:
        run(preview=not args.no_preview, camera_index=args.camera)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
