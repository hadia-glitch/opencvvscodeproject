# Gesture Shortcuts for VS Code

Control **Visual Studio Code** using **hand gestures** with your webcam!  
This project uses **MediaPipe Hands** + **OpenCV** + **PyAutoGUI** to recognize gestures and trigger keyboard shortcuts in VS Code.

---

##  Features
-  Detect hand gestures in real-time using webcam.
- Map gestures to VS Code shortcuts .
- Fully customizable gesture → action mappings.

---

##  Installation

### 1️ Clone the Repository

git clone https://github.com/yourusername/gesture-shortcuts-vscode.git
cd gesture-shortcuts-vscode

### 2 Create a Virtual Environment
python -m venv myvenv301

Activate the environment:

Windows (PowerShell)

myvenv301\Scripts\activate


macOS/Linux

source myvenv301/bin/activate

### 3  Install Dependencies
pip install -r requirements.txt

requirements:
opencv-python
mediapipe
pyautogui
numpy

### 3 Usage
Run the main script:

python gesture_shortcuts.py

### Current Mappings
"FIST": ("ctrl,"), # Toggle integrated terminal 
"ONE": ("ctrl+b",), # Toggle Side Bar
"TWO": ("ctrl+shift+m",), # Problems panel
"THREE": ("ctrl+k", "ctrl+s"), # Keyboard Shortcuts (chord)
"FOUR": ("ctrl+shift+p",), # Command Palette 
"FIVE": ("alt+shift+f",),# Format Document 
"PINCH": ("ctrl+s",), # Save

### Screenshots
<img width="1492" height="908" alt="Screenshot 2025-08-28 220945" src="https://github.com/user-attachments/assets/533b1f38-7fe9-409d-aa8e-ca6b0087ffb2" />




