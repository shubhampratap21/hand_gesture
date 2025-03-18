# **Hand Gesture Audio Controller**

### *OpenCV, MediaPipe Hands, PyDub, FFmpeg*

- **Overview**: A computer vision-based hand gesture audio controller that adjusts speed and volume based on the distance between fingers.  
- **Core Functionalities**:
  - **Volume & Speed Control**: Adjusts volume and playback speed using hand gestures.  
  - **Play/Pause & Mute**: Enables play/pause and mute functionalities via gestures.  
  - **Skip & Rewind**: Allows skipping to the next song or rewinding the current track.  
  - **Multiple Songs**: Supports uploading multiple songs for the skip feature.  
- **Technologies Used**:
  - OpenCV for image processing  
  - MediaPipe Hands for hand tracking  
  - PyDub & FFmpeg for audio control  
- **Usage**:
  1. Run the script and position your hand in front of the camera.  
  2. Use predefined gestures to control playback.  
  3. Upload songs to enable the skip feature.  

### **Installation**
```sh
pip install opencv-python mediapipe pydub
