# Autonomous-Search-Bot
project:
  name: Autonomous Search Bot (Vision + Voice Enabled)
  description: >
    An intelligent ROS2-based robot that autonomously explores an environment,
    detects user-specified objects using YOLOv8, and interacts via voice commands
    to continue search or return after task completion.

key_features:
  - Autonomous exploration using frontier-based search
  - Real-time object detection using YOLOv8
  - Vision processing with OpenCV
  - Voice-controlled task input (Whisper)
  - Audio feedback using eSpeak
  - Dynamic goal switching based on detection results
  - Human-robot interaction loop (search → detect → ask → act)

tech_stack:
  middleware: ROS2 (Jazzy)
  language: Python (rclpy)/C++(cpp)
  slam: slam_toolbox
  navigation: nav2
  perception: YOLOv8 + OpenCV
  speech_to_text: OpenAI Whisper(base model)
  text_to_speech: eSpeak
  simulator: gazebo

perception_system:

  object_detection:
    model: YOLOv8 (Ultralytics)
    input: Camera image (/camera/image_raw)
    output: Detected objects with bounding boxes and labels

    installation:
      - pip install ultralytics
      - pip install opencv-python

    usage:
      description: >
        Continuously processes camera feed and detects target objects
      output_topic: /detected_objects

  vision_processing:
    library: OpenCV
    tasks:
      - Image preprocessing
      - Frame conversion (ROS → OpenCV)
      - Bounding box visualization

voice_system:

  speech_to_text:
    model: OpenAI Whisper

    installation:
      - pip install openai-whisper
      - pip install sounddevice
      - sudo apt install ffmpeg

    usage:
      description: Converts user speech into commands
      example_commands:
        - "find bottle"
        - "find person"
        - "stop search"

  text_to_speech:
    engine: eSpeak

    installation:
      - sudo apt install espeak

    usage:
      description: Provides feedback and asks next action

interaction_loop:

  description: >
    Continuous human-robot interaction cycle after object detection

  steps:
    - Robot searches environment autonomously
    - YOLO detects target object
    - Robot stops and announces detection
    - Robot asks user: "Object found. Return or continue?"
    - User responds via voice
    - Robot executes command

system_workflow:
  - step: 1
    name: Receive Target
    description: User says object name (e.g., "find bottle")

  - step: 2
    name: Exploration
    description: Robot explores using frontier-based navigation

  - step: 3
    name: Detection
    description: YOLOv8 processes camera feed continuously

  - step: 4
    name: Target Found
    description: Match detected object with user query

  - step: 5
    name: Decision Loop
    description: Ask user for next action (return / continue)

  - step: 6
    name: Action Execution
    description: Navigate back or resume search

deep_process:

  perception_pipeline:
    input: /camera/image_raw
    processing:
      - Convert ROS image to OpenCV format
      - Run YOLOv8 inference
      - Filter detections by target label
    output: Target detection flag + coordinates

  exploration_logic:
    method: Frontier-based exploration
    output: Next search goal

  decision_logic:
    input: Detection result + user command
    output:
      - Continue search
      - Return to start

  return_navigation:
    method: Save initial pose and navigate back using Nav2

ros2_commands:

  start_camera:
    command: |
      ros2 launch <camera_package> camera_launch.py

  start_slam:
    command: |
      ros2 launch slam_toolbox online_async_launch.py

  start_navigation:
    command: |
      ros2 launch nav2_bringup navigation_launch.py use_sim_time:=true

  perception_node:
    description: YOLOv8 detection node
    command: |
      ros2 run <your_package> object_detection_node.py

  exploration_node:
    command: |
      ros2 run <your_package> exploration_node.py

  voice_node:
    command: |
      ros2 run <your_package> voice_control_node.py

  tts_node:
    command: |
      ros2 run <your_package> tts_node.py

topics_used:
  - /camera/image_raw
  - /detected_objects
  - /scan
  - /map
  - /cmd_vel
  - /goal_pose
  - /voice_command
  - /robot_status

file_structure:
  src:
    - exploration_node.py
    - object_detection_node.py
    - voice_control_node.py
    - tts_node.py
    - decision_node.py

  models:
    - yolov8n.pt

  launch:
    - full_system_launch.py

challenges_faced:
  - Real-time YOLO inference latency on CPU
  - False positives in object detection
  - Synchronization between detection and navigation
  - Voice command delays and noise issues
  - Managing state transitions (search → detect → decide)

future_improvements:
  - GPU acceleration for YOLO
  - Multi-object priority search
  - 3D object localization
  - Integration with LLM for smarter dialogue

author:
  name: Rohan Nanekarents using \textbf{PETG 3D printing} and integrated BLDC propulsion systems.
\end{itemize}

\end{document}
