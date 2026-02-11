
## Installation

```bash
conda create -n ros_webcam python=3.11
conda install -c conda-forge -c robostack-noetic ros-noetic-desktop
conda install -c conda-forge -c robostack-noetic ros-noetic-rosbridge-server

pip install -r requirements.txt
```

## Usage

Run each command in a separate terminal (with `conda activate ros_webcam`):

1. Start the ROS bridge server:
```bash
roslaunch rosbridge_server rosbridge_websocket.launch
```

2. Start the Flask server:
```bash
python app.py
```

3. Open `https://<server-ip>:5000` on your iPhone to begin streaming.

4. (Optional) View the stream on your machine:
```bash
python camera_listener.py
```