The default settings implemented an object tracking algorithm with camera position control, designed to focus the UAV on the target, based on deep machine learning and neural networks.
The program allows us to detect an object and center it so that it is constantly in the center of our camera at a distance that allows us to see the target in full. The software product was created in the Python programming language using UnrealEngine as a drone flight simulator.
The AirSim library for Python was used to create the drone simulation, and Yolo8 and OpenCV were used for deep learning.

The Dependencies section contains all the utilities used:
Dependencies:
- pip=24.0 is a package manager for Python.
- python=3.9=h7a1cb2a_0 is a widely used 
high-level, general-purpose, interpreted, dynamic 
- Unreal Engine 4.27 is a game engine developed and maintained by Epic Games. 
developed and maintained by Epic Games.
- AirSim is a simulator for drones, cars, and more, 
built on the Unreal Engine
- VS Code is a lightweight but powerful source code editor that runs on your desktop 
code editor that runs on your desktop and is available for 
Windows, macOS, and Linux.

The "pip" section contains a list of Python packages that are installed 
through pip, which is the standard package manager for Python. Their list with 
versions is given below: 
Pip:
- msgpack-rpc-python - cross-language RPC library for 
client, server, and cluster applications
- numpy>=1.25.0,<1.26 - a fundamental package for 
scientific computing in Python. It is a Python library that provides 
a multidimensional array object, various derived objects (such as 
arrays and matrices with masks), as well as a set of procedures for quick 
operations on arrays, including mathematical, logical 
manipulation of shapes, sorting, selection, input/output, 
discrete Fourier transforms, basic linear algebra, basic 
statistical operations, random modeling, and much more.
- opencv-python>=4.5.1,<4.7
- airsim==1.8.1 - an open source simulator based on the 
based Unreal Engine for autonomous cars from Microsoft AI & 
Research
- ultralytics==8.2.32 - a library for training NMs, 
includes Yolov8 in its arsenal
- PIL ==10.3.0 - utility for creating and storing 
images

