"""
airsim 1.8.1

we need two airsim MultirotorClient objects because the comm lib we use (rpclib) is not thread safe
so we poll images in a thread using one airsim MultirotorClient object
and use another airsim MultirotorClient for querying state commands

took from https://github.com/microsoft/AirSim-Drone-Racing-Lab/blob/master/baselines/baseline_racer.py
"""

import logging
from collections import OrderedDict
from threading import Thread
from time import sleep, time
from typing import Any, Dict, List, Tuple

import airsim
import cv2
import numpy as np
from airsim import YawMode
from airsim.types import CameraInfo, ImageResponse
from PIL import Image

logger = logging.getLogger(__name__)
logger_format = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=logger_format,
    handlers=[logging.FileHandler("airsim.log", mode="w"), logging.StreamHandler()],
)


def radians_to_degrees(radians):
    return radians * 180.0 / np.pi


def degrees_to_radians(degrees):
    return degrees * np.pi / 180.0


class OpticalFlow:
    def __init__(
        self,
        winSize: Tuple[int] = (31, 31),
        maxLevel: int = 3,
        termcrit: Tuple[Any] = None,
    ):
        self.winSize = winSize
        self.maxLevel = maxLevel
        self.termcrit = (
            termcrit
            if termcrit
            else (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03)
        )

        self.prev_img = None
        self.prev_pts = []

    def initialize(self, image: np.ndarray, x: float, y: float) -> None:
        self.prev_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.prev_pts = [np.array((x, y)).reshape((1, 2)).astype(np.float32)]

    def track(self, image: np.ndarray) -> None:
        next_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_img,
            next_img,
            self.prev_pts[-1],
            None,
            winSize=self.winSize,
            maxLevel=self.maxLevel,
            criteria=self.termcrit,
        )
        next_pts = next_pts[status.flatten() == 1]

        self.prev_img = next_img
        self.prev_pts = [next_pts]

    def target(self) -> Tuple[float, float]:
        if len(self.prev_pts[0]) == 0:
            return None
        else:
            return self.prev_pts[-1][0]

    def reset(self) -> None:
        self.prev_img = None
        self.prev_pts = []

    def initialized(self) -> bool:
        return self.prev_img is not None and len(self.prev_pts) > 0

    def found(self) -> bool:
        return len(self.prev_pts[0]) > 0


class Experiment:
    def __init__(
        self,
        drone_name: str = "Drone",
        target_regex: str = ".*Sphere.*",
        target_object_id: int = 20,
        target_color: tuple = (146, 52, 70),
    ):
        self.drone_name = drone_name
        self.target_regex = target_regex
        self.target_object_id = target_object_id
        self.target_color = target_color

        #self.initial_drone_pose = airsim.Pose(
            #airsim.Vector3r(0, 0, 0),
            #airsim.to_quaternion(0, 0, degrees_to_radians(180)),
        #)

        self.client_airsim = airsim.MultirotorClient()
        self.client_airsim.confirmConnection()

        self.client_images = airsim.MultirotorClient()
        self.client_images.confirmConnection()

        self.client_odometry = airsim.MultirotorClient()
        self.client_odometry.confirmConnection()

        self.image_callback_thread = Thread(
            target=self.repeat_timer_image_callback,
            args=(self.image_callback, 0.02),
            daemon=True,
        )
        self.odometry_callback_thread = Thread(
            target=self.repeat_timer_odometry_callback,
            args=(self.odometry_callback, 0.02),
            daemon=True,
        )
        self.tracking_callback_thread = Thread(
            target=self.repeat_timer_tracking_callback,
            args=(self.tracking_callback, 0.02),
            daemon=True,
        )
        self.moving_callback_thread = Thread(
            target=self.repeat_timer_moving_callback,
            args=(self.moving_callback, 0.02),
            daemon=True,
        )
        self.visualize_callback_thread = Thread(
            target=self.repeat_timer_visualize_callback,
            args=(self.visualize_callback, 0.02),
            daemon=True,
        )

        self.is_image_thread_active = False
        self.is_odometry_thread_active = False
        self.is_tracking_thread_active = False
        self.is_moving_thread_active = False
        self.is_visualize_thread_active = False

        self.trackers = {
            "BOOSTING": cv2.legacy.TrackerBoosting_create(),
            "MIL": cv2.TrackerMIL_create(),
            "KCF": cv2.TrackerKCF_create(),
            "CSRT": cv2.TrackerCSRT_create(),
            "MedianFlow": cv2.legacy.TrackerMedianFlow_create(),
            "TLD": cv2.legacy.TrackerTLD_create(),
            "MOSSE": cv2.legacy.TrackerMOSSE_create(),
        }

        self.tracker = OpticalFlow()
        #self.tracker = self.trackers["BOOSTING"]
        #self.tracker = self.trackers["KCF"]
        #self.tracker = self.trackers["CSRT"]
        #self.tracker = self.trackers["MedianFlow"]
        #self.tracker = self.trackers["TLD"]
        #self.tracker = self.trackers["MOSSE"]
        self.tracker_status = "Not initialized"

        # Set the camera requests
        self.airsim_camera_requests = OrderedDict(
            rgb=airsim.ImageRequest(
                camera_name="rgb",
                image_type=airsim.ImageType.Scene,
                pixels_as_float=False,
                compress=False,
            ),
            seg=airsim.ImageRequest(
                camera_name="seg",
                image_type=airsim.ImageType.Segmentation,
                pixels_as_float=False,
                compress=False,
            ),
        )

        self.img_rgb = None
        self.img_seg = None

        self.target_x = None
        self.target_y = None

        self.tracking_x = None
        self.tracking_y = None

        self.drone_state = None

        self.img_counter = 0

        # Set the segmentation object color
        self.client_airsim.simSetSegmentationObjectID(
            mesh_name=target_regex, object_id=target_object_id, is_name_regex=True
        )

        self._print_camera_info()

    def _print_camera_info(self):
        # Get the camera information
        resp = self.client_images.simGetImages(requests=[self.airsim_camera_requests["rgb"]])[
            0
        ]
        self.rgb_height, self.rgb_width = resp.height, resp.width

        logger.info(f"Camera RGB resolution: {self.rgb_width}x{self.rgb_height}")

    def initialize_drone(self):
        pass
        #Disarm the drone
        #self.client_airsim.armDisarm(False, vehicle_name=self.drone_name)
        #logger.info(f"Drone disarmed")

        # Disable API control
        #self.client_airsim.enableApiControl(False, vehicle_name=self.drone_name)
        #logger.info(f"API control disabled")

        # Enable API control
        #self.client_airsim.enableApiControl(True, vehicle_name=self.drone_name)
        #logger.info(f"API control enabled")

        # Set the initial drone pose
        #self.client_airsim.simSetVehiclePose(
            #self.initial_drone_pose, ignore_collision=False, vehicle_name=self.drone_name)
        #logger.info(f"Drone pose set to: {self.initial_drone_pose}")

        # Arm the drone
        #self.client_airsim.armDisarm(True, vehicle_name=self.drone_name)
        #logger.info(f"Drone armed")

        # Take off
        self.client_airsim.takeoffAsync(vehicle_name=self.drone_name).join()
        logger.info(f"Drone took off")



    def read_images(self, include_segmenation: bool = True):
        requests = OrderedDict()

        if include_segmenation == True:
            requests["seg"] = self.airsim_camera_requests["seg"]

        requests["rgb"] = self.airsim_camera_requests["rgb"]
        logger.debug(f"Requesting {len(requests)} images")

        responses: Dict[str, ImageResponse] = OrderedDict(
            zip(
                list(requests.keys()),
                self.client_images.simGetImages(requests=list(requests.values())),
            )
        )
        logger.debug(f"Got {len(responses)} images")

        response = responses["rgb"]
        img1d = np.frombuffer(
            response.image_data_uint8, dtype=np.uint8
        )  # get numpy array
        img_rgb = img1d.reshape(
            response.height, response.width, 3
        )  # reshape array to 4 channel image array H X W X 3

        if include_segmenation:
            response = responses["seg"]
            img1d = np.frombuffer(
                response.image_data_uint8, dtype=np.uint8
            )  # get numpy array
            img_seg = img1d.reshape(
                response.height, response.width, 3
            )  # reshape array to 4 channel image array H X W X 3
            img_seg = cv2.cvtColor(img_seg, cv2.COLOR_RGB2BGR)

            # Resize the segmentation image to the same size as the RGB image
            img_seg = cv2.resize(img_seg, (self.rgb_width, self.rgb_height), interpolation=cv2.INTER_NEAREST)

            target_pts = np.where(np.all(img_seg == self.target_color, axis=-1))

            if len(target_pts[0]) == 0 or len(target_pts[1]) == 0:
                x = None
                y = None
            else:
                x = target_pts[0].mean()
                y = target_pts[1].mean()
        else:
            img_seg = None
            x = None
            y = None

        return dict(
            img_rgb=img_rgb,
            img_seg=img_seg,
            x=x,
            y=y,
        )

    def image_callback(self):

        response = self.read_images(include_segmenation=True)

        self.img_rgb = response["img_rgb"]
        logger.info(f"Received RGB image: {self.img_rgb.shape}")
        self.img_seg = response["img_seg"]
        logger.debug(f"Received segmentation image: {self.img_seg.shape}")

        # Save image from camera
        img = self.client_images.simGetImages(requests=[self.airsim_camera_requests["rgb"]])[0] 
        if self.img_rgb is None:
            print("Failed to capture image")
        else:
            img1d = np.frombuffer(img.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape((img.height, img.width, 3))
            image = Image.fromarray(img_rgb)
            image.save(f"Dataset/{self.img_counter}.png")
            self.img_counter += 1
            if self.img_counter == 4999:
                logger.info("STOP STOP STOP STOP STOP")
        logger.info(f"Saved RGB image {self.img_counter}")
        


        self.target_x = response["y"]
        self.target_y = response["x"]
        logger.debug(f"Target at: {self.target_x}, {self.target_y}")



    def draw_multiline_text(
        self, img, text, color, font, font_scale, thickness, line_type
    ):
        lines = text.split("\n")
        line_width, line_height = cv2.getTextSize(lines[0], font, font_scale, thickness)[0]
        x_offset = round(line_width * 0.1)  # add some padding
        y_offset = round(line_height * 0.4) # add some padding
        for i, line in enumerate(lines, 1):
            img = cv2.putText(
                img,
                line,
                (x_offset, y_offset + i * (y_offset + line_height)),
                font,
                font_scale,
                color,
                thickness,
                line_type,
            )
        return img
    
    def visualize_callback(self):
        img_rgb = self.img_rgb.copy()
        img_seg = self.img_seg.copy()

        text_color = (0, 0, 0)
        target_color = (0, 255, 0)
        tracking_color = (255, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        font_scale = 0.75
        line_type = cv2.LINE_8

        info_schema = (
            "Image: {width}x{height}\n"
            "Target at: ({target_x}, {target_y})\n"
            "Tracking at: ({tracking_x}, {tracking_y})\n"
            "Yaw rate: {yaw_rate}\n"
            "Tracking: {tracking}\n"
        )

        width, height = img_rgb.shape[1], img_rgb.shape[0]
        target_x = round(self.target_x) if self.target_x is not None else None
        target_y = round(self.target_y) if self.target_y is not None else None
        tracking_x = round(self.tracking_x) if self.tracking_x is not None else None
        tracking_y = round(self.tracking_y) if self.tracking_y is not None else None
        yaw_rate = "%.3f" % self.drone_state.kinematics_estimated.angular_velocity.z_val
        tracking = self.tracker_status 

        info = info_schema.format(
            width=width,
            height=height,
            target_x=target_x,
            target_y=target_y,
            tracking_x=tracking_x,
            tracking_y=tracking_y,
            yaw_rate=yaw_rate,
            tracking=tracking,
        )
        
        # Combine images
        img = np.concatenate([img_rgb, img_seg], axis=1)

        # draw multiline text using opencv
        img = self.draw_multiline_text(img, info, text_color, font, font_scale, thickness, line_type)

        has_target = self.target_x is not None and self.target_y is not None
        has_tracking = self.tracking_x is not None and self.tracking_y is not None
        
        if has_target:
            img = cv2.circle(
                img, (int(self.target_x), int(self.target_y)), 5, target_color, -1
            )
            img = cv2.putText(
                img,
                f"target",
                (int(self.target_x), int(self.target_y)),
                font,
                font_scale,
                target_color,
                thickness,
                line_type,
            )
            
        if has_tracking:
            img = cv2.circle(
                img, (int(self.target_x), int(self.target_y)), 5, tracking_color, -1
            )
            text_size = cv2.getTextSize("tracking", font, font_scale, thickness)[0]
            img = cv2.putText(
                img,
                f"tracking",
                (int(self.target_x) - text_size[0], int(self.target_y) + text_size[1]),
                font,
                font_scale,
                tracking_color,
                thickness,
                line_type,
            )

        cv2.imshow(self.level_name, img)
        cv2.waitKey(1)

    def odometry_callback(self):
        self.drone_state = self.client_odometry.getMultirotorState()

    def takeoff_async(self):
        self.client_airsim.takeoffAsync(self.drone_name).join()

    def move_drone(
        self, roll: float, pitch: float, yaw: float, throttle: float, duration: float
    ):
        self.client_airsim.moveByManualAsync(
            vx_max=1e9,
            vy_max=1e9,
            z_min=-1e9,
            duration=duration,  # v(x,y,z) represents the velocity in the global coordinate system
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=YawMode(is_rate=True, yaw_or_rate=10.0 * (1 if yaw > 0 else -1)),
        )
        self.client_airsim.moveByRC(
            rcdata=airsim.RCData(
                roll=roll,
                throttle=throttle,
                yaw=yaw,
                pitch=pitch,
                is_initialized=False,
                is_valid=True,
            )
        )
        sleep(duration)

    def wait_for_images(self):
        while self.img_rgb is None:
            sleep(0.1)
            logger.debug("Waiting for images")

        # Sleep for a while
        sleep(2)

    def repeat_timer_image_callback(self, callback, period_sec):
        self.is_image_thread_active = True
        while self.is_image_thread_active:
            callback()
            sleep(period_sec)

    def repeat_timer_odometry_callback(self, callback, period_sec):
        self.is_odometry_thread_active = True
        while self.is_odometry_thread_active:
            callback()
            sleep(period_sec)

    def repeat_timer_tracking_callback(self, callback, period_sec):
        self.is_tracking_thread_active = True
        while self.is_tracking_thread_active:
            callback()
            sleep(period_sec)

    def repeat_timer_moving_callback(self, callback, period_sec):
        self.is_moving_thread_active = True
        while self.is_moving_thread_active:
            callback()
            sleep(period_sec)
    
    def repeat_timer_visualize_callback(self, callback, period_sec):
        self.is_visualize_thread_active = True
        while self.is_visualize_thread_active:
            callback()
            sleep(period_sec)

    def moving_callback(self):
        if self.tracking_x is not None and self.tracking_y is not None:
            logger.debug(f"Moving callback")

            # just move yaw using x
            roll = 0.0
            pitch = 0.0
            throttle = 0.0

            #dx = 0
            #yaw = 0
            #duration = 0
            dx = self.tracking_x - self.rgb_width / 2  # -width/2 to width/2
            yaw = dx / self.rgb_width * 2  # -1 to 1
            duration = 0.010  # 10 ms

            self.move_drone(roll, pitch, yaw, throttle, duration)

    def tracking_callback(self):
        logger.debug(f"Tracking callback")

        if not self.tracker.initialized():
            logger.debug(f"Initializing tracker")

            if self.target_x is None or self.target_y is None:
                logger.debug(f"Target not set, can not initialize tracker")
                self.tracker_status = "Target not set"
                return

            logger.debug(f"Using frame {self.img_rgb.shape} to initialize tracker")
            self.tracker.initialize(
                self.img_rgb,
                self.target_x,
                self.target_y,
            )
            logger.debug(f"Tracker initialized")
            self.tracker_status = "Initialized"

        else:
            self.tracker.track(self.img_rgb)
            logger.debug(f"Tracker tracked")

            if self.tracker.found():
                logger.debug(f"Target found at: {self.tracker.target()}")
                self.tracker_status = "Found"
                target = self.tracker.target()
                self.tracking_x = target[0]
                self.tracking_y = target[1]

            else:
                logger.debug(f"Target not found")
                self.tracker_status = "Not found"
                self.tracker.reset()
                self.tracking_x = None
                self.tracking_y = None
                logger.debug(f"Tracker reset")
    
    def load_level(self, level_name, sleep_sec=2.0):
        self.level_name = level_name
        self.client_airsim.simLoadLevel(self.level_name)
        self.client_airsim.confirmConnection()  # failsafe
        sleep(sleep_sec)  # let the environment load completely
        logger.info(f"Level {self.level_name} loaded")

    def start_image_callback_thread(self):
        if not self.is_image_thread_active:
            self.is_image_thread_active = True
            self.image_callback_thread.start()
            logger.info("Started image callback thread")

    def stop_image_callback_thread(self):
        if self.is_image_thread_active:
            self.is_image_thread_active = False
            self.image_callback_thread.join()
            logger.info("Stopped image callback thread.")

    def start_odometry_callback_thread(self):
        if not self.is_odometry_thread_active:
            self.is_odometry_thread_active = True
            self.odometry_callback_thread.start()
            logger.info("Started odometry callback thread")

    def stop_odometry_callback_thread(self):
        if self.is_odometry_thread_active:
            self.is_odometry_thread_active = False
            self.odometry_callback_thread.join()
            logger.info("Stopped odometry callback thread.")

    def start_tracking_callback_thread(self):
        if not self.is_tracking_thread_active:
            self.is_tracking_thread_active = True
            self.tracking_callback_thread.start()
            logger.info("Started tracking callback thread")

    def stop_tracking_callback_thread(self):
        if self.is_tracking_thread_active:
            self.is_tracking_thread_active = False
            self.tracking_callback_thread.join()
            logger.info("Stopped tracking callback thread.")

    def start_moving_callback_thread(self):
        if not self.is_moving_thread_active:
            self.is_moving_thread_active = True
            self.moving_callback_thread.start()
            logger.info("Started moving callback thread")

    def stop_moving_callback_thread(self):
        if self.is_moving_thread_active:
            self.is_moving_thread_active = False
            self.moving_callback_thread.join()
            logger.info("Stopped moving callback thread.")
    
    def start_visualize_callback_thread(self):
        if not self.is_visualize_thread_active:
            self.is_visualize_thread_active = True
            self.visualize_callback_thread.start()
            logger.info("Started visualize callback thread")
            
    def stop_visualize_callback_thread(self):
        if self.is_visualize_thread_active:
            self.is_visualize_thread_active = False
            self.visualize_callback_thread.join()
            logger.info("Stopped visualize callback thread.")
    
    def run(self):
        self.load_level("DefaultMap")
        self.initialize_drone()
        self.start_image_callback_thread()
        self.start_odometry_callback_thread()

        self.wait_for_images()
        self.start_visualize_callback_thread()
        self.start_tracking_callback_thread()
        self.start_moving_callback_thread()

        sleep(30000)

        self.stop_image_callback_thread()
        self.stop_odometry_callback_thread()
        self.stop_tracking_callback_thread()
        self.stop_moving_callback_thread()


def main():
    experiment = Experiment()
    experiment.run()


if __name__ == "__main__":
    main()
