"""
Offshore wind farm PX4 demo.

Run this while the OffShoreWindFarm Unreal map is playing. The script loads a
PX4 scene, spawns Drone1 at OFFSHORE_ROUTE[0], and shows FPV with a
chase-camera picture-in-picture. For now there is no route flying; E/R teleport
between the route points.

Controls in the OpenCV preview:
  e  teleport to next route point
  r  teleport to previous route point
  q/esc  stop the route and exit
"""

import argparse
import asyncio
import math
import queue
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, List, Optional, Sequence, Tuple

import commentjson

from projectairsim import Drone, ProjectAirSimClient, World
from projectairsim.types import Pose, Quaternion, Vector3
from projectairsim.utils import projectairsim_log, unpack_image


SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLE_SCRIPTS_DIR = SCRIPT_DIR.parent / "example_user_scripts"
if str(EXAMPLE_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_SCRIPTS_DIR))

from px4_astar_autopilot import (  # noqa: E402
    distance_between,
    format_vector3,
    get_pose_position_ned,
    get_pose_yaw_ned,
    heading_deg_360,
    wait_for_px4_ready,
)


OFFSHORE_ROUTE = [
    ("Origin", [0.0, 148.0, -2.0], 90.0),
    ("Waypoint 1", [1.28, -14.66, -115.16], 127.6),
    ("Waypoint 2", [893.69, 7.89, -112.77], 324.3),
    ("Waypoint 3", [899.44, -892.28, -12.28], 279.5),
    ("Waypoint 4", [-3.35, -907.1, -114.81], 55.2),
    ("Waypoint 5", [-894.61, -901.37, 114.0], 159.9),
    ("Waypoint 6", [-899.67, -5.87, -7.82], 111.0),
    ("Return Origin", [0.0, 148.0, -2.0], 90.0),
]


@dataclass(frozen=True)
class RoutePoint:
    label: str
    position: List[float]
    yaw_deg: float


def route_to_ned(position: Sequence[float]) -> List[float]:
    return [float(position[0]), float(position[1]), float(position[2])]


def ned_to_route(position_ned: Sequence[float]) -> List[float]:
    return [float(position_ned[0]), float(position_ned[1]), float(position_ned[2])]


class DemoState:
    def __init__(self, route: Sequence[RoutePoint]):
        self.route = list(route)
        self.current_index = 0
        self.target_index = 1
        self.position = list(self.route[0].position)
        self.heading_deg = self.route[0].yaw_deg
        self.teleport_requests = queue.SimpleQueue()
        self.stop_requested = False
        self.lock = Lock()

    def snapshot(self):
        with self.lock:
            return {
                "current_index": self.current_index,
                "target_index": self.target_index,
                "position": list(self.position),
                "heading_deg": self.heading_deg,
                "stop_requested": self.stop_requested,
            }

    def update_pose(self, position: Sequence[float], heading_deg: float):
        with self.lock:
            self.position = [float(position[0]), float(position[1]), float(position[2])]
            self.heading_deg = float(heading_deg) % 360.0

    def mark_reached(self, index: int):
        with self.lock:
            self.current_index = index
            self.target_index = min(index + 1, len(self.route) - 1)

    def queue_teleport(self, delta: int):
        self.teleport_requests.put(delta)

    def request_stop(self):
        with self.lock:
            self.stop_requested = True


class OffshorePreview:
    def __init__(
        self,
        state: DemoState,
        window_name: str,
        width: int,
        height: int,
        pip_scale: float,
        max_fps: float,
    ):
        self.state = state
        self.window_name = window_name
        self.width = width
        self.height = height
        self.pip_scale = max(0.1, min(0.5, pip_scale))
        self.max_fps = max(1.0, max_fps)
        self.fpv_images = queue.SimpleQueue()
        self.chase_images = queue.SimpleQueue()
        self.running = False
        self.thread = None
        self.error = None

        xs = [point.position[0] for point in self.state.route]
        ys = [point.position[1] for point in self.state.route]
        self.map_min_x = min(xs)
        self.map_max_x = max(xs)
        self.map_min_y = min(ys)
        self.map_max_y = max(ys)

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = Thread(target=self.display_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None

    def receive_fpv(self, _, image):
        self._push_latest(self.fpv_images, image)

    def receive_chase(self, _, image):
        self._push_latest(self.chase_images, image)

    def _push_latest(self, image_queue, image):
        if not self.running or image is None:
            return
        while not image_queue.empty() and image_queue.qsize() > 2:
            image_queue.get()
        image_queue.put(image)

    def _pop_latest_frame(self, image_queue):
        image = None
        while not image_queue.empty():
            image = image_queue.get()
        if image is None:
            return None
        frame = unpack_image(image)
        if frame is None:
            return None
        return frame.copy()

    def display_loop(self):
        import cv2

        created = False
        frame_interval_sec = 1.0 / self.max_fps
        next_frame_at = time.monotonic()
        try:
            while self.running:
                now = time.monotonic()
                if now < next_frame_at:
                    key = cv2.waitKey(max(1, int((next_frame_at - now) * 1000.0)))
                    self.handle_key(key)
                    continue

                fpv = self._pop_latest_frame(self.fpv_images)
                if fpv is None:
                    key = cv2.waitKey(1)
                    self.handle_key(key)
                    continue

                frame = fpv
                frame = self.ensure_bgr(cv2, frame)
                frame = cv2.resize(frame, (self.width, self.height))
                chase = self._pop_latest_frame(self.chase_images)
                if chase is not None:
                    chase = self.ensure_bgr(cv2, chase)
                    self.draw_chase_pip(cv2, frame, chase)

                self.draw_route_overlay(cv2, frame)
                self.draw_status(cv2, frame)

                if not created:
                    cv2.namedWindow(
                        self.window_name,
                        flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE,
                    )
                    created = True

                cv2.imshow(self.window_name, frame)
                self.handle_key(cv2.waitKey(1))
                next_frame_at = time.monotonic() + frame_interval_sec
        except Exception as exc:
            self.error = exc
            self.state.request_stop()
        finally:
            if created:
                cv2.destroyWindow(self.window_name)

    def handle_key(self, key: int):
        if key < 0:
            return
        key &= 0xFF
        if key == ord("e"):
            self.state.queue_teleport(1)
        elif key == ord("r"):
            self.state.queue_teleport(-1)
        elif key in (ord("q"), 27):
            self.state.request_stop()
            self.running = False

    @staticmethod
    def ensure_bgr(cv2, frame):
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 1:
            return cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def draw_chase_pip(self, cv2, frame, chase):
        height, width = frame.shape[:2]
        pip_w = int(width * self.pip_scale)
        pip_h = int(pip_w * 9 / 16)
        pip_h = min(pip_h, int(height * 0.35))
        pip = cv2.resize(chase, (pip_w, pip_h))
        x0 = width - pip_w - 16
        y0 = 16
        frame[y0 : y0 + pip_h, x0 : x0 + pip_w] = pip
        cv2.rectangle(frame, (x0, y0), (x0 + pip_w, y0 + pip_h), (255, 255, 255), 2)
        self.draw_text(cv2, frame, "Chase", (x0 + 8, y0 + 22), scale=0.55)

    def draw_route_overlay(self, cv2, frame):
        origin_x = 18
        origin_y = 56
        map_w = 300
        map_h = 300
        pad_m = 80.0
        span_x = max(1.0, self.map_max_x - self.map_min_x + pad_m * 2)
        span_y = max(1.0, self.map_max_y - self.map_min_y + pad_m * 2)

        def to_px(position: Sequence[float]) -> Tuple[int, int]:
            x_norm = (position[0] - self.map_min_x + pad_m) / span_x
            y_norm = (position[1] - self.map_min_y + pad_m) / span_y
            px = origin_x + int(x_norm * map_w)
            py = origin_y + map_h - int(y_norm * map_h)
            return px, py

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (origin_x - 10, origin_y - 30),
            (origin_x + map_w + 10, origin_y + map_h + 10),
            (18, 18, 18),
            -1,
        )
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0.0, frame)
        cv2.rectangle(
            frame,
            (origin_x - 10, origin_y - 30),
            (origin_x + map_w + 10, origin_y + map_h + 10),
            (235, 235, 235),
            1,
        )
        self.draw_text(cv2, frame, "Route", (origin_x, origin_y - 8), scale=0.55)

        route_pixels = [to_px(point.position) for point in self.state.route]
        for start, end in zip(route_pixels, route_pixels[1:]):
            cv2.line(frame, start, end, (255, 0, 0), 2, cv2.LINE_AA)

        snapshot = self.state.snapshot()
        for index, point in enumerate(self.state.route):
            color = (0, 0, 255)
            radius = 6
            if index == snapshot["current_index"]:
                color = (0, 210, 255)
                radius = 8
            elif index == snapshot["target_index"]:
                color = (0, 255, 255)
                radius = 8
            cv2.circle(frame, to_px(point.position), radius, color, -1, cv2.LINE_AA)

        pos_px = to_px(snapshot["position"])
        cv2.drawMarker(
            frame,
            pos_px,
            (80, 255, 80),
            markerType=cv2.MARKER_TRIANGLE_UP,
            markerSize=18,
            thickness=2,
            line_type=cv2.LINE_AA,
        )

    def draw_status(self, cv2, frame):
        snapshot = self.state.snapshot()
        target = self.state.route[snapshot["target_index"]]
        position = snapshot["position"]
        distance = distance_between(position, target.position)
        lines = [
            f"Target: {target.label}  {distance:.1f} m",
            f"NED: x={position[0]:.1f} y={position[1]:.1f} z={position[2]:.1f}",
            f"Heading: {snapshot['heading_deg']:.1f} deg  Mode: teleport only",
            "e next waypoint | r previous waypoint | q/esc quit",
        ]
        y = frame.shape[0] - 92
        for line in lines:
            self.draw_text(cv2, frame, line, (18, y), scale=0.55)
            y += 22

    @staticmethod
    def draw_text(cv2, frame, text: str, origin: Tuple[int, int], scale: float = 0.6):
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def make_pose_ned_yaw(position_ned: Sequence[float], yaw_deg: float) -> Pose:
    half_yaw = math.radians(yaw_deg) / 2.0
    return Pose(
        {
            "translation": Vector3(
                {
                    "x": float(position_ned[0]),
                    "y": float(position_ned[1]),
                    "z": float(position_ned[2]),
                }
            ),
            "rotation": Quaternion(
                {
                    "w": math.cos(half_yaw),
                    "x": 0.0,
                    "y": 0.0,
                    "z": math.sin(half_yaw),
                }
            ),
            "frame_id": "DEFAULT_ID",
        }
    )


def format_scene_origin_xyz(position_ned: Sequence[float]) -> str:
    return " ".join(f"{component:g}" for component in position_ned)


def format_scene_origin_rpy(yaw_deg: float) -> str:
    return f"0 0 {yaw_deg:g}"


def resolve_config_path(config_name: str, sim_config_path: str) -> Path:
    config_path = Path(config_name)
    if config_path.is_absolute():
        return config_path

    config_dir = Path(sim_config_path)
    candidates = [
        config_dir / config_path,
        SCRIPT_DIR / config_dir / config_path,
        EXAMPLE_SCRIPTS_DIR / config_dir / config_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_jsonc(path: Path):
    return commentjson.loads(path.read_text(encoding="utf-8"))


def ensure_scene_camera_capture(
    sensor: Dict,
    width: int,
    height: int,
    fov_degrees: float,
    capture_interval_sec: float,
):
    sensor["enabled"] = True
    sensor["capture-interval"] = capture_interval_sec
    capture_settings = sensor.setdefault("capture-settings", [])
    scene_capture = next(
        (capture for capture in capture_settings if capture.get("image-type") == 0),
        None,
    )
    if scene_capture is None:
        scene_capture = {"image-type": 0}
        capture_settings.append(scene_capture)

    scene_capture.update(
        {
            "width": width,
            "height": height,
            "fov-degrees": fov_degrees,
            "capture-enabled": True,
            "streaming-enabled": True,
            "pixels-as-float": False,
            "compress": False,
            "target-gamma": 2.5,
        }
    )


def default_front_camera_sensor(
    width: int,
    height: int,
    fov_degrees: float,
    capture_interval_sec: float,
):
    return {
        "id": "FrontCamera",
        "type": "camera",
        "enabled": True,
        "parent-link": "Frame",
        "capture-interval": capture_interval_sec,
        "capture-settings": [
            {
                "image-type": 0,
                "width": width,
                "height": height,
                "fov-degrees": fov_degrees,
                "capture-enabled": True,
                "streaming-enabled": True,
                "pixels-as-float": False,
                "compress": False,
                "target-gamma": 2.5,
            }
        ],
        "origin": {
            "xyz": "0.5 0.0 0.0",
            "rpy-deg": "0 0 0",
        },
    }


def ensure_camera(
    robot_config: Dict,
    camera_id: str,
    width: int,
    height: int,
    fov_degrees: float,
    capture_interval_sec: float,
):
    sensors = robot_config.setdefault("sensors", [])
    sensor = next((candidate for candidate in sensors if candidate.get("id") == camera_id), None)
    if sensor is None:
        if camera_id != "FrontCamera":
            raise RuntimeError(f"Camera '{camera_id}' is not present in the robot config")
        sensors.append(
            default_front_camera_sensor(width, height, fov_degrees, capture_interval_sec)
        )
        projectairsim_log().info("Runtime config added FrontCamera for FPV")
        return
    if sensor.get("type") != "camera":
        raise RuntimeError(f"Sensor '{camera_id}' exists but is not a camera")
    ensure_scene_camera_capture(sensor, width, height, fov_degrees, capture_interval_sec)


def make_runtime_scene_config(args, route: Sequence[RoutePoint]):
    scene_path = resolve_config_path(args.scene, args.sim_config_path)
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene config not found: {scene_path}")

    scene_config = load_jsonc(scene_path)
    actors = scene_config.get("actors", [])
    target_actor = next(
        (
            actor
            for actor in actors
            if actor.get("type") == "robot" and actor.get("name") == args.drone_name
        ),
        None,
    )
    if target_actor is None or not target_actor.get("robot-config"):
        raise RuntimeError(
            f"Could not find robot actor '{args.drone_name}' with a robot-config"
        )

    start = route[0]
    origin = target_actor.setdefault("origin", {})
    origin["xyz"] = format_scene_origin_xyz(route_to_ned(start.position))
    origin["rpy-deg"] = format_scene_origin_rpy(start.yaw_deg)

    temp_dir = tempfile.TemporaryDirectory(prefix="offshore_demo_")
    temp_config_dir = Path(temp_dir.name)

    try:
        for actor_index, actor in enumerate(actors):
            if actor.get("type") != "robot" or not actor.get("robot-config"):
                continue

            robot_config_path = resolve_config_path(
                actor["robot-config"],
                str(scene_path.parent),
            )
            robot_config = load_jsonc(robot_config_path)
            if actor is target_actor:
                ensure_camera(
                    robot_config,
                    args.fpv_camera,
                    args.camera_width,
                    args.camera_height,
                    args.camera_fov_degrees,
                    args.camera_capture_interval_sec,
                )
                ensure_camera(
                    robot_config,
                    args.chase_camera,
                    args.camera_width,
                    args.camera_height,
                    args.camera_fov_degrees,
                    args.camera_capture_interval_sec,
                )

            suffix = robot_config_path.suffix or ".jsonc"
            output_name = f"{robot_config_path.stem}_{actor_index}_offshore{suffix}"
            (temp_config_dir / output_name).write_text(
                commentjson.dumps(robot_config, indent=2) + "\n",
                encoding="utf-8",
            )
            actor["robot-config"] = output_name

        for env_actor in scene_config.get("environment-actors", []):
            if env_actor.get("type") != "env_actor" or not env_actor.get("env-actor-config"):
                continue
            env_config_path = resolve_config_path(
                env_actor["env-actor-config"],
                str(scene_path.parent),
            )
            shutil.copy2(env_config_path, temp_config_dir / env_config_path.name)
            env_actor["env-actor-config"] = env_config_path.name

        scene_name = scene_path.name
        (temp_config_dir / scene_name).write_text(
            commentjson.dumps(scene_config, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        temp_dir.cleanup()
        raise

    projectairsim_log().info(
        "Runtime offshore scene: %s with %s origin xyz=%s rpy-deg=%s",
        temp_config_dir / scene_name,
        args.drone_name,
        format_scene_origin_xyz(route_to_ned(start.position)),
        format_scene_origin_rpy(start.yaw_deg),
    )
    return temp_dir, scene_name, str(temp_config_dir)


def route_from_constants() -> List[RoutePoint]:
    return [
        RoutePoint(label, [float(pos[0]), float(pos[1]), float(pos[2])], float(yaw_deg))
        for label, pos, yaw_deg in OFFSHORE_ROUTE
    ]


def drain_teleport_request(state: DemoState) -> Optional[int]:
    delta_total = 0
    while not state.teleport_requests.empty():
        delta_total += state.teleport_requests.get()
    if delta_total == 0:
        return None

    snapshot = state.snapshot()
    route_len = len(state.route)
    return (snapshot["current_index"] + delta_total) % route_len


async def apply_teleport(drone: Drone, state: DemoState, index: int):
    point = state.route[index]
    drone.cancel_last_task()
    target_ned = route_to_ned(point.position)
    drone.set_pose(make_pose_ned_yaw(target_ned, point.yaw_deg), reset_kinematics=True)
    await asyncio.sleep(0.1)
    actual_ned = get_pose_position_ned(drone)
    actual_route = ned_to_route(actual_ned)
    state.update_pose(actual_route, point.yaw_deg)
    with state.lock:
        state.current_index = index
        state.target_index = min(index + 1, len(state.route) - 1)
    projectairsim_log().info(
        "Teleported to %s NED requested=%s actual=%s heading=%.1f deg",
        point.label,
        format_vector3(target_ned),
        format_vector3(actual_ned),
        point.yaw_deg % 360.0,
    )


async def run_teleport_viewer(drone: Drone, state: DemoState, args):
    await apply_teleport(drone, state, 0)
    projectairsim_log().info("Teleport-only mode active. Press e/r in the preview.")
    last_report_at = 0.0

    while not state.snapshot()["stop_requested"]:
        teleport_index = drain_teleport_request(state)
        if teleport_index is not None:
            await apply_teleport(drone, state, teleport_index)

        current_ned = get_pose_position_ned(drone)
        current = ned_to_route(current_ned)
        current_heading = heading_deg_360(get_pose_yaw_ned(drone))
        state.update_pose(current, current_heading)

        now = time.time()
        if now - last_report_at >= args.pose_report_interval_sec:
            projectairsim_log().info(
                "Teleport viewer NED %s heading %.1f deg",
                format_vector3(current),
                current_heading,
            )
            last_report_at = now

        await asyncio.sleep(0.05)


async def run_demo(args):
    route = route_from_constants()
    state = DemoState(route)
    temp_config_dir = None
    drone = None
    client = ProjectAirSimClient(
        address=args.server_ip,
        port_topics=args.topics_port,
        port_services=args.services_port,
    )
    preview = OffshorePreview(
        state,
        args.window_name,
        args.preview_width,
        args.preview_height,
        args.pip_scale,
        args.preview_fps,
    )

    try:
        temp_config_dir, scene_name, sim_config_path = make_runtime_scene_config(args, route)

        projectairsim_log().info("Connecting to Project AirSim")
        client.connect()
        world = World(
            client,
            scene_name,
            delay_after_load_sec=args.load_delay_sec,
            sim_config_path=sim_config_path,
        )
        projectairsim_log().info(
            "Loaded %s. If PX4 was already waiting on TCP port 4560 before this "
            "scene loaded, restart PX4 now and let this script keep waiting.",
            scene_name,
        )

        drone = Drone(client, world, args.drone_name)
        fpv_topic = drone.sensors.get(args.fpv_camera, {}).get("scene_camera")
        chase_topic = drone.sensors.get(args.chase_camera, {}).get("scene_camera")
        if not fpv_topic or not chase_topic:
            available_topics = {
                sensor: sorted(topics.keys())
                for sensor, topics in drone.sensors.items()
                if topics
            }
            raise RuntimeError(
                "FPV or chase camera topic is not available. Available sensor topics: "
                f"{available_topics}"
            )

        preview.start()
        client.subscribe(fpv_topic, preview.receive_fpv)
        client.subscribe(chase_topic, preview.receive_chase)
        projectairsim_log().info(
            "Preview opened. Controls: e next waypoint, r previous waypoint, q/esc quit"
        )

        await wait_for_px4_ready(drone, args.px4_ready_timeout_sec)
        initial_pose_ned = get_pose_position_ned(drone)
        initial_pose_route = ned_to_route(initial_pose_ned)
        state.update_pose(initial_pose_route, heading_deg_360(get_pose_yaw_ned(drone)))
        projectairsim_log().info(
            "Initial NED %s",
            format_vector3(initial_pose_route),
        )

        await run_teleport_viewer(drone, state, args)
    finally:
        state.request_stop()
        preview.stop()
        if preview.error is not None:
            projectairsim_log().warning("Preview stopped with error: %s", preview.error)
        try:
            client.unsubscribe_all()
        except Exception:
            pass
        try:
            client.disconnect()
        except Exception:
            pass
        if temp_config_dir is not None:
            temp_config_dir.cleanup()


def build_parser():
    parser = argparse.ArgumentParser(description="PX4 offshore wind farm route demo.")
    parser.add_argument("--scene", default="scene_px4_sitl.jsonc")
    parser.add_argument(
        "--sim-config-path",
        default=str(EXAMPLE_SCRIPTS_DIR / "sim_config"),
        help="Directory containing Project AirSim scene/robot configs.",
    )
    parser.add_argument("--drone-name", default="Drone1")
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--topics-port", type=int, default=8989)
    parser.add_argument("--services-port", type=int, default=8990)
    parser.add_argument("--load-delay-sec", type=float, default=2.0)
    parser.add_argument("--pose-report-interval-sec", type=float, default=2.0)
    parser.add_argument("--px4-ready-timeout-sec", type=float, default=300.0)
    parser.add_argument("--fpv-camera", default="FrontCamera")
    parser.add_argument("--chase-camera", default="Chase")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fov-degrees", type=float, default=90.0)
    parser.add_argument("--camera-capture-interval-sec", type=float, default=0.03)
    parser.add_argument("--window-name", default="Offshore PX4 Demo")
    parser.add_argument("--preview-width", type=int, default=1280)
    parser.add_argument("--preview-height", type=int, default=720)
    parser.add_argument("--preview-fps", type=float, default=30.0)
    parser.add_argument("--pip-scale", type=float, default=0.30)
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    asyncio.run(run_demo(parsed_args))
