"""
Validate Project AirSim RGB, depth, and lidar sensor streams.

Examples:
    python check_all_cameras.py --camera rgb
    python check_all_cameras.py --camera front_rgb
    python check_all_cameras.py --camera left_rgb
    python check_all_cameras.py --camera right_rgb
    python check_all_cameras.py --camera down_rgb
    python check_all_cameras.py --camera depth
    python check_all_cameras.py --camera lidar
    python check_all_cameras.py --camera all --fly-pattern
    python check_all_cameras.py --camera depth --fly-pattern --avoid-obstacles
"""

import argparse
import asyncio
import queue
import time
from dataclasses import dataclass
from threading import Thread
from typing import Dict, Iterable, Optional

import numpy as np

from projectairsim import Drone, ProjectAirSimClient, World
from projectairsim.utils import projectairsim_log, unpack_image


RGB_CAMERA_SENSORS = {
    "rgb": "FrontCamera",
    "front_rgb": "FrontCamera",
    "left_rgb": "LeftCamera",
    "right_rgb": "RightCamera",
    "down_rgb": "DownCamera",
}
CAMERA_CHOICES = tuple(RGB_CAMERA_SENSORS.keys()) + (
    "all_rgb",
    "depth",
    "lidar",
    "all",
)
DEPTH_UNIT_CHOICES = ("m", "mm")


def depth_frame_to_meters(frame, units: str):
    depth = np.asarray(frame).squeeze().astype(np.float32)
    if units == "mm":
        return depth / 1000.0
    return depth


class DepthSafetyMonitor:
    def __init__(self, units: str, stop_distance_m: float, roi_fraction: float):
        self.units = units
        self.stop_distance_m = stop_distance_m
        self.roi_fraction = max(0.05, min(1.0, roi_fraction))
        self.closest_forward_m: Optional[float] = None
        self.last_seen_at: Optional[float] = None

    def update(self, image):
        frame = np.asarray(unpack_image(image)).squeeze()
        if frame.ndim != 2:
            return

        height, width = frame.shape
        roi_width = max(1, int(width * self.roi_fraction))
        roi_height = max(1, int(height * self.roi_fraction))
        x0 = max(0, (width - roi_width) // 2)
        y0 = max(0, (height - roi_height) // 2)
        roi = frame[y0 : y0 + roi_height, x0 : x0 + roi_width]
        valid = roi[roi > 0]
        if valid.size == 0:
            return

        self.closest_forward_m = float(depth_frame_to_meters(valid, self.units).min())
        self.last_seen_at = time.time()

    def should_stop(self) -> bool:
        if self.closest_forward_m is None or self.last_seen_at is None:
            return False
        if time.time() - self.last_seen_at > 1.0:
            return False
        return self.closest_forward_m <= self.stop_distance_m


@dataclass
class StreamStats:
    name: str
    count: int = 0
    first_seen_at: Optional[float] = None
    last_seen_at: Optional[float] = None
    last_summary: str = "no data"

    def record(self, summary: str):
        now = time.time()
        self.count += 1
        self.first_seen_at = self.first_seen_at or now
        self.last_seen_at = now
        self.last_summary = summary

    def report(self) -> str:
        age = "never" if self.last_seen_at is None else f"{time.time() - self.last_seen_at:.1f}s ago"
        return f"{self.name}: {self.count} frames, last={age}, {self.last_summary}"


class OpenCvPreview:
    def __init__(
        self,
        width: int,
        height: int,
        depth_min_m: float,
        depth_max_m: float,
        depth_units: str,
        depth_invert: bool,
    ):
        self.width = width
        self.height = height
        self.depth_min_m = max(0.0, depth_min_m)
        self.depth_max_m = max(depth_max_m, self.depth_min_m + 0.001)
        self.depth_units = depth_units
        self.depth_invert = depth_invert
        self.windows: Dict[str, Dict] = {}
        self.running = False
        self.thread = None

    def add_window(self, name: str, mode: str):
        self.windows[name] = {
            "mode": mode,
            "queue": queue.SimpleQueue(),
            "created": False,
        }

    def receive(self, name: str, image):
        if not self.running or image is None:
            return
        image_queue = self.windows[name]["queue"]
        while not image_queue.empty() and image_queue.qsize() > 3:
            image_queue.get()
        image_queue.put(image)

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = Thread(target=self._display_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        self.thread = None

    def _display_loop(self):
        import cv2

        try:
            while self.running:
                for name, state in self.windows.items():
                    image_queue = state["queue"]
                    if image_queue.empty():
                        continue

                    image = image_queue.get()
                    while not image_queue.empty():
                        image = image_queue.get()

                    frame = self._make_preview_frame(image, state["mode"])
                    if frame is None:
                        continue

                    if not state["created"]:
                        cv2.namedWindow(
                            name,
                            flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE,
                        )
                        state["created"] = True

                    frame = cv2.resize(frame, (self.width, self.height))
                    cv2.imshow(name, frame)

                if cv2.waitKey(1) == 27:
                    self.running = False
        finally:
            for name, state in self.windows.items():
                if state["created"]:
                    cv2.destroyWindow(name)
                    state["created"] = False

    def _make_preview_frame(self, image, mode: str):
        import cv2

        frame = unpack_image(image)
        if mode == "depth":
            return self._depth_to_colormap(frame)

        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 1:
            return cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
        return frame

    def _depth_to_colormap(self, frame):
        import cv2

        depth_m = depth_frame_to_meters(frame, self.depth_units)
        valid_mask = depth_m > 0

        clipped = np.clip(depth_m, self.depth_min_m, self.depth_max_m)
        normalized = (clipped - self.depth_min_m) / (
            self.depth_max_m - self.depth_min_m
        )
        if self.depth_invert:
            normalized = 1.0 - normalized

        preview = (normalized * 255.0).astype(np.uint8)
        preview[~valid_mask] = 0
        return cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)


def selected_modes(camera: str) -> Iterable[str]:
    if camera == "all":
        return ("front_rgb", "left_rgb", "right_rgb", "down_rgb", "depth", "lidar")
    if camera == "all_rgb":
        return ("front_rgb", "left_rgb", "right_rgb", "down_rgb")
    return (camera,)


def summarize_image(image) -> str:
    encoding = image.get("encoding", "unknown")
    width = image.get("width", "?")
    height = image.get("height", "?")
    data = image.get("data", [])
    data_len = len(data) if data is not None else 0
    return f"{width}x{height}, encoding={encoding}, bytes={data_len}"


def summarize_depth(image, units: str) -> str:
    try:
        frame = np.asarray(unpack_image(image)).squeeze()
        valid = frame[frame > 0]
        if valid.size == 0:
            return f"{summarize_image(image)}, depth=no valid pixels"
        valid_m = depth_frame_to_meters(valid, units)
        return (
            f"{summarize_image(image)}, depth_units={units}, "
            f"raw_min={valid.min()}, raw_max={valid.max()}, "
            f"depth_min_m={valid_m.min():.2f}, "
            f"depth_p50_m={np.percentile(valid_m, 50):.2f}, "
            f"depth_p95_m={np.percentile(valid_m, 95):.2f}, "
            f"depth_max_m={valid_m.max():.2f}"
        )
    except Exception as exc:
        return f"{summarize_image(image)}, depth_stats_error={exc}"


def summarize_lidar(lidar) -> str:
    points = lidar.get("point_cloud") or []
    intensities = lidar.get("intensity_cloud") or []
    segment_ids = lidar.get("segmentation_cloud") or []
    return (
        f"points={len(points) // 3}, intensities={len(intensities)}, "
        f"segments={len(segment_ids)}"
    )


def require_topic(drone: Drone, sensor_name: str, topic_name: str) -> str:
    if sensor_name not in drone.sensors:
        raise RuntimeError(
            f"Sensor '{sensor_name}' is not available. Available sensors: "
            f"{sorted(drone.sensors.keys())}"
        )
    if topic_name not in drone.sensors[sensor_name]:
        raise RuntimeError(
            f"Topic '{topic_name}' is not available on sensor '{sensor_name}'. "
            f"Available topics: {sorted(drone.sensors[sensor_name].keys())}"
        )
    return drone.sensors[sensor_name][topic_name]


def subscribe_rgb(client, drone, args, stats, preview, mode: str):
    sensor_name = args.rgb_sensor if mode == "rgb" and args.rgb_sensor else RGB_CAMERA_SENSORS[mode]
    topic = require_topic(drone, sensor_name, "scene_camera")
    window_name = f"{mode} - {sensor_name}"
    if preview:
        preview.add_window(window_name, "rgb")

    def callback(_, image):
        stats[mode].record(summarize_image(image))
        if preview:
            preview.receive(window_name, image)

    client.subscribe(topic, callback)
    projectairsim_log().info("Subscribed %s topic: %s", mode, topic)


def subscribe_depth(client, drone, args, stats, preview, safety_monitor=None):
    topic = require_topic(drone, args.depth_sensor, "depth_camera")
    window_name = f"Depth - {args.depth_sensor}"
    if preview:
        preview.add_window(window_name, "depth")

    def callback(_, image):
        stats["depth"].record(summarize_depth(image, args.depth_units))
        if safety_monitor:
            safety_monitor.update(image)
        if preview:
            preview.receive(window_name, image)

    client.subscribe(topic, callback)
    projectairsim_log().info("Subscribed depth topic: %s", topic)


def subscribe_lidar(client, drone, args, stats, lidar_display):
    topic = require_topic(drone, args.lidar_sensor, "lidar")

    def callback(_, lidar):
        stats["lidar"].record(summarize_lidar(lidar))
        if lidar_display:
            lidar_display.receive(lidar)

    client.subscribe(topic, callback)
    projectairsim_log().info("Subscribed lidar topic: %s", topic)


async def run_flight_pattern(
    drone: Drone,
    velocity_mps: float,
    safety_monitor: Optional[DepthSafetyMonitor] = None,
):
    if not drone.enable_api_control():
        raise RuntimeError("Failed to enable API control")
    if not drone.arm():
        raise RuntimeError("Failed to arm drone")

    commands = [
        ("up", 0.0, 0.0, -velocity_mps, 3.0),
        ("north", velocity_mps, 0.0, 0.0, 4.0),
        ("east", 0.0, velocity_mps, 0.0, 4.0),
        ("west", 0.0, -velocity_mps, 0.0, 4.0),
        ("west", 0.0, -velocity_mps, 0.0, 4.0),
        ("west", 0.0, -velocity_mps, 0.0, 4.0),
        ("north", velocity_mps, 0.0, 0.0, 4.0),
        ("north", velocity_mps, 0.0, 0.0, 4.0),
        ("north", velocity_mps, 0.0, 0.0, 4.0),
        ("north", velocity_mps, 0.0, 0.0, 4.0),
        ("north", velocity_mps, 0.0, 0.0, 4.0),
        ("down", 0.0, 0.0, velocity_mps, 3.0),
    ]
    try:
        command_step_sec = 0.25
        for label, north, east, down, duration in commands:
            projectairsim_log().info("Flight pattern move %s", label)
            remaining = duration
            while remaining > 0:
                if safety_monitor and safety_monitor.should_stop():
                    projectairsim_log().warning(
                        "Stopping flight pattern: depth obstacle at %.2fm",
                        safety_monitor.closest_forward_m,
                    )
                    task = await drone.move_by_velocity_async(0.0, 0.0, 0.0, 0.5)
                    await task
                    return

                step_duration = min(command_step_sec, remaining)
                task = await drone.move_by_velocity_async(
                    north, east, down, step_duration
                )
                await task
                remaining -= step_duration
    finally:
        drone.disarm()
        drone.disable_api_control()


async def main(args):
    modes = tuple(selected_modes(args.camera))
    if args.depth_max_m <= args.depth_min_m:
        raise ValueError("--depth-max-m must be greater than --depth-min-m")

    stats = {mode: StreamStats(mode) for mode in modes}
    client = ProjectAirSimClient(
        address=args.server_ip,
        port_topics=args.topics_port,
        port_services=args.services_port,
    )
    preview = None
    lidar_display = None
    flight_task = None
    safety_monitor = None

    try:
        client.connect()
        world = World(
            client,
            args.scene,
            delay_after_load_sec=args.load_delay_sec,
            sim_config_path=args.sim_config_path,
        )
        drone = Drone(client, world, args.drone_name)

        if not args.no_display and any(mode in RGB_CAMERA_SENSORS or mode == "depth" for mode in modes):
            preview = OpenCvPreview(
                args.preview_width,
                args.preview_height,
                args.depth_min_m,
                args.depth_max_m,
                args.depth_units,
                args.depth_invert,
            )

        if not args.no_display and "lidar" in modes:
            from projectairsim.lidar_utils import LidarDisplay

            lidar_display = LidarDisplay(
                win_name=f"LIDAR - {args.lidar_sensor}",
                width=args.preview_width,
                height=args.preview_height,
                view=LidarDisplay.VIEW_FORWARD,
            )

        if args.avoid_obstacles and "depth" in modes:
            safety_monitor = DepthSafetyMonitor(
                args.depth_units,
                args.obstacle_stop_distance_m,
                args.obstacle_roi_fraction,
            )

        for mode in modes:
            if mode in RGB_CAMERA_SENSORS:
                subscribe_rgb(client, drone, args, stats, preview, mode)
        if "depth" in modes:
            subscribe_depth(client, drone, args, stats, preview, safety_monitor)
        if "lidar" in modes:
            subscribe_lidar(client, drone, args, stats, lidar_display)

        if preview:
            preview.start()
        if lidar_display:
            lidar_display.start()

        if args.fly_pattern:
            flight_task = asyncio.create_task(
                run_flight_pattern(drone, args.velocity_mps, safety_monitor)
            )

        started_at = time.time()
        last_report_at = 0.0
        while args.duration_sec <= 0 or time.time() - started_at < args.duration_sec:
            elapsed = time.time() - started_at
            if elapsed - last_report_at >= args.report_every_sec:
                for stream_stats in stats.values():
                    projectairsim_log().info(stream_stats.report())
                last_report_at = elapsed

            if preview and not preview.running:
                break
            if flight_task and flight_task.done() and args.duration_sec <= 0:
                break

            await asyncio.sleep(0.2)

        if flight_task:
            await flight_task

        missing = [name for name, stream_stats in stats.items() if stream_stats.count == 0]
        if missing:
            raise RuntimeError(f"No data received for: {', '.join(missing)}")

        for stream_stats in stats.values():
            projectairsim_log().info("Final %s", stream_stats.report())

    finally:
        if preview:
            preview.stop()
        if lidar_display:
            lidar_display.stop()
        if client.state:
            client.disconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate RGB, depth, and lidar streams from a Project AirSim drone."
    )
    parser.add_argument(
        "--camera",
        type=str,
        choices=CAMERA_CHOICES,
        default="all",
        help=(
            "Sensor stream to validate: rgb/front_rgb/left_rgb/right_rgb/"
            "down_rgb/all_rgb/depth/lidar/all."
        ),
    )
    parser.add_argument(
        "--scene",
        default="scene_lidar_drone.jsonc",
        help="Scene config to load. Default includes RGB, depth, and lidar.",
    )
    parser.add_argument("--sim-config-path", default="sim_config/")
    parser.add_argument("--drone-name", default="Drone1")
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--topics-port", type=int, default=8989)
    parser.add_argument("--services-port", type=int, default=8990)
    parser.add_argument("--load-delay-sec", type=float, default=2.0)
    parser.add_argument("--duration-sec", type=float, default=30.0)
    parser.add_argument("--report-every-sec", type=float, default=2.0)
    parser.add_argument(
        "--rgb-sensor",
        default=None,
        help="Override the sensor used by --camera rgb. Directional rgb modes use their named sensors.",
    )
    parser.add_argument(
        "--depth-sensor",
        default="FrontCamera",
        help="Depth camera sensor to validate. Use DownCamera for the old downward view.",
    )
    parser.add_argument(
        "--depth-min-m",
        type=float,
        default=0.1,
        help="Nearest depth value mapped into the preview color range.",
    )
    parser.add_argument(
        "--depth-max-m",
        type=float,
        default=15.0,
        help="Farthest depth value mapped into the preview color range.",
    )
    parser.add_argument(
        "--depth-units",
        type=str,
        choices=DEPTH_UNIT_CHOICES,
        default="m",
        help="Units used by the raw depth stream before preview scaling.",
    )
    parser.add_argument(
        "--no-depth-invert",
        dest="depth_invert",
        action="store_false",
        help="Map near depth to cool colors and far depth to warm colors.",
    )
    parser.set_defaults(depth_invert=True)
    parser.add_argument("--lidar-sensor", default="lidar1")
    parser.add_argument("--preview-width", type=int, default=800)
    parser.add_argument("--preview-height", type=int, default=450)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--fly-pattern", action="store_true")
    parser.add_argument("--velocity-mps", type=float, default=2.0)
    parser.add_argument(
        "--avoid-obstacles",
        action="store_true",
        help="Use the selected depth stream to stop --fly-pattern before a close obstacle.",
    )
    parser.add_argument("--obstacle-stop-distance-m", type=float, default=3.0)
    parser.add_argument("--obstacle-roi-fraction", type=float, default=0.35)
    return parser


if __name__ == "__main__":
    asyncio.run(main(build_parser().parse_args()))
