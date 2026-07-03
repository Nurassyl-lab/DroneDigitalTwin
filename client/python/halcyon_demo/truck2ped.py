"""
Keyboard drone + BP_Truck controller for the truck-to-pedestrian demo.

Examples:
    python truck2ped.py --start 0 0 -2
    python truck2ped.py --start "10,0,-3" --truck BP_Truck_1

Controls:
    W/S: drone forward/back
    A/D: drone left/right
    Up/Down: drone up/down
    Left/Right: drone yaw
    N: stop truck by setting TargetSpeed to 0
    M: move truck by setting TargetSpeed to 100
    L: land
    Q: quit

The truck movement remains inside BP_Truck Tick. Python calls
SetTruckSpeed(NewSpeed), which should update TargetSpeed. If that event is not
available, Python falls back to setting TargetSpeed directly. It never writes
CurrentSpeed and never teleports the truck.
"""

import argparse
import asyncio
import math
from pathlib import Path
import re
import shutil
import tempfile
import time

import commentjson
import projectairsim
from projectairsim import Drone, World
from projectairsim.image_utils import ImageDisplay
from projectairsim.types import Pose, Quaternion, Vector3
from projectairsim.utils import projectairsim_log, rpy_to_quaternion

keyboard = None

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SIM_CONFIG_PATH = SCRIPT_DIR.parent / "example_user_scripts" / "sim_config"


def parse_vector3(value):
    if isinstance(value, list):
        parts = value
    else:
        parts = value.replace(",", " ").split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected three coordinates, got {len(parts)} from '{value}'"
        )
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Coordinates must be numeric: '{value}'"
        ) from exc


def parse_start(values):
    if len(values) == 1:
        return parse_vector3(values[0])
    return parse_vector3(values)


def parse_float3(value, default=None):
    if value is None:
        return default if default is not None else [0.0, 0.0, 0.0]
    return [float(part) for part in value.replace(",", " ").split()]


def make_pose_ned(position_ned):
    return Pose(
        {
            "translation": Vector3(
                {
                    "x": position_ned[0],
                    "y": position_ned[1],
                    "z": position_ned[2],
                }
            ),
            "rotation": Quaternion({"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}),
            "frame_id": "DEFAULT_ID",
        }
    )


def make_camera_pose(translation_xyz, angle_down_deg):
    w, x, y, z = rpy_to_quaternion(0.0, math.radians(-angle_down_deg), 0.0)
    return Pose(
        {
            "translation": Vector3(
                {
                    "x": translation_xyz[0],
                    "y": translation_xyz[1],
                    "z": translation_xyz[2],
                }
            ),
            "rotation": Quaternion({"w": w, "x": x, "y": y, "z": z}),
            "frame_id": "DEFAULT_ID",
        }
    )


def get_pose_position_ned(drone):
    position = drone.get_ground_truth_kinematics()["pose"]["position"]
    return [float(position["x"]), float(position["y"]), float(position["z"])]


def print_live_ned(drone, last_print_at, interval_sec, force=False):
    now = time.time()
    if not force and now - last_print_at < interval_sec:
        return last_print_at

    position = get_pose_position_ned(drone)
    print(
        f"[LIVE NED] x={position[0]:8.2f}  "
        f"y={position[1]:8.2f}  z={position[2]:8.2f}",
        flush=True,
    )
    return now


def resolve_config_path(config_name, sim_config_path):
    path = Path(config_name)
    if path.is_absolute():
        return path
    return Path(sim_config_path) / config_name


def load_jsonc(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return commentjson.load(handle)


def write_jsonc(path, data):
    Path(path).write_text(commentjson.dumps(data, indent=2) + "\n", encoding="utf-8")


def default_front_camera_sensor(args):
    return {
        "id": "FrontCamera",
        "type": "camera",
        "enabled": True,
        "parent-link": "Frame",
        "capture-interval": args.camera_capture_interval_sec,
        "capture-settings": [
            {
                "image-type": 0,
                "width": args.camera_capture_width,
                "height": args.camera_capture_height,
                "fov-degrees": args.camera_fov_degrees,
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


def ensure_scene_camera_capture(sensor, args):
    capture_settings = sensor.setdefault("capture-settings", [])
    scene_capture = next(
        (capture for capture in capture_settings if capture.get("image-type") == 0),
        None,
    )
    if scene_capture is None:
        capture_settings.append(default_front_camera_sensor(args)["capture-settings"][0])
        return

    scene_capture["capture-enabled"] = True
    scene_capture["streaming-enabled"] = True
    scene_capture.setdefault("width", args.camera_capture_width)
    scene_capture.setdefault("height", args.camera_capture_height)
    scene_capture.setdefault("fov-degrees", args.camera_fov_degrees)
    scene_capture.setdefault("pixels-as-float", False)
    scene_capture.setdefault("compress", False)
    scene_capture.setdefault("target-gamma", 2.5)


def ensure_requested_camera(robot_config, camera_sensor_id, args):
    sensors = robot_config.setdefault("sensors", [])
    sensor = next(
        (candidate for candidate in sensors if candidate.get("id") == camera_sensor_id),
        None,
    )

    if sensor is None and camera_sensor_id == "FrontCamera":
        sensors.append(default_front_camera_sensor(args))
        print("Added runtime FrontCamera RGB sensor to the drone config.")
        return

    if sensor is not None and sensor.get("type") == "camera":
        sensor["enabled"] = True
        ensure_scene_camera_capture(sensor, args)


def make_runtime_scene_config(args):
    scene_path = resolve_config_path(args.sceneconfigfile, args.simconfigpath)
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene config not found: {scene_path}")

    scene_config = load_jsonc(scene_path)
    target_actor = next(
        (
            actor
            for actor in scene_config.get("actors", [])
            if actor.get("type") == "robot" and actor.get("name") == args.drone_name
        ),
        None,
    )
    if target_actor is None or not target_actor.get("robot-config"):
        raise RuntimeError(
            f"Could not find robot actor '{args.drone_name}' with a robot-config"
        )

    temp_dir = tempfile.TemporaryDirectory(prefix="truck2ped_scene_")
    temp_config_dir = Path(temp_dir.name)

    try:
        for actor_index, actor in enumerate(scene_config.get("actors", [])):
            if actor.get("type") != "robot" or not actor.get("robot-config"):
                continue

            robot_config_path = resolve_config_path(
                actor["robot-config"],
                scene_path.parent,
            )
            robot_config = load_jsonc(robot_config_path)

            if actor is target_actor:
                ensure_requested_camera(robot_config, args.camera, args)

            output_name = f"{robot_config_path.stem}_{actor_index}_truck2ped.jsonc"
            write_jsonc(temp_config_dir / output_name, robot_config)
            actor["robot-config"] = output_name

        for env_actor in scene_config.get("environment-actors", []):
            if env_actor.get("type") != "env_actor" or not env_actor.get(
                "env-actor-config"
            ):
                continue
            env_config_path = resolve_config_path(
                env_actor["env-actor-config"],
                scene_path.parent,
            )
            shutil.copy2(env_config_path, temp_config_dir / env_config_path.name)
            env_actor["env-actor-config"] = env_config_path.name

        write_jsonc(temp_config_dir / scene_path.name, scene_config)
    except Exception:
        temp_dir.cleanup()
        raise

    return temp_dir, scene_path.name, str(temp_config_dir)


def read_camera_origin(scene_name, sim_config_path, drone_name, camera_sensor_id):
    try:
        scene_config = load_jsonc(resolve_config_path(scene_name, sim_config_path))
        actor = next(
            (
                item
                for item in scene_config.get("actors", [])
                if item.get("type") == "robot" and item.get("name") == drone_name
            ),
            None,
        )
        if actor is None or not actor.get("robot-config"):
            return [0.5, 0.0, 0.0]

        robot_config = load_jsonc(
            resolve_config_path(actor["robot-config"], sim_config_path)
        )
        sensor = next(
            (
                item
                for item in robot_config.get("sensors", [])
                if item.get("id") == camera_sensor_id
            ),
            None,
        )
        if sensor is None:
            return [0.5, 0.0, 0.0]
        return parse_float3(sensor.get("origin", {}).get("xyz"), [0.5, 0.0, 0.0])
    except Exception as exc:
        projectairsim_log().warning(
            "Could not read %s origin; using default front camera origin: %s",
            camera_sensor_id,
            exc,
        )
        return [0.5, 0.0, 0.0]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fly Drone1 while controlling BP_Truck TargetSpeed."
    )
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--topicsport", type=int, default=8989)
    parser.add_argument("--servicesport", type=int, default=8990)
    parser.add_argument("--sceneconfigfile", default="scene_basic_drone.jsonc")
    parser.add_argument(
        "--simconfigpath",
        default=str(DEFAULT_SIM_CONFIG_PATH),
        help="Directory containing ProjectAirSim scene config files.",
    )
    parser.add_argument("--drone-name", default="Drone1")
    parser.add_argument(
        "--start",
        nargs="+",
        type=str,
        default=None,
        help="Optional drone spawn NED x y z, for example --start 0 0 -2.",
    )
    parser.add_argument(
        "--truck",
        default="Truck",
        help="Truck actor name/substring or Unreal tag. Defaults to 'Truck'.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Optional initial truck TargetSpeed before keyboard control starts.",
    )
    parser.add_argument("--truck-stop-speed", type=float, default=0.0)
    parser.add_argument("--truck-move-speed", type=float, default=100.0)
    parser.add_argument("--flight-speed", type=float, default=5.0)
    parser.add_argument("--yaw-speed", type=float, default=20.0)
    parser.add_argument("--command-duration-sec", type=float, default=0.1)
    parser.add_argument("--live-ned-interval-sec", type=float, default=0.5)
    parser.add_argument("--no-live-ned", action="store_true")
    parser.add_argument(
        "--camera",
        default="FrontCamera",
        help="RGB camera sensor id to show and tilt. Defaults to FrontCamera.",
    )
    parser.add_argument(
        "--front-rgb-angle",
        "--camera-angle-deg",
        dest="camera_angle_deg",
        type=float,
        default=22.0,
        help="Tilt the selected front RGB camera down by this many degrees.",
    )
    parser.add_argument("--camera-capture-width", type=int, default=1280)
    parser.add_argument("--camera-capture-height", type=int, default=720)
    parser.add_argument("--camera-capture-interval-sec", type=float, default=0.03)
    parser.add_argument("--camera-fov-degrees", type=float, default=90.0)
    parser.add_argument("--camera-display-width", type=int, default=960)
    parser.add_argument("--camera-display-height", type=int, default=540)
    parser.add_argument("--no-camera", action="store_true")
    return parser


def find_matching_actors(world, pattern):
    try:
        return world.list_objects(pattern)
    except Exception:
        return []


def resolve_truck_actor(world, truck_name_or_tag):
    escaped = re.escape(truck_name_or_tag)
    exact_or_contains = find_matching_actors(world, f".*{escaped}.*")
    if exact_or_contains:
        if truck_name_or_tag in exact_or_contains:
            return truck_name_or_tag
        return exact_or_contains[0]

    unreal_instance_match = re.match(r"^(.*)_(\d+)$", truck_name_or_tag)
    if unreal_instance_match and not truck_name_or_tag.endswith("_C"):
        candidate = f"{unreal_instance_match.group(1)}_C_{unreal_instance_match.group(2)}"
        candidate_matches = find_matching_actors(world, f".*{re.escape(candidate)}.*")
        if candidate_matches:
            return candidate_matches[0]

    bp_matches = find_matching_actors(world, r".*BP_Truck.*")
    if bp_matches:
        print("Available BP_Truck-like actor names:")
        for name in bp_matches:
            print(f"  {name}")

    return truck_name_or_tag


def set_truck_speed(world, truck_name_or_tag, speed):
    truck_actor = resolve_truck_actor(world, truck_name_or_tag)
    if truck_actor != truck_name_or_tag:
        print(f"Resolved truck '{truck_name_or_tag}' to actor '{truck_actor}'.")

    print(f"Setting {truck_actor} target speed to {speed:g}...")

    try:
        called = world.call_actor_event(
            truck_actor,
            "SetTruckSpeed",
            {"NewSpeed": speed},
        )
    except RuntimeError as exc:
        if "CallActorFloatEvent" in str(exc) and "not supported" in str(exc):
            print(
                "The running Unreal plugin does not expose CallActorFloatEvent. "
                "Recompile/rebuild the WalkingNPCs ProjectAirSim plugin, restart "
                "the editor, press Play, then run this script again."
            )
        return False

    if called:
        print(
            "Called BP_Truck.SetTruckSpeed(NewSpeed); "
            "Blueprint will interpolate CurrentSpeed."
        )
        return True

    print("SetTruckSpeed was not found; trying direct TargetSpeed variable set...")
    set_property = world.set_actor_float_property(truck_actor, "TargetSpeed", speed)
    if set_property:
        print("Set BP_Truck TargetSpeed variable directly.")
        return True

    print(
        "Could not find the truck/event/property. Check the actor name/tag, "
        "SetTruckSpeed(NewSpeed), and TargetSpeed variable."
    )
    return False


async def takeoff(drone):
    print("Arming the drone...")
    drone.arm()
    print("Taking off...")
    await drone.takeoff_async()
    time.sleep(1)


async def land(drone):
    print("Landing...")
    await drone.land_async()
    print("Disarming the drone...")
    drone.disarm()


def require_keyboard():
    global keyboard
    if keyboard is not None:
        return keyboard

    try:
        import keyboard as keyboard_module
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This script needs the 'keyboard' Python package. Install it in "
            "DroneSimDev_ENV with: pip install keyboard"
        ) from exc

    keyboard = keyboard_module
    return keyboard


def require_scene_camera_topic(drone, camera_sensor_id):
    if camera_sensor_id not in drone.sensors:
        raise RuntimeError(
            f"Camera sensor '{camera_sensor_id}' is not available. "
            f"Available sensors: {sorted(drone.sensors.keys())}"
        )
    if "scene_camera" not in drone.sensors[camera_sensor_id]:
        raise RuntimeError(
            f"Sensor '{camera_sensor_id}' has no RGB scene_camera topic. "
            f"Available topics: {sorted(drone.sensors[camera_sensor_id].keys())}"
        )
    return drone.sensors[camera_sensor_id]["scene_camera"]


def start_front_rgb_display(client, drone, args, scene_name, sim_config_path):
    camera_origin = read_camera_origin(
        scene_name,
        sim_config_path,
        args.drone_name,
        args.camera,
    )
    pose = make_camera_pose(camera_origin, args.camera_angle_deg)
    if drone.set_camera_pose(args.camera, pose):
        print(
            f"Set {args.camera} RGB camera angle to "
            f"{args.camera_angle_deg:g} degrees down."
        )
    else:
        print(f"Warning: failed to set {args.camera} camera pose.")

    camera_topic = require_scene_camera_topic(drone, args.camera)
    image_display = ImageDisplay(num_subwin=1)
    window_name = f"Front RGB ({args.camera})"
    image_display.add_image(
        window_name,
        resize_x=args.camera_display_width,
        resize_y=args.camera_display_height,
    )
    image_display.start()
    client.subscribe(
        camera_topic,
        lambda _, image: image_display.receive(image, window_name),
    )
    print(f"Subscribed front RGB camera topic: {camera_topic}")
    return image_display, camera_topic


async def run_keyboard_control(drone, world, args):
    keyboard_module = require_keyboard()

    drone.enable_api_control()
    await takeoff(drone)

    print("\n--- Drone + Truck Keyboard Control ---")
    print("W/S: drone forward/back")
    print("A/D: drone left/right")
    print("Up/Down: drone up/down")
    print("Left/Right: drone yaw")
    print(f"N: truck stop TargetSpeed={args.truck_stop_speed:g}")
    print(f"M: truck move TargetSpeed={args.truck_move_speed:g}")
    print("L: land")
    print("Q: quit")
    if not args.no_live_ned:
        print(f"Live NED: printing every {args.live_ned_interval_sec:g}s")
    print("-------------------------------------")

    keep_running = True
    last_live_ned_at = 0.0
    n_was_pressed = False
    m_was_pressed = False

    while keep_running:
        if not args.no_live_ned:
            last_live_ned_at = print_live_ned(
                drone,
                last_live_ned_at,
                args.live_ned_interval_sec,
            )

        vx, vy, vz, yaw_rate = 0.0, 0.0, 0.0, 0.0

        if keyboard_module.is_pressed("w"):
            vx = args.flight_speed
        elif keyboard_module.is_pressed("s"):
            vx = -args.flight_speed

        if keyboard_module.is_pressed("a"):
            vy = -args.flight_speed
        elif keyboard_module.is_pressed("d"):
            vy = args.flight_speed

        if keyboard_module.is_pressed("up"):
            vz = -args.flight_speed
        elif keyboard_module.is_pressed("down"):
            vz = args.flight_speed

        if keyboard_module.is_pressed("left"):
            yaw_rate = -args.yaw_speed
        elif keyboard_module.is_pressed("right"):
            yaw_rate = args.yaw_speed

        n_is_pressed = keyboard_module.is_pressed("n")
        if n_is_pressed and not n_was_pressed:
            set_truck_speed(world, args.truck, args.truck_stop_speed)
        n_was_pressed = n_is_pressed

        m_is_pressed = keyboard_module.is_pressed("m")
        if m_is_pressed and not m_was_pressed:
            set_truck_speed(world, args.truck, args.truck_move_speed)
        m_was_pressed = m_is_pressed

        if keyboard_module.is_pressed("l"):
            await land(drone)
            if not args.no_live_ned:
                last_live_ned_at = print_live_ned(
                    drone,
                    last_live_ned_at,
                    args.live_ned_interval_sec,
                    force=True,
                )
            keep_running = False

        if keyboard_module.is_pressed("q"):
            if not args.no_live_ned:
                last_live_ned_at = print_live_ned(
                    drone,
                    last_live_ned_at,
                    args.live_ned_interval_sec,
                    force=True,
                )
            keep_running = False

        if vx != 0.0 or vy != 0.0 or vz != 0.0:
            await drone.move_by_velocity_body_frame_async(
                vx,
                vy,
                vz,
                args.command_duration_sec,
            )
        if yaw_rate != 0.0:
            await drone.rotate_by_yaw_rate_async(yaw_rate, args.command_duration_sec)

        await asyncio.sleep(0.01)


async def main():
    args = build_parser().parse_args()
    if args.start is not None:
        args.start = parse_start(args.start)

    temp_scene_dir = None
    image_display = None
    camera_topic = None
    drone = None

    client = projectairsim.ProjectAirSimClient(
        address=args.address,
        port_topics=args.topicsport,
        port_services=args.servicesport,
    )

    try:
        temp_scene_dir, scene_name, sim_config_path = make_runtime_scene_config(args)

        client.connect()
        world = World(
            client=client,
            scene_config_name=scene_name,
            sim_config_path=sim_config_path,
        )
        drone = Drone(client, world, args.drone_name)

        if args.start is not None:
            print(f"Spawning {args.drone_name} at NED {args.start}...")
            drone.set_pose(make_pose_ned(args.start), reset_kinematics=True)
            time.sleep(1)
            if not args.no_live_ned:
                print_live_ned(
                    drone,
                    last_print_at=0.0,
                    interval_sec=args.live_ned_interval_sec,
                    force=True,
                )

        if args.speed is not None:
            set_truck_speed(world, args.truck, args.speed)

        if not args.no_camera:
            image_display, camera_topic = start_front_rgb_display(
                client,
                drone,
                args,
                scene_name,
                sim_config_path,
            )

        await run_keyboard_control(drone, world, args)
        return 0

    except Exception as exc:
        print(f"An error occurred: {exc}")
        return 1

    finally:
        if camera_topic:
            try:
                client.unsubscribe(camera_topic)
            except Exception:
                pass
        if image_display:
            image_display.stop()
        if drone:
            try:
                drone.disarm()
                drone.disable_api_control()
            except Exception:
                pass
        client.disconnect()
        if temp_scene_dir:
            temp_scene_dir.cleanup()
        print("Cleaned up and disconnected.")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
