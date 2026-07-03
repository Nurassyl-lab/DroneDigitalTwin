"""
Control BP_Truck target speed through ProjectAirSim without teleporting the truck.

Examples:
    python truck2ped.py --truck BP_Truck_C_1 --speed 0
    python truck2ped.py --truck BP_Truck_1 --speed 700
    python truck2ped.py --truck Truck --stop
    python truck2ped.py --truck Truck --demo

The Unreal actor can be addressed by exact/partial actor name or by an Actor
tag such as "Truck". The Blueprint should expose SetTruckSpeed(NewSpeed).
SetTruckSpeed should update TargetSpeed; BP_Truck Tick should interpolate
CurrentSpeed toward TargetSpeed and move the actor.
"""

import argparse
import re
import time

import projectairsim


def build_parser():
    parser = argparse.ArgumentParser(
        description="Set BP_Truck TargetSpeed from ProjectAirSim Python."
    )
    parser.add_argument(
        "--address",
        default="127.0.0.1",
        help="IP address of the ProjectAirSim host.",
    )
    parser.add_argument(
        "--topicsport",
        type=int,
        default=8989,
        help="ProjectAirSim topic pub-sub port.",
    )
    parser.add_argument(
        "--servicesport",
        type=int,
        default=8990,
        help="ProjectAirSim services port.",
    )
    parser.add_argument(
        "--sceneconfigfile",
        default="scene_basic_drone.jsonc",
        help="Scene config to load when --load-scene is set.",
    )
    parser.add_argument(
        "--simconfigpath",
        default="sim_config/",
        help="Directory containing ProjectAirSim scene config files.",
    )
    parser.add_argument(
        "--load-scene",
        action="store_true",
        help="Load --sceneconfigfile before controlling the truck.",
    )
    parser.add_argument(
        "--scene-id",
        default="DefaultScene",
        help=(
            "ProjectAirSim scene ID to attach to when --load-scene is not set. "
            "Pressing Play in Unreal normally starts 'DefaultScene'."
        ),
    )
    parser.add_argument(
        "--truck",
        default="Truck",
        help="Truck actor name/substring or Unreal tag. Defaults to 'Truck'.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=700.0,
        help="Truck target speed to set. Use 0 to brake/stop, 700 to move.",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Shortcut for --speed 0.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Stop, wait, move at 700, wait, then stop again.",
    )
    parser.add_argument(
        "--demo-wait-sec",
        type=float,
        default=3.0,
        help="Seconds to wait between demo speed changes.",
    )
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
        print("Called BP_Truck.SetTruckSpeed(NewSpeed); Blueprint will interpolate CurrentSpeed.")
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


def main():
    args = build_parser().parse_args()
    if args.stop:
        args.speed = 0.0

    client = projectairsim.ProjectAirSimClient(
        address=args.address,
        port_topics=args.topicsport,
        port_services=args.servicesport,
    )

    try:
        client.connect()
        world = projectairsim.World(
            client=client,
            scene_config_name=args.sceneconfigfile if args.load_scene else "",
            sim_config_path=args.simconfigpath,
            scene_id=args.scene_id,
        )

        if args.demo:
            if not set_truck_speed(world, args.truck, 0.0):
                return 1
            time.sleep(args.demo_wait_sec)
            if not set_truck_speed(world, args.truck, 700.0):
                return 1
            time.sleep(args.demo_wait_sec)
            if not set_truck_speed(world, args.truck, 0.0):
                return 1
            return 0

        return 0 if set_truck_speed(world, args.truck, args.speed) else 1
    finally:
        client.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
