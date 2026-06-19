# How to run the code

## How to run the simulation with autonavigation using PX4 SITL over Blocks environment

This example uses airsim provided Blocks.uproject

1. Open the Blocks.uproject in Unreal Engine Editor and press play button
2. Launch PX4 by running `make px4_sitl_default none_iris`
3. Run `python ./DroneSimDev/client/python/example_user_scripts/px4_astar_autopilot.py` and watch the drone fly a mission

```python
python px4_astar_autopilot.py `
  --start "30,0,-6" `
  --goal "30,-48,-10" `
  --velocity-mps 2 `
  --land-at-goal `
  --print-waypoints `
  --px4-ready-timeout-sec 300
```

### Fast bug-fixing loop for `px4_astar_autopilot.py`

Keep Unreal open and pressing Play. Most Python bugs do not require restarting the
uproject. Treat the three pieces separately:

1. Unreal / Project AirSim: restart only if the editor crashes, Play stops, the
   sim server stops accepting clients, or you changed Unreal assets/plugins.
2. PX4 SITL: restart when Project AirSim reloads the PX4 scene and PX4 is still
   waiting on TCP port `4560`, when you stopped Unreal, or when a crash leaves
   PX4 armed/offboard and the next run cannot arm.
3. Python script: restart this freely. This is the normal inner loop.

For A*, map, argument parsing, and waypoint bugs, skip PX4 completely:

```powershell
cd W:\UnsyncProjects\DroneSimDev\client\python\example_user_scripts
python px4_astar_autopilot.py `
  --scene scene_px4_sitl.jsonc `
  --start "30,0,-6" `
  --goal "30,-48,-10" `
  --plan-only `
  --print-waypoints
```

For flight bugs, use this loop:

```powershell
cd W:\UnsyncProjects\DroneSimDev\client\python\example_user_scripts
python px4_astar_autopilot.py `
  --scene scene_px4_sitl.jsonc `
  --start "30,0,-6" `
  --goal "30,-48,-10" `
  --velocity-mps 2 `
  --land-at-goal `
  --print-waypoints `
  --px4-ready-timeout-sec 300
```

If the script reaches `Waiting for PX4...` but PX4 still says
`Waiting for simulator to connect on TCP port 4560`, restart PX4 only:

```bash
make px4_sitl_default none_iris
```

When the connection is healthy, PX4 prints messages like:

```text
Simulator connected on UDP port 14560
EKF GPS checks passed
EKF commencing GPS fusion
```

This script now tries to cancel the active task, disarm, and disable API control
on exit, even if a Python exception happens after arming. That should reduce the
"crash once, restart everything" pain during debugging.

## How to run the simulation with rgb, depth, and lidar cameras

This example uses airsim provided Blocks.uproject

1. Open the Blocks.uproject in Unreal Engine Editor and press play button
2. Run `python ./DroneSimDev/client/python/example_user_scripts/check_all_cameras.py --camera [rgb|depth|lidar]`

```python
python check_all_cameras.py --camera all --fly-pattern
```

```python
python check_all_cameras.py --camera depth --depth-min-m 0.1 --depth-max-m 80
```

Units of distance are provided in meters for the depth camera. I found out the 30-80 meters works fine with Blocs.uproject environment.

