# How to run the code

## Setup environment
1. Download environment into project root folder. Download from: [https://drive.google.com/file/d/1cp5fLizh9YM2r82fpCsFRUhZ__Qp8qbL/view?usp=sharing](https://drive.google.com/file/d/1cp5fLizh9YM2r82fpCsFRUhZ__Qp8qbL/view?usp=sharing)
2. Unpack the downloaded file `tar -xzvf LowPolyRiverForest_Win11.tar.gz`, it should create a folder `.\unreal\LowPolyRiverForest\`

## How to run the simulation over River-side forest environment

This example uses Free Fab provided River-side ForestDomeEnv.uproject

### How to scan the map

1. Open the River-side ForestDomeEnv.uproject in Unreal Engine Editor and press play button
2. Run

```python
   python px4_map_viewer.py `
  --start "40,-20,-6" `
   --goal "50,20,-6" `
   --slice-z-ned -8 `
   --resolution-m 1 `
   --grid-step-m 10 `
   --label-step-m 20 `
   --output riverside_forest.png `
   --output-3d riverside_forest_3d.png `
   --map-size "500,500,10"
```

It saves both, 2D and 3D plots of the map.


### How to fly a mission manually using keyboard

1. Open the River-side ForestDomeEnv.uproject in Unreal Engine Editor and press play button
2. Run PX4 keyboard control from the same script/scene used for A* missions. Use the printed live NED as the start point for PX4 missions.
3. When the script says the generated scene is loaded and is waiting for PX4, launch or restart PX4 by running `make px4_sitl_default none_iris`

```python
python px4_astar_autopilot.py `
  --keyboard-control `
  --start "72,-8,-4" `
  --start-as-scene-origin `
  --px4-ready-timeout-sec 300 `
  --live-ned-interval-sec 1.0 `
  --keyboard-acceleration-limit-mps2 4 `
  --keyboard-yaw-acceleration-dps2 110
```

### How to fly a mission with PX4

1. Open the River-side ForestDomeEnv.uproject in Unreal Engine Editor and press play button
2. Run the command below, the drone will cross the river
3. When the script says the generated scene is loaded and is waiting for PX4, launch or restart PX4 by running `make px4_sitl_default none_iris`

```python
python px4_astar_autopilot.py `
  --scene scene_px4_sitl.jsonc `
  --start "72,-8,-4" `
  --start-as-scene-origin `
  --goal "33, -19, -6" `
  --velocity-mps 2 `
  --land-at-goal `
  --print-waypoints `
  --px4-ready-timeout-sec 300 `
  --start-as-scene-origin `
  --live-ned-interval-sec 1 `
  --acceleration-limit-mps2 1.5 `
  --slowdown-distance-m 6 `
  --waypoint-acceptance-m 1.5
```

- add `--plan-only` if you want to see the planned path without flying the drone.
- add `--acceleration-limit-mps2 1.5 --slowdown-distance-m 6 --waypoint-acceptance-m 1.5` if the PX4 mission still feels too jerky.
- for `--keyboard-control`, tune `--keyboard-acceleration-limit-mps2 4` and `--keyboard-yaw-acceleration-dps2 110` if key presses feel too sharp.
- yaw is kept stable by default for smoother motion; add `--face-travel-direction` if you want the drone nose to turn into each path leg.
- Short-path: start at "72,-8,-4" and goal at "33, -19, -6"
- Long-path: start at "72,-8,-4" and goal at "-50, 76, -25"

*Unresolved Problem: Encountered when working on Windows 11 using WSL2*: to rerun the sim, user has to restart Unreal Engine Editor and PX4 SITL. The script will not work if the sim is restarted without restarting PX4 SITL.
