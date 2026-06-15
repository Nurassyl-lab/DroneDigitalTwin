# How to run stuff

## How to run the simulation with autonavigation using PX4 SITL

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

