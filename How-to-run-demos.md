## How to run route_replan_static.py demo

1. Open the River-side ForestDomeEnv.uproject in Unreal Engine Editor and press play button.
2. After the scene is loaded, run PX4 using `make px4_sitl_default none_iris`. If the demo waits on TCP port 4560, restart PX4 after Unreal is already playing.
3. Run 

```python
python route_replan_static.py `
  --route "72,-8,-4; 69,-10,-40; 66,-11,-40; 63,-11,-40; 60,-12,-40; 57,-12,-40; 54,-12,-40; 51,-12,-40; 48,-12,-40; 45,-13,-40; 42,-13,-40; 39,-16,-40; 36,-17,-40; 33,-19,-6" `
  --start "72,-8,-4" `
  --start-as-scene-origin `
  --px4-ready-timeout-sec 300 `
  --flight-driver velocity `
  --velocity-lookahead-m 8 `
  --path-yaw-rate-dps 10 `
  --path-yaw-deadband-deg 5 `
  --path-yaw-response-sec 1.5
```

```python
python route_replan_static.py `
  --route "72.0,-8.0,-4.0; 69.93,-5.19,-5.36; 66.98,1.81,-4.61; 64.38,6.0,-4.00; 61.0,10.0,-4.5; 58.0,14.0,-5.0; 55.0,-18.0,-5.5;45.0,17.0,-16.0;28.0,25.0,-24.0;13.0,39.0,-24.0;-2.0,54.0,-27.0;-17.0,69.0,-27.0;-38.0,72.0,-28.0;-50.0,76.0,-25.0" `
  --start "72,-8.0,-4.0" `
  --start-as-scene-origin `
  --px4-ready-timeout-sec 300 `
  --object-stop-distance-m 2.0 `
  --dynamic-replan-lookahead-waypoints 1 `
  --replan-rejoin-waypoints-ahead 3 `
  --dynamic-replan-max-segment-m 0 `
  --dynamic-replan-stop-hold-sec 1.0 `
  --dynamic-replan-check-sec 0.5 `
  --replan-rejoin-point "45.0,17.0,-16.0" `
  --replan-emergency-node "62.18,-0.41,-5.38" `
  --velocity-command-duration-sec 0.1 `
  --acceleration-limit-mps2 1.0 `
  --velocity-lookahead-m 8 `
  --path-yaw-rate-dps 10 `
  --path-yaw-deadband-deg 5 `
  --path-yaw-response-sec 1.5 `
  --waypoint-acceptance-m 2 `
  --flight-driver velocity
```

Use `--flight-driver path-api` only for experiments without `--start-as-scene-origin`; with scene-NED routes, the velocity driver keeps the route frame consistent.

## Test paths
- short without any obstacles: `72,-8,-4; 69,-10,-40; 66,-11,-40; 63,-11,-40; 60,-12,-40; 57,-12,-40; 54,-12,-40; 51,-12,-40; 48,-12,-40; 45,-13,-40; 42,-13,-40; 39,-16,-40; 36,-17,-40; 33,-19,-6`
- long without any obstacles: `72.0, -8.0, -4.0;69.0, -6.0, -4.0;66.0, -3.0, -5.0;64.0, -1.0, -7.0;62.0, 1.0, -9.0;59.0, 2.0, -10.0;56.0, 2.0, -10.0;53.0, 2.0, -10.0;50.0, 3.0, -11.0;48.0, 5.0, -13.0;45.0, 8.0, -13.0;42.0, 11.0, -13.0;39.0, 14.0, -13.0;36.0, 17.0, -13.0;34.0, 19.0, -15.0;32.0, 21.0, -17.0;30.0, 23.0, -19.0;28.0, 25.0, -21.0;25.0, 27.0, -21.0;22.0, 30.0, -21.0;19.0, 33.0, -21.0;16.0, 36.0, -21.0;13.0, 39.0, -21.0;10.0, 42.0, -21.0;7.0, 45.0, -21.0;4.0, 48.0, -21.0;1.0, 51.0, -21.0;-2.0, 54.0, -21.0;-5.0, 57.0, -21.0;-8.0, 60.0, -21.0;-11.0, 63.0, -21.0;-14.0, 66.0, -21.0;-17.0, 69.0, -21.0;-20.0, 71.0, -21.0;-23.0, 71.0, -21.0;-26.0, 72.0, -22.0;-29.0, 72.0, -22.0;-32.0, 72.0, -22.0;-35.0, 72.0, -22.0;-38.0, 72.0, -22.0;-41.0, 72.0, -22.0;-43.0, 74.0, -24.0;-46.0, 75.0, -25.0;-50.0, 76.0, -25.0`
- Long path with obstacle:
```python
Approach a tree: 72.0,-8.0,-4.0; 69.93,-5.19,-5.36; 66.98,1.81,-4.61; 64.38,6.0,-4.00; 
Path through the tree: 61.0, 10.0, -4.5; 58.0, 14.0, -5.0; 55.0, -18.0, -5.5;
Continue: 45.0, 17.0, -13.0; 36.0, 17.0, -13.0;34.0, 19.0, -15.0;32.0, 21.0, -17.0;30.0, 23.0, -19.0;28.0, 25.0, -21.0;25.0, 27.0, -21.0;22.0, 30.0, -21.0;19.0, 33.0, -21.0;16.0, 36.0, -21.0;13.0, 39.0, -21.0;10.0, 42.0, -21.0;7.0, 45.0, -21.0;4.0, 48.0, -21.0;1.0, 51.0, -21.0;-2.0, 54.0, -21.0;-5.0, 57.0, -21.0;-8.0, 60.0, -21.0;-11.0, 63.0, -21.0;-14.0, 66.0, -21.0;-17.0, 69.0, -21.0;-20.0, 71.0, -21.0;-23.0, 71.0, -21.0;-26.0, 72.0, -22.0;-29.0, 72.0, -22.0;-32.0, 72.0, -22.0;-35.0, 72.0, -22.0;-38.0, 72.0, -22.0;-41.0, 72.0, -22.0;-43.0, 74.0, -24.0;-46.0, 75.0, -25.0;-50.0, 76.0, -25.
```
