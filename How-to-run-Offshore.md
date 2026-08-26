### Manual keyboard piloting
```python
python px4_astar_autopilot.py `
  --keyboard-control `
  --keyboard-flight-mode px4 `
  --start "0,148,-2" `
  --start-as-scene-origin `
  --start-yaw-deg -90 `
  --live-ned-interval-sec 1.0 `
  --keyboard-speed-mps 360 `
  --keyboard-speed-step-mps 20 `
  --keyboard-yaw-rate-dps 60 `
  --wind-speed-mps 15 `
  --wind-direction-deg 90
```
*max speed might be unbounded*

Notes:
- Use `--keyboard-flight-mode direct` to fly in direct mode, which is more responsive but less realistic.
- `--wind-vector "0,15,0"` you can also specify wind vector in NED coordinates instead of speed and direction.
- You can change in real time too, by pressing `G` and then `20,90,-3` will set wind to 20 m/s from 90 degrees (East). You can also use `G` and then `0,0,0` to turn off the wind. Where z is the vertical wind component, positive is upward.

Keyboard controls:
- W: Move forward
- S: Move backward
- A: Move left
- D: Move right
- Arrow Up: Move up
- Arrow Down: Move down
- Arrow Left: Yaw left
- Arrow Right: Yaw right
- K: increase speed
- L: decrease speed
- T: Teleport
    - After pressing it, type coordinates in the format "x,y,z" and press enter to teleport to that location
- Q: quit