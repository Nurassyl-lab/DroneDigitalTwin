### Manual keyboard piloting
```python
python px4_astar_autopilot.py `
  --keyboard-control `
  --keyboard-flight-mode direct `
  --start "0,148,-2" `
  --start-as-scene-origin `
  --start-yaw-deg -90 `
  --live-ned-interval-sec 1.0 `
  --keyboard-speed-mps 360 `
  --keyboard-speed-step-mps 20 `
  --keyboard-yaw-rate-dps 60
```
*max speed might be unbounded*
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