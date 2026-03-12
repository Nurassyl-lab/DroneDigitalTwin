## Getting Started

### System requirements:
- Ubuntu 24.04 (I use WSL for now)
- NVIDIA GPU Drivers (verify by running `nvidia-smi`)
- Check if you have Vulkan libraries installed `dpkg -s libvulkan1 vulkan-tools`
    - if NOT installed run `sudo apt install libvulkan1 vulkan-tools`
    - Confirm that Vulkan libraries work by running `vulkaninfo`

*Since we want to have a full control over the simulation, we would develop with Project AirSim source*

- Follow [Build From Source as a Developer](https://github.com/Nurassyl-lab/DroneSimDev/blob/main/docs/development/use_source.md#build-from-source-as-a-developer) for Linux
- Caution: [Developing Project AirSim Sim Libs](https://github.com/Nurassyl-lab/DroneSimDev/blob/main/docs/development/use_source.md#developing-project-airsim-sim-libs)

### VSCode setup
*Note: You can develop within WSL using VSCode extension [Extension WSL](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl)*
1. `cd DroneSimDev`
2. `./unreal/Blocks/blocks_genprojfiles_vscode.sh`
