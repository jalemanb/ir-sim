# JuPedSim Integration Guide for IR-SIM

This guide explains how to use JuPedSim pedestrian simulation with IR-SIM to create realistic human obstacles in your robot navigation scenarios.

## Overview

JuPedSim provides realistic pedestrian simulation with:
- **Collision-free movement** using the Collision Free Speed Model
- **Room-based spawning** - pedestrians spawn in different rooms
- **Dynamic goal assignment** - pedestrians move between rooms
- **Automatic retargeting** - new goals assigned when reached
- **Full integration** - pedestrians visible to robot sensors (LiDAR, etc.)

## Installation

### 1. Install JuPedSim

```bash
pip install jupedsim
```

### 2. Install Shapely (if not already installed)

```bash
pip install shapely
```

## Configuration

### Basic Configuration

Add the following to your YAML configuration file:

```yaml
jupedsim:
  enabled: true  # Enable JuPedSim
  persons_per_room: 2  # Number of pedestrians per room
  pedestrian_radius: 0.3  # Pedestrian radius in meters
  pedestrian_speed: 1.2  # Walking speed in m/s
  reach_distance: 0.5  # Goal reassignment threshold
  seed: 42  # Random seed (optional)
  color: "orange"  # Visualization color
```

### Required: HouseExpo Environment

JuPedSim integration requires a HouseExpo environment. When creating your environment:

```python
from irsim.env import EnvBase

env = EnvBase(
    world_name="your_config.yaml",
    house_expo_path="/path/to/HouseExpo",  # Required!
    house_expo_map_name="map_name",         # Required!
    display=True
)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Enable/disable JuPedSim |
| `persons_per_room` | int | `2` | Default pedestrians per room |
| `pedestrian_radius` | float | `0.3` | Radius in meters (typical: 0.25-0.35) |
| `pedestrian_speed` | float | `1.2` | Walking speed in m/s (typical: 1.0-1.5) |
| `reach_distance` | float | `0.5` | Distance threshold for goal reassignment |
| `seed` | int | `None` | Random seed for reproducibility |
| `color` | str | `"orange"` | Pedestrian color in visualization |
| `description` | str | `None` | Path to custom image file |
| `room_overrides` | dict | `None` | Per-room-type person counts |

### Room Overrides

Customize pedestrian counts for specific room types:

```yaml
jupedsim:
  enabled: true
  persons_per_room: 2  # Default
  room_overrides:
    living_room: 5  # More people in living room
    bedroom: 1      # Fewer in bedrooms
    bathroom: 0     # No one in bathrooms
    kitchen: 3      # Medium in kitchen
```

### Custom Visualization

#### Option 1: Use a Color

```yaml
jupedsim:
  color: "red"  # Any matplotlib color
```

#### Option 2: Use a Custom Image

1. Place your image file in: `ir-sim/irsim/world/description/`
2. Configure the description:

```yaml
jupedsim:
  description: "person_icon.png"  # Your custom image
  pedestrian_radius: 0.3  # Image will be scaled to this size
```

## Usage Examples

### Example 1: Basic Setup

```python
from irsim.env import EnvBase

# Create environment with JuPedSim
env = EnvBase(
    world_name="config_with_jupedsim.yaml",
    house_expo_path="/data/HouseExpo",
    house_expo_map_name="00001",
    display=True
)

# Run simulation
for i in range(1000):
    env.step()  # Pedestrians move automatically
    env.render()
```

### Example 2: With Robot Navigation

```python
from irsim.env import EnvBase

env = EnvBase(
    world_name="robot_with_pedestrians.yaml",
    house_expo_path="/data/HouseExpo",
    house_expo_map_name="00001",
    display=True
)

# Robot navigates while avoiding pedestrians
while not all([robot.arrive for robot in env.robot_list]):
    # Your navigation algorithm here
    action = your_planner.plan(env.robot_list[0].state)
    env.step(action)
    env.render()
```

### Example 3: Get Pedestrian States

```python
# Access pedestrian information
if env.jupedsim_manager is not None:
    # Get all pedestrian states
    ped_states = env.jupedsim_manager.get_agent_states()

    for agent_id, (position, velocity) in ped_states.items():
        print(f"Pedestrian {agent_id}: pos={position}, vel={velocity}")

    # Get number of pedestrians
    num_peds = env.jupedsim_manager.get_num_agents()
    print(f"Total pedestrians: {num_peds}")

# Access pedestrian objects directly
for ped in env.pedestrian_objects.values():
    print(f"Position: {ped.state[:2]}")
    print(f"Geometry: {ped.geometry}")
```

## How It Works

### Initialization Flow

1. **Load HouseExpo map** - Extract room boundaries and building polygon
2. **Create JuPedSim simulation** - Initialize with building geometry
3. **Spawn pedestrians** - Distribute across rooms based on configuration
4. **Assign goals** - Each pedestrian gets a goal in a different room
5. **Create IR-SIM objects** - `ObstaclePedestrian` objects for visualization/collision

### Simulation Loop

Each `env.step()`:

1. **Step JuPedSim** - Pedestrians move toward goals
2. **Check goals** - Reassign if pedestrian reached target
3. **Update IR-SIM objects** - Sync positions and orientations
4. **Collision detection** - Robots detect pedestrians via sensors
5. **Render** - Visualize pedestrians with robots

### Goal Assignment Strategy

- Pedestrians spawn in one room
- Initial goal: random position in a **different** room
- When goal reached: new goal in **another different** room
- Creates natural "walking between rooms" behavior

## Complete Configuration Example

```yaml
# config_with_pedestrians.yaml
world:
  step_time: 0.05  # 20Hz for smooth movement
  sample_time: 0.1
  collision_mode: 'stop'
  control_mode: 'auto'

jupedsim:
  enabled: true
  persons_per_room: 2
  pedestrian_radius: 0.3
  pedestrian_speed: 1.2
  reach_distance: 0.5
  color: "orange"
  seed: 42

  room_overrides:
    living_room: 4
    bedroom: 1

robot:
  - kinematics: {name: 'diff'}
    shape: {name: 'circle', radius: 0.3}
    state: [5, 5, 0]
    goal: [40, 40, 0]
    behavior: {name: 'rvo'}  # Good for avoiding pedestrians
    sensors:
      - type: 'lidar2d'
        range_max: 10
        number: 360
```

## Python Usage

```python
from irsim.env import EnvBase

# Initialize with HouseExpo
env = EnvBase(
    world_name="config_with_pedestrians.yaml",
    house_expo_path="/path/to/HouseExpo",
    house_expo_map_name="00001",  # or any HouseExpo map
    display=True,
    seed=42
)

# Simulation loop
for step in range(2000):
    # Robot control (optional)
    if not env.robot_list[0].arrive:
        env.step()  # Auto behavior or provide action
    else:
        break

    # Render every 10 steps
    if step % 10 == 0:
        env.render(interval=0.01)

env.end()
```

## Troubleshooting

### JuPedSim not found

```
ImportError: JuPedSim is not installed
```

**Solution**: Install JuPedSim
```bash
pip install jupedsim
```

### No map_attr warning

```
JuPedSim requires HouseExpo map data
```

**Solution**: Make sure you provide `house_expo_path` and `house_expo_map_name` when creating the environment:
```python
env = EnvBase(
    world_name="config.yaml",
    house_expo_path="/path/to/HouseExpo",  # Add this!
    house_expo_map_name="map_name"          # And this!
)
```

### Pedestrians not moving

**Check**:
1. `enabled: true` in YAML configuration
2. HouseExpo map is loaded correctly
3. `persons_per_room > 0` or room_overrides have non-zero values
4. Calling `env.step()` in your loop

### Image not showing

**Check**:
1. Image file exists in `ir-sim/irsim/world/description/`
2. File name matches the `description` parameter
3. Image format is supported (PNG, JPG, etc.)

## Performance Tips

1. **Reduce step_time**: Use `step_time: 0.05` or smaller for smoother pedestrian movement
2. **Limit pedestrians**: Too many pedestrians (>50) may slow down simulation
3. **Adjust reach_distance**: Larger values reduce goal reassignment frequency
4. **Disable visualization**: Set `display=False` for faster training

## Advanced: Accessing JuPedSim Directly

```python
# Access the underlying JuPedSim simulation
if env.jupedsim_manager is not None:
    sim = env.jupedsim_manager.sim

    # Get all agent IDs
    agent_ids = env.jupedsim_manager.agent_ids

    # Access individual agents
    for aid in agent_ids:
        agent = sim.agent(aid)
        print(f"Agent {aid}: pos={agent.position}, target={agent.target}")

    # Manually change a target
    new_target = (10.0, 15.0)
    sim.agent(agent_ids[0]).target = new_target
```

## Files Created

The integration adds these files to IR-SIM:

- `irsim/world/obstacles/obstacle_pedestrian.py` - Pedestrian object class
- `irsim/world/jupedsim_manager.py` - JuPedSim manager
- Modifications to:
  - `irsim/env/env_base.py` - Integration hooks
  - `irsim/world/world.py` - Store HouseExpo data
  - `irsim/world/obstacles/__init__.py` - Export pedestrian class

## License

This integration uses JuPedSim under its respective license. See [JuPedSim documentation](https://www.jupedsim.org/) for details.
