"""JuPedSim integration manager for ir-sim."""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

# JuPedSim imports - optional dependency
try:
    import jupedsim as jps
    from shapely.geometry import Point, Polygon, box
    from jupedsim.distributions import distribute_by_number
    JUPEDSIM_AVAILABLE = True
except ImportError:
    JUPEDSIM_AVAILABLE = False
    jps = None
    Point = None
    Polygon = None
    box = None
    distribute_by_number = None


class JuPedSimManager:
    """Manages JuPedSim pedestrian simulation integrated with ir-sim."""

    def __init__(self,
                 map_attr: dict,
                 step_time: float = 0.1,
                 number_of_agents: int = 10,
                 pedestrian_radius: float = 0.3,
                 pedestrian_speed: float = 1.2,
                 reach_distance: float = 0.5,
                 wall_margin: float = 0.5,
                 avoid_robots: bool = True,
                 seed: Optional[int] = None):
        """Initialize JuPedSim manager with grid-based spawning.

        Args:
            map_attr: HouseExpo map attributes containing 'verts' and bbox.
            step_time: Simulation time step in seconds.
            number_of_agents: Total number of pedestrian agents to spawn.
            pedestrian_radius: Radius of pedestrian agents in meters.
            pedestrian_speed: Desired walking speed in m/s.
            reach_distance: Distance threshold for goal reassignment.
            wall_margin: Minimum distance from walls in meters.
            avoid_robots: Whether pedestrians should avoid robots during movement.
            seed: Random seed for reproducibility.
        """
        if not JUPEDSIM_AVAILABLE:
            raise ImportError(
                "JuPedSim is not installed. Install with: pip install jupedsim"
            )

        self.logger = logging.getLogger(__name__)
        self.map_attr = map_attr
        self.step_time = step_time
        self.number_of_agents = number_of_agents
        self.pedestrian_radius = pedestrian_radius
        self.pedestrian_speed = pedestrian_speed
        self.reach_distance = reach_distance
        self.wall_margin = wall_margin
        self.avoid_robots = avoid_robots
        self.rng = np.random.default_rng(seed)

        # Robot tracking for avoidance
        self.robot_obstacles = []  # List of JuPedSim obstacle IDs

        # Extract building polygon and bbox
        self.building_polygon = self._load_building_polygon()
        self.bbox = self._get_bbox()

        # Grid will be computed during spawning based on total entities
        self.grid_cells = []  # List of (center_x, center_y) for each grid cell
        self.valid_grid_cells = []  # Grid cells inside building polygon

        # Initialize JuPedSim simulation
        self.sim = jps.Simulation(
            model=jps.CollisionFreeSpeedModel(),
            geometry=self.building_polygon,
            dt=step_time,
        )

        # Setup direct steering
        self.direct_stage = self.sim.add_direct_steering_stage()
        self.journey_id = self.sim.add_journey(
            jps.JourneyDescription([self.direct_stage])
        )

        # Track agents and their targets
        self.agent_ids: List[int] = []
        self.targets: Dict[int, np.ndarray] = {}

        self.logger.info(
            f"JuPedSimManager initialized for grid-based spawning with {number_of_agents} agents"
        )

    def _load_building_polygon(self) -> Polygon:
        """Load building boundary polygon from HouseExpo verts."""
        verts = np.asarray(self.map_attr["verts"], dtype=float)

        # Handle case where first vertex was re-added to close the polygon
        # (Shapely automatically closes polygons, so we need to remove the duplicate)
        if len(verts) > 1 and np.allclose(verts[0], verts[-1]):
            verts = verts[:-1]  # Remove duplicate last vertex

        poly = Polygon(verts)

        # Fix occasional self-intersections
        if not poly.is_valid:
            poly = poly.buffer(0)

        # If cleanup returns MultiPolygon, keep largest piece
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)

        if poly.geom_type != "Polygon":
            raise ValueError(f"Expected Polygon, got {poly.geom_type}")

        return poly

    def _get_bbox(self) -> Dict[str, List[float]]:
        """Get bounding box from map_attr or compute from building polygon.

        Returns:
            Dict with 'min' and 'max' keys containing [x, y] coordinates.
        """
        if "bbox" in self.map_attr:
            return self.map_attr["bbox"]
        else:
            # Compute from building polygon bounds
            bounds = self.building_polygon.bounds
            xmin, ymin, xmax, ymax = bounds
            return {"min": [xmin, ymin], "max": [xmax, ymax]}

    def _create_grid(self, num_robots: int):
        """Create grid based on total number of entities (agents + robots).

        Args:
            num_robots: Number of robots in the environment.
        """
        total_entities = self.number_of_agents + num_robots

        # Compute grid dimensions: create enough cells for all entities with some buffer
        # Use 1.5x factor to ensure we have more cells than entities
        grid_dim = int(np.ceil(np.sqrt(total_entities * 1.5)))

        # Get bbox coordinates
        xmin, ymin = self.bbox["min"]
        xmax, ymax = self.bbox["max"]

        # Compute cell size
        cell_width = (xmax - xmin) / grid_dim
        cell_height = (ymax - ymin) / grid_dim

        # Create grid cell centers
        self.grid_cells = []
        self.valid_grid_cells = []

        for i in range(grid_dim):
            for j in range(grid_dim):
                # Calculate cell center
                center_x = xmin + (i + 0.5) * cell_width
                center_y = ymin + (j + 0.5) * cell_height

                cell_center = (center_x, center_y)
                self.grid_cells.append(cell_center)

                # Check if center is inside building polygon
                if self.building_polygon.contains(Point(center_x, center_y)):
                    self.valid_grid_cells.append(cell_center)

        msg = f"Grid created: {grid_dim}x{grid_dim} = {len(self.grid_cells)} cells, {len(self.valid_grid_cells)} valid cells inside polygon"
        self.logger.info(msg)
        print(msg)

        # Store grid parameters for spawn validation
        self.grid_dim = grid_dim
        self.cell_width = cell_width
        self.cell_height = cell_height

    def _sample_point_in_grid_cell(self, grid_idx: int, max_attempts: int = 50) -> Optional[Tuple[float, float]]:
        """Sample a valid point within a grid cell that's inside the building polygon.

        Args:
            grid_idx: Index of the grid cell in self.grid_cells.
            max_attempts: Maximum number of sampling attempts.

        Returns:
            (x, y) coordinates within the grid cell and inside polygon, or None if failed.
        """
        center_x, center_y = self.grid_cells[grid_idx]

        # First try the center
        if self.building_polygon.contains(Point(center_x, center_y)):
            return (center_x, center_y)

        # If center is outside, sample random points within the cell
        xmin = self.bbox["min"][0] + (grid_idx % self.grid_dim) * self.cell_width
        ymin = self.bbox["min"][1] + (grid_idx // self.grid_dim) * self.cell_height
        xmax = xmin + self.cell_width
        ymax = ymin + self.cell_height

        for _ in range(max_attempts):
            x = self.rng.uniform(xmin, xmax)
            y = self.rng.uniform(ymin, ymax)

            if self.building_polygon.contains(Point(x, y)):
                return (float(x), float(y))

        return None  # Failed to find valid point in this cell

    def sample_goal_from_grid(self) -> np.ndarray:
        """Sample a random goal from valid grid cell centers.

        Returns:
            Goal position as numpy array [x, y] (center of a random valid grid cell).
        """
        if not self.valid_grid_cells:
            # Fallback to random point if no valid cells
            bounds = self.building_polygon.bounds
            xmin, ymin, xmax, ymax = bounds
            x = self.rng.uniform(xmin, xmax)
            y = self.rng.uniform(ymin, ymax)
            return np.array([x, y], dtype=float)

        # Randomly select a valid grid cell center
        center_x, center_y = self.rng.choice(self.valid_grid_cells)
        return np.array([center_x, center_y], dtype=float)

    def get_robot_spawn_positions(self, num_robots: int) -> Tuple[List[Tuple[float, float, float]], List[int]]:
        """Get grid-based spawn positions for robots.

        Args:
            num_robots: Number of robots to spawn.

        Returns:
            Tuple of (robot_positions, used_indices):
                - robot_positions: List of (x, y, yaw) tuples for robot spawn positions
                - used_indices: List of grid cell indices used by robots
        """
        # Create grid first (this also stores it for pedestrian spawning)
        self._create_grid(num_robots)

        if len(self.valid_grid_cells) == 0:
            msg = "ERROR: No valid grid cells for robot spawning!"
            self.logger.error(msg)
            print(msg)
            return [], []

        # Select random valid grid cells for robots (without replacement)
        num_to_spawn = min(num_robots, len(self.valid_grid_cells))
        selected_indices = self.rng.choice(
            len(self.valid_grid_cells),
            size=num_to_spawn,
            replace=False
        )

        robot_positions = []
        for idx in selected_indices:
            center_x, center_y = self.valid_grid_cells[idx]
            yaw = self.rng.uniform(-np.pi, np.pi)  # Random orientation
            robot_positions.append((float(center_x), float(center_y), float(yaw)))

        msg = f"Grid-based robot spawning: {len(robot_positions)} positions generated"
        self.logger.info(msg)
        print(msg)

        return robot_positions, selected_indices.tolist()

    def spawn_pedestrians(self, num_robots: int = 1, robot_grid_indices: Optional[List[int]] = None) -> int:
        """Spawn pedestrians using grid-based spawning.

        Creates a grid over the bbox area and randomly assigns pedestrians to grid cells.
        Spawns pedestrians at grid cell centers (or samples within cell if center is outside polygon).

        Args:
            num_robots: Number of robots (used to compute grid size if not already created).
            robot_grid_indices: Optional list of grid cell indices already used by robots to avoid.

        Returns:
            Number of pedestrians spawned.
        """
        # Create grid if not already created (happens if get_robot_spawn_positions was not called)
        if not hasattr(self, 'grid_dim') or self.grid_dim is None:
            self._create_grid(num_robots)

        if len(self.valid_grid_cells) == 0:
            msg = "ERROR: No valid grid cells found inside building polygon!"
            self.logger.error(msg)
            print(msg)
            return 0

        # Get available cells (excluding those used by robots)
        if robot_grid_indices:
            available_indices = [i for i in range(len(self.valid_grid_cells)) if i not in robot_grid_indices]
        else:
            available_indices = list(range(len(self.valid_grid_cells)))

        # Check if we have enough valid cells
        if len(available_indices) < self.number_of_agents:
            msg = f"WARNING: Only {len(available_indices)} available cells for {self.number_of_agents} agents. Some agents may not spawn."
            self.logger.warning(msg)
            print(msg)

        # Randomly select grid cells for pedestrians (without replacement)
        num_to_spawn = min(self.number_of_agents, len(available_indices))
        selected_cells = self.rng.choice(
            available_indices,
            size=num_to_spawn,
            replace=False
        )

        spawned = 0
        failed = 0

        for idx in selected_cells:
            center_x, center_y = self.valid_grid_cells[idx]

            # Spawn at grid cell center (already validated as inside polygon)
            position = (center_x, center_y)

            # Assign random goal from grid
            goal_pos = self.sample_goal_from_grid()

            # Create JuPedSim agent
            try:
                params = jps.CollisionFreeSpeedModelAgentParameters(
                    position=position,
                    desired_speed=self.pedestrian_speed,
                    radius=self.pedestrian_radius,
                    journey_id=self.journey_id,
                    stage_id=self.direct_stage,
                )

                agent_id = self.sim.add_agent(params)
                self.agent_ids.append(agent_id)
                self.targets[agent_id] = goal_pos

                # Set initial target
                self.sim.agent(agent_id).target = (float(goal_pos[0]), float(goal_pos[1]))

                spawned += 1

            except Exception as e:
                failed += 1
                self.logger.warning(f"Failed to spawn agent at {position}: {e}")

        msg = f"\n{'='*60}\nGRID-BASED SPAWNING COMPLETE\n"
        msg += f"Grid: {self.grid_dim}x{self.grid_dim} = {len(self.grid_cells)} total cells\n"
        msg += f"Valid cells (inside polygon): {len(self.valid_grid_cells)}\n"
        msg += f"Pedestrians spawned: {spawned}/{self.number_of_agents}\n"
        if failed > 0:
            msg += f"Failed spawns: {failed}\n"
        msg += f"{'='*60}"
        self.logger.info(msg)
        print(msg)

        return spawned

    def step(self):
        """Advance JuPedSim simulation by one step and update targets."""
        # Step simulation
        self.sim.iterate(1)

        # Check and update targets for agents that reached their goals
        for agent_id in self.agent_ids:
            agent = self.sim.agent(agent_id)
            pos = np.array(agent.position, dtype=float)

            # Check if agent reached target
            if np.linalg.norm(pos - self.targets[agent_id]) < self.reach_distance:
                # Assign new goal from grid
                new_goal = self.sample_goal_from_grid()
                self.targets[agent_id] = new_goal
                agent.target = (float(new_goal[0]), float(new_goal[1]))

    def get_agent_states(self) -> Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Get current states of all pedestrian agents.

        Returns:
            Dict mapping agent_id to ((x, y), (vx, vy))
        """
        states = {}
        for agent_id in self.agent_ids:
            agent = self.sim.agent(agent_id)
            pos = agent.position

            # Get velocity - JuPedSim agents have orientation property
            try:
                # JuPedSim provides orientation as a tuple (cos, sin) representing direction
                vel = agent.orientation if hasattr(agent, 'orientation') else (0.0, 0.0)
                # Scale by speed to get actual velocity vector
                if hasattr(agent, 'desired_speed'):
                    speed = agent.desired_speed
                    vel = (vel[0] * speed, vel[1] * speed)
            except:
                vel = (0.0, 0.0)

            states[agent_id] = (pos, vel)
        return states

    def update_robot_obstacles(self, robot_positions: List[Tuple[float, float, float]]):
        """Update robot positions as obstacles in JuPedSim.

        Args:
            robot_positions: List of (x, y, radius) for each robot.
        """
        if not self.avoid_robots:
            return

        # Remove old robot obstacles
        for obs_id in self.robot_obstacles:
            try:
                self.sim.remove_obstacle(obs_id)
            except:
                pass  # Obstacle might not exist
        self.robot_obstacles.clear()

        # Add updated robot obstacles as circles
        for x, y, radius in robot_positions:
            try:
                # Add robot as circular obstacle
                obs_id = self.sim.add_obstacle(
                    jps.CircularObstacle((float(x), float(y)), float(radius + 0.2))  # Add 20cm safety margin
                )
                self.robot_obstacles.append(obs_id)
            except Exception as e:
                self.logger.debug(f"Could not add robot obstacle at ({x}, {y}): {e}")

    def get_num_agents(self) -> int:
        """Get total number of active pedestrian agents."""
        return len(self.agent_ids)
