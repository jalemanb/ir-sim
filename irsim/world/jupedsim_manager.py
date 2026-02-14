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
                 persons_per_room: int = 2,
                 pedestrian_radius: float = 0.3,
                 pedestrian_speed: float = 1.2,
                 reach_distance: float = 0.5,
                 wall_margin: float = 0.5,
                 avoid_robots: bool = True,
                 seed: Optional[int] = None,
                 room_overrides: Optional[Dict[str, int]] = None):
        """Initialize JuPedSim manager.

        Args:
            map_attr: HouseExpo map attributes containing 'verts' and 'room_category'.
            step_time: Simulation time step in seconds.
            persons_per_room: Default number of pedestrians per room (best effort).
            pedestrian_radius: Radius of pedestrian agents in meters.
            pedestrian_speed: Desired walking speed in m/s.
            reach_distance: Distance threshold for goal reassignment.
            wall_margin: Minimum distance from walls in meters.
            avoid_robots: Whether pedestrians should avoid robots during movement.
            seed: Random seed for reproducibility.
            room_overrides: Dict mapping room type to custom person count.
        """
        if not JUPEDSIM_AVAILABLE:
            raise ImportError(
                "JuPedSim is not installed. Install with: pip install jupedsim"
            )

        self.logger = logging.getLogger(__name__)
        self.map_attr = map_attr
        self.step_time = step_time
        self.persons_per_room = persons_per_room
        self.pedestrian_radius = pedestrian_radius
        self.pedestrian_speed = pedestrian_speed
        self.reach_distance = reach_distance
        self.wall_margin = wall_margin
        self.avoid_robots = avoid_robots
        self.room_overrides = room_overrides or {}
        self.rng = np.random.default_rng(seed)

        # Robot tracking for avoidance
        self.robot_obstacles = []  # List of JuPedSim obstacle IDs

        # Extract building polygon and rooms
        self.building_polygon = self._load_building_polygon()
        self.rooms = self._extract_rooms()

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
        self.spawn_rooms: Dict[int, int] = {}  # agent_id -> room_index

        self.logger.info(
            f"JuPedSimManager initialized with {len(self.rooms)} rooms"
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

    def _extract_rooms(self) -> List[Tuple[str, List[float]]]:
        """Extract all rooms with their types and bounding boxes.

        Returns:
            List of tuples: (room_type, [xmin, ymin, xmax, ymax])
        """
        rooms = []
        for room_type, room_list in self.map_attr["room_category"].items():
            for room_bbox in room_list:
                rooms.append((room_type, room_bbox))
        return rooms

    def sample_point_in_room(self, room_bbox: List[float], margin: float = 0.25) -> Tuple[float, float]:
        """Sample a random point within a room bounding box.

        Args:
            room_bbox: [xmin, ymin, xmax, ymax]
            margin: Margin from walls in meters.

        Returns:
            (x, y) coordinates within the room.
        """
        xmin, ymin, xmax, ymax = room_bbox
        x = self.rng.uniform(xmin + margin, xmax - margin)
        y = self.rng.uniform(ymin + margin, ymax - margin)
        return (float(x), float(y))

    def sample_goal_in_bbox_area(self, margin: float = 0.5) -> np.ndarray:
        """Sample a random goal position in the bbox interaction area.

        Args:
            margin: Margin from bbox boundaries in meters.

        Returns:
            Goal position as numpy array [x, y].
        """
        # Use the bbox from map_attr (the interaction area)
        bbox = self.map_attr.get("bbox", None)
        if bbox is None:
            # Fallback to building polygon bounds
            bounds = self.building_polygon.bounds
            xmin, ymin, xmax, ymax = bounds
        else:
            xmin = bbox["min"][0] + margin
            ymin = bbox["min"][1] + margin
            xmax = bbox["max"][0] - margin
            ymax = bbox["max"][1] - margin

        # Sample random point in bbox
        x = self.rng.uniform(xmin, xmax)
        y = self.rng.uniform(ymin, ymax)
        return np.array([x, y], dtype=float)

    def spawn_pedestrians(self, exclude_positions: Optional[List[Tuple[float, float, float]]] = None) -> int:
        """Spawn pedestrians in rooms according to configuration.

        Uses JuPedSim's distribute_by_number per room to ensure proper spacing
        between pedestrians and from walls.

        Args:
            exclude_positions: List of (x, y, radius) tuples to avoid (e.g., robot positions).

        Returns:
            Number of pedestrians spawned.
        """
        if exclude_positions is None:
            exclude_positions = []

        total_spawned = 0

        for room_idx, (room_type, room_bbox) in enumerate(self.rooms):
            # Determine how many people for this room
            num_people = self.room_overrides.get(room_type, self.persons_per_room)

            msg = f"Room {room_idx} ({room_type}): Attempting to spawn {num_people} pedestrians"
            self.logger.info(msg)
            print(msg)  # Also print to console

            if num_people <= 0:
                self.logger.info(f"Room {room_idx} ({room_type}): Skipped (num_people=0)")
                continue

            # Create polygon for this room
            xmin, ymin, xmax, ymax = room_bbox
            room_box = box(xmin, ymin, xmax, ymax)

            # Ensure room polygon is valid and inside building
            room_polygon = room_box.intersection(self.building_polygon)

            # Handle different geometry types from intersection
            if room_polygon.is_empty:
                msg = f"Room {room_idx} ({room_type}): Skipped (outside building)"
                print(msg)
                continue

            if room_polygon.geom_type == "MultiPolygon":
                # Take the largest piece
                room_polygon = max(room_polygon.geoms, key=lambda g: g.area)

            if room_polygon.geom_type != "Polygon":
                # Skip non-polygon results (Point, LineString, GeometryCollection)
                msg = f"Room {room_idx} ({room_type}): Skipped (intersection is {room_polygon.geom_type})"
                print(msg)
                continue

            if room_polygon.area < 0.5:  # Reduced from 1.0 to 0.5
                msg = f"Room {room_idx} ({room_type}): Skipped (too small: {room_polygon.area:.2f}m²)"
                self.logger.info(msg)
                print(msg)
                continue

            # Try to spawn exact number of pedestrians (no partial spawning)
            try:
                # Use JuPedSim's distribute_by_number for proper spacing
                # Minimum distance between agents = 2 * radius
                # Distance from walls = wall_margin
                positions = distribute_by_number(
                    polygon=room_polygon,
                    number_of_agents=num_people,
                    distance_to_agents=max(2 * self.pedestrian_radius, 0.4),  # Min spacing
                    distance_to_polygon=max(self.wall_margin, 0.4),  # User-configurable wall clearance
                    seed=self.rng.integers(0, 2**31) if hasattr(self.rng, 'integers') else None,
                )
            except Exception as spawn_error:
                # Cannot place exact number - skip this room entirely
                msg = f"Room {room_idx} ({room_type}): ✗ Cannot place {num_people} pedestrians (room too small or constrained): {spawn_error}"
                self.logger.warning(msg)
                print(msg)
                continue  # Skip this room entirely

            # Validate all positions first (check for robot proximity)
            valid_positions = []
            for pos in positions:
                too_close = False
                for ex_x, ex_y, ex_radius in exclude_positions:
                    # Calculate proper exclusion distance: robot_radius + pedestrian_radius + safety margin
                    min_distance = ex_radius + self.pedestrian_radius + 0.3  # 0.3m safety margin
                    distance = np.linalg.norm(np.array(pos) - np.array([ex_x, ex_y]))
                    if distance < min_distance:
                        too_close = True
                        break

                if not too_close:
                    valid_positions.append(pos)

            # For exact spawning: only spawn if we can place ALL requested pedestrians
            if len(valid_positions) < num_people:
                msg = f"Room {room_idx} ({room_type}): ✗ Cannot spawn exact number {num_people} (only {len(valid_positions)} valid positions - {num_people - len(valid_positions)} too close to robot)"
                self.logger.warning(msg)
                print(msg)
                continue  # Skip this room entirely

            # Spawn all pedestrians in this room (we know all positions are valid)
            room_spawned = 0
            for pos in valid_positions:
                # Initial goal in bbox area
                goal_pos = self.sample_goal_in_bbox_area()

                # Create JuPedSim agent
                params = jps.CollisionFreeSpeedModelAgentParameters(
                    position=(float(pos[0]), float(pos[1])),
                    desired_speed=self.pedestrian_speed,
                    radius=self.pedestrian_radius,
                    journey_id=self.journey_id,
                    stage_id=self.direct_stage,
                )

                agent_id = self.sim.add_agent(params)
                self.agent_ids.append(agent_id)
                self.targets[agent_id] = goal_pos
                self.spawn_rooms[agent_id] = room_idx

                # Set initial target
                self.sim.agent(agent_id).target = (float(goal_pos[0]), float(goal_pos[1]))

                total_spawned += 1
                room_spawned += 1

            # Log success for this room
            msg = f"Room {room_idx} ({room_type}): ✓ Successfully spawned {room_spawned}/{num_people} pedestrians"
            self.logger.info(msg)
            print(msg)

        msg = f"\n{'='*60}\nSPAWNING COMPLETE: {total_spawned} pedestrians spawned across {len(self.rooms)} rooms\n{'='*60}"
        self.logger.info(msg)
        print(msg)
        return total_spawned

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
                # Assign new goal in bbox area
                new_goal = self.sample_goal_in_bbox_area()
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
