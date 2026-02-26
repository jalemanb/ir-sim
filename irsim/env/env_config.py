from __future__ import annotations

from operator import attrgetter
from typing import TYPE_CHECKING, Any, Optional

import yaml, json
import os

from irsim.util.util import file_check
from irsim.util.random import rng as global_rng
from irsim.world import World
from irsim.world.object_factory import ObjectFactory
from irsim.world.object_group import ObjectGroup

from .env_plot import EnvPlot

if TYPE_CHECKING:
    from irsim.config.env_param import EnvParam
    from irsim.config.world_param import WorldParam

from typing import List, Sequence, Tuple, Optional
import numpy as np

Box = Tuple[float, float, float, float]  # xmin, ymin, xmax, ymax

def box_center(box: Box) -> Tuple[float, float]:
    xmin, ymin, xmax, ymax = box
    return ((xmin + xmax) * 0.5, (ymin + ymax) * 0.5)


def _point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test (works for concave polygons)."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i, 0], poly[i, 1]
        xj, yj = poly[j, 0], poly[j, 1]
        if ((yi > y) != (yj > y)) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _segment_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Shortest distance from point P=(px,py) to segment A-B."""
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    if len2 == 0.0:
        return float(np.hypot(px - ax, py - ay))
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len2))
    return float(np.hypot(px - (ax + t * dx), py - (ay + t * dy)))


def _min_dist_to_boundary(x: float, y: float, poly: np.ndarray) -> float:
    """Minimum distance from (x, y) to any edge of the polygon."""
    n = len(poly)
    return min(
        _segment_dist(x, y, poly[i, 0], poly[i, 1], poly[(i + 1) % n, 0], poly[(i + 1) % n, 1])
        for i in range(n)
    )


def _robot_corners(
    x: float, y: float, yaw: float, half_l: float, half_w: float
) -> list:
    """Return the 4 corners of a robot rectangle oriented at *yaw*.

    Args:
        x, y:   Robot centre position.
        yaw:    Robot heading (radians).
        half_l: Half-length along the forward axis.
        half_w: Half-width along the lateral axis.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    fwd = np.array([c, s])    # forward unit vector
    lat = np.array([-s, c])   # left unit vector
    ctr = np.array([x, y])
    return [
        ctr + half_l * fwd + half_w * lat,
        ctr + half_l * fwd - half_w * lat,
        ctr - half_l * fwd + half_w * lat,
        ctr - half_l * fwd - half_w * lat,
    ]


def _sample_pose_in_polygon(
    rng: np.random.Generator,
    verts: list,
    bbox: dict,
    robot_half_l: float = 0.0,
    robot_half_w: float = 0.0,
    margin: float = 0.05,
    max_tries: int = 10000,
) -> Tuple[float, float, float]:
    """Rejection-sample a pose (x, y, yaw) so the robot fits inside the polygon.

    For rectangular robots the oriented bounding box (all 4 corners) is
    checked — every corner must be strictly inside the polygon and at least
    *margin* metres from the nearest wall.  Yaw is sampled inside the loop so
    that the orientation is also varied during rejection.

    Args:
        rng:          numpy random Generator.
        verts:        List of [x, y] vertex pairs (open or closed polygon).
        bbox:         Dict with ``'min'`` and ``'max'`` keys, each ``[x, y]``.
        robot_half_l: Half-length of the robot along its forward axis (metres).
        robot_half_w: Half-width of the robot along its lateral axis (metres).
        margin:       Extra clearance beyond the robot body from polygon walls.
        max_tries:    Attempts before raising ``RuntimeError``.

    Returns:
        ``(x, y, yaw)`` with *yaw* uniform in ``[-pi, pi]``.
    """
    poly = np.asarray(verts, dtype=float)
    # Drop duplicate closing vertex if present (handles both open and closed inputs)
    if len(poly) > 1 and np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError(f"verts must be [[x, y], ...] pairs, got shape {poly.shape}")

    xmin, ymin = float(bbox["min"][0]), float(bbox["min"][1])
    xmax, ymax = float(bbox["max"][0]), float(bbox["max"][1])

    use_corners = robot_half_l > 0.0 and robot_half_w > 0.0

    for _ in range(max_tries):
        x = float(rng.uniform(xmin, xmax))
        y = float(rng.uniform(ymin, ymax))
        yaw = float(rng.uniform(-np.pi, np.pi))

        if use_corners:
            corners = _robot_corners(x, y, yaw, robot_half_l, robot_half_w)
            if all(
                _point_in_polygon(float(cx), float(cy), poly)
                and _min_dist_to_boundary(float(cx), float(cy), poly) >= margin
                for cx, cy in corners
            ):
                return x, y, yaw
        else:
            # Fallback: treat as a point with a margin
            if _point_in_polygon(x, y, poly) and _min_dist_to_boundary(x, y, poly) >= margin:
                return x, y, yaw

    raise RuntimeError(
        f"Could not find a valid spawn pose after {max_tries} tries "
        f"(bbox={bbox}, half_l={robot_half_l}, half_w={robot_half_w}, margin={margin}). "
        "The polygon may be too small or margins too large."
    )

class EnvConfig:
    """
    Environment configuration loader and builder from YAML.

    Responsibilities:
        - Resolve and parse a YAML configuration into structured dictionaries
          (basic categories: ``world``, ``gui``, ``robot``, ``obstacle``)
        - Construct ``World`` and object collections and produce an ``EnvPlot``
        - Support reloading the YAML and updating the scene in the same figure
    """

    def __init__(
        self,
        world_name: Optional[str],
        env_param_instance: Optional[EnvParam] = None,
        world_param_instance: Optional[WorldParam] = None,
        house_expo_path: Optional[str] = None,
        house_expo_map_name: Optional[str] = None,
    ) -> None:
        self.object_factory = ObjectFactory()
        self._env_param = env_param_instance
        self._world_param = world_param_instance
        # Constructor args are defaults; yaml world section can override them
        self.house_expo_path = house_expo_path
        self.house_expo_map_name = house_expo_map_name
        self.seed = None  # populated by load_yaml if world.seed is present

        self.load_yaml(world_name)

    def load_yaml(self, world_name: Optional[str] = None) -> None:
        """Parse the YAML file and populate internal configuration state.

        Args:
            world_name: Path or name of the YAML file. If ``None``, will try to
                resolve via ``file_check`` and fall back to empty/defaults.

        """

        self.world_name = world_name
        self.world_file_path = file_check(world_name)

        self._kwargs_parse: dict[str, Any] = {
            "world": {},
            "gui": {},
            "robot": None,
            "obstacle": None,
            "jupedsim": {},  # JuPedSim pedestrian simulation config
        }

        self.world_file_path = self.world_file_path
        self.world_name = world_name

        if self.world_file_path is not None:
            with open(self.world_file_path) as file:
                com_list = yaml.load(file, Loader=yaml.FullLoader)

                for key in com_list:
                    if key in self._kwargs_parse:
                        self._kwargs_parse[key] = com_list[key]
                    else:
                        self.logger.error(
                            f"There are invalid key: '{key}' in {self.world_name} file!"
                        )
                        raise KeyError

            # Extract world-level meta-params so env_base can apply them
            # before initialize_objects() is called.
            world_raw = self._kwargs_parse.get("world") or {}
            if "seed" in world_raw:
                self.seed = world_raw["seed"]
            if "house_expo_path" in world_raw:
                self.house_expo_path = world_raw["house_expo_path"]
            if "house_expo_map_name" in world_raw:
                raw = world_raw["house_expo_map_name"]
                # Strip .json extension if the user included it
                if isinstance(raw, str) and raw.endswith(".json"):
                    raw = raw[:-5]
                self.house_expo_map_name = raw

        else:
            self.logger.error(
                f"{self.world_name} YAML File not found!, using default world config as alternative."
            )
    
    def _apply_house_expo(self, world_kwargs: dict) -> None:
        """Load a HouseExpo JSON map and place the robot inside the room.

        Mutates *world_kwargs* (adds ``map_attr``, ``width``, ``height``) and
        ``self.parse["obstacle"]`` / ``self.parse["robot"]`` in-place.

        If ``self.house_expo_map_name`` is ``None`` a map is chosen randomly
        from ``{house_expo_path}/json/`` using the global seeded RNG so that
        the choice is deterministic when a seed is set and fully random otherwise.
        """
        world_kwargs.pop("obstacle_map", None)

        map_name = self.house_expo_map_name
        if map_name is None:
            # Pick a random map from the json directory
            json_dir = os.path.join(self.house_expo_path, "json")
            available = sorted(
                f[:-5] for f in os.listdir(json_dir) if f.endswith(".json")
            )
            if not available:
                raise RuntimeError(f"No JSON map files found in {json_dir}")
            idx = int(global_rng.integers(0, len(available)))
            map_name = available[idx]
            print(f"[HouseExpo] Randomly selected map: {map_name}")

        map_path = os.path.join(self.house_expo_path, "json", map_name + ".json")
        with open(map_path, "r", encoding="utf-8") as f:
            world_kwargs["map_attr"] = json.load(f)

        bbox = world_kwargs["map_attr"]["bbox"]
        world_kwargs["width"]  = bbox["min"][0] + bbox["max"][0]
        world_kwargs["height"] = bbox["min"][1] + bbox["max"][1]

        print(f"[HouseExpo] map={map_name}  bbox={bbox}")
        print(f"[HouseExpo] width={world_kwargs['width']:.2f}  height={world_kwargs['height']:.2f}")

        if not isinstance(self.parse.get("obstacle"), list):
            self.parse["obstacle"] = []

        # Extract robot shape for corner-based collision checking
        robot_half_l, robot_half_w = 0.0, 0.0
        if self.parse.get("robot"):
            shape_cfg = self.parse["robot"][0].get("shape") or {}
            shape_name = shape_cfg.get("name", "circle")
            if shape_name == "rectangle":
                robot_half_l = shape_cfg.get("length", 0.5) / 2.0
                robot_half_w = shape_cfg.get("width",  0.3) / 2.0
            else:
                r = shape_cfg.get("radius", 0.25)
                robot_half_l = r
                robot_half_w = r

        verts = world_kwargs["map_attr"]["verts"]
        x, y, yaw = _sample_pose_in_polygon(
            global_rng, verts, bbox,
            robot_half_l=robot_half_l,
            robot_half_w=robot_half_w,
            margin=0.05,
        )
        if self.parse.get("robot"):
            self.parse["robot"][0]["state"] = [x, y, yaw]

        # Closed linestring obstacle representing room walls
        closed_verts = list(verts) + [verts[0]]
        self.parse["obstacle"].append({
            "shape": {"name": "linestring", "vertices": closed_verts},
            "state": [0.0, 0.0, 0.0],
            "unobstructed": False,
            "color": "royalblue",
        })

    def initialize_objects(self) -> Any:
        """Construct world, objects and plot from the current parsed config.

        Returns:
            Tuple: ``(world, objects, env_plot, robot_collection, obstacle_collection, map_collection)``

        Notes:
            - Caches the created ``EnvPlot`` and ``objects`` internally for use
              during in-place reloads.
        """

        world_kwargs = dict(self.parse["world"])  # copy
        # Remove meta-params that belong to EnvConfig, not World
        for _key in ("seed", "house_expo_path", "house_expo_map_name"):
            world_kwargs.pop(_key, None)

        if not isinstance(self.parse.get("obstacle"), list):
            self.parse["obstacle"] = []

        if not isinstance(self.parse.get("robot"), list):
            self.parse["robot"] = []

        if self.house_expo_path is not None:
            self._apply_house_expo(world_kwargs)


        world = World(
            self.world_name,
            world_param_instance=self._world_param,
            **world_kwargs,
        )

        robot_collection = self.object_factory.create_from_parse(
            self.parse["robot"], "robot"
        )

        obstacle_collection = self.object_factory.create_from_parse(
            self.parse["obstacle"],
            "obstacle",
            group_start_index=(
                max((obj.group for obj in robot_collection), default=-1) + 1
            ),
        )
        map_collection = self.object_factory.create_from_map(
            world.obstacle_positions, world.buffer_reso
        )

        objects = robot_collection + obstacle_collection + map_collection

        objects.sort(key=attrgetter("id"))

        # Initialize groups (unique and inclusive)
        group_ids = sorted({obj.group for obj in objects})
        object_groups = [
            ObjectGroup([obj for obj in objects if obj.group == gid], gid)
            for gid in group_ids
        ]

        env_plot = EnvPlot(world, objects)

        # cache for in-place reload
        self._env_plot = env_plot
        self._objects = objects

        return (
            world,
            objects,
            self._env_plot,
            robot_collection,
            obstacle_collection,
            map_collection,
            object_groups,
        )

    def reload_objects(self) -> Any:
        """Rebuild world/objects and update the current figure in-place.

        This method reuses the existing ``EnvPlot`` instance and its figure/axes,
        clearing old artists and re-initializing with the new world and objects.

        Returns:
            Tuple: ``(world, objects, env_plot, robot_collection, obstacle_collection, map_collection)``
        """

        world_kwargs = dict(self.parse["world"])  # copy
        # Remove meta-params that belong to EnvConfig, not World
        for _key in ("seed", "house_expo_path", "house_expo_map_name"):
            world_kwargs.pop(_key, None)

        if not isinstance(self.parse.get("obstacle"), list):
            self.parse["obstacle"] = []

        if not isinstance(self.parse.get("robot"), list):
            self.parse["robot"] = []

        if self.house_expo_path is not None:
            self._apply_house_expo(world_kwargs)

        world = World(
            self.world_name,
            world_param_instance=self._world_param,
            **world_kwargs,
        )

        robot_collection = self.object_factory.create_from_parse(
            self.parse["robot"], "robot"
        )
        obstacle_collection = self.object_factory.create_from_parse(
            self.parse["obstacle"],
            "obstacle",
            group_start_index=(
                max((obj.group for obj in robot_collection), default=-1) + 1
            ),
        )
        map_collection = self.object_factory.create_from_map(
            world.obstacle_positions, world.buffer_reso
        )

        objects = robot_collection + obstacle_collection + map_collection
        objects.sort(key=attrgetter("id"))

        group_ids = sorted({obj.group for obj in objects})
        object_groups = [
            ObjectGroup([obj for obj in objects if obj.group == gid], gid)
            for gid in group_ids
        ]

        # env_plot = EnvPlot(world, objects, **world.plot_parse)
        self._env_plot.clear_components("all", self._objects)
        self._env_plot._init_plot(world, objects)

        return (
            world,
            objects,
            self._env_plot,
            robot_collection,
            obstacle_collection,
            map_collection,
            object_groups,
        )

    def reload_yaml_objects(self, world_name) -> Any:
        """Reload YAML and update the scene using the existing figure.

        This re-parses the YAML and then calls :py:meth:`reload_objects` to
        apply the new configuration without creating a new figure window.

        Args:
            world_name: Optional path/name of the YAML to reload. If ``None``,
                uses the previously resolved YAML file.

        Returns:
            Tuple: ``(world, objects, env_plot, robot_collection, obstacle_collection, map_collection)``
        """

        reload_world_name = world_name if world_name is not None else self.world_name
        self.load_yaml(reload_world_name)
        (
            world,
            objects,
            env_plot,
            robot_collection,
            obstacle_collection,
            map_collection,
            object_groups,
        ) = self.reload_objects()

        return (
            world,
            objects,
            env_plot,
            robot_collection,
            obstacle_collection,
            map_collection,
            object_groups,
        )

    @property
    def parse(self) -> dict[str, Any]:
        """
        The parsed kwargs from the yaml file.
        """
        return self._kwargs_parse

    @property
    def logger(self):
        """
        Get the logger of the env_param.
        """
        if self._env_param is not None:
            return self._env_param.logger
        from irsim.config import env_param

        return env_param.logger
