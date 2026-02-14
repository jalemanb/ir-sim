from __future__ import annotations

from operator import attrgetter
from typing import TYPE_CHECKING, Any, Optional

import yaml, json
import os

from irsim.util.util import file_check
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
        self.house_expo_path = house_expo_path
        self.house_expo_map_name = house_expo_map_name


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

        else:
            self.logger.error(
                f"{self.world_name} YAML File not found!, using default world config as alternative."
            )
    
    def initialize_objects(self) -> Any:
        """Construct world, objects and plot from the current parsed config.

        Returns:
            Tuple: ``(world, objects, env_plot, robot_collection, obstacle_collection, map_collection)``

        Notes:
            - Caches the created ``EnvPlot`` and ``objects`` internally for use
              during in-place reloads.
        """

        world_kwargs = dict(self.parse["world"])  # copy

        if not isinstance(self.parse.get("obstacle"), list):
            self.parse["obstacle"] = []

        if not isinstance(self.parse.get("robot"), list):
            self.parse["robot"] = []

        if self.house_expo_path is not None:
            world_kwargs.pop("obstacle_map", None)
            with open(os.path.join(self.house_expo_path, "json", self.house_expo_map_name + ".json"), "r", encoding="utf-8") as f:
                world_kwargs["map_attr"]  = json.load(f)

            world_kwargs["width"] = world_kwargs["map_attr"]["bbox"]["min"][0] + world_kwargs["map_attr"]["bbox"]["max"][0]
            world_kwargs["height"] = world_kwargs["map_attr"]["bbox"]["min"][1] + world_kwargs["map_attr"]["bbox"]["max"][1]

            if not isinstance(self.parse.get("obstacle"), list):
                self.parse["obstacle"] = []

            print(world_kwargs["map_attr"]["bbox"])
            print("width: ", world_kwargs["width"], " height: ",world_kwargs["height"])

            rooms = []

            for room_type in world_kwargs["map_attr"]["room_category"].keys():
                for room in world_kwargs["map_attr"]["room_category"][room_type]:
                    rooms.append(room)

            seed = None
            rng = np.random.default_rng(seed)
            method = "center"
            xy_noise_std = 0.01,
            clamp_to_room = True

            # pick a room index uniformly
            idx = int(rng.integers(0, len(rooms)))
            box = tuple(map(float, rooms[idx]))  # ensure float
            xmin, ymin, xmax, ymax = box
            if xmax <= xmin or ymax <= ymin:
                raise ValueError(f"Invalid box at idx={idx}: {box}")

            cx, cy = box_center(box)

            if method == "center":
                    x = cx + rng.normal(0.0, xy_noise_std)
                    y = cy + rng.normal(0.0, xy_noise_std)
            elif method == "uniform":
                # sample anywhere inside the room, then optionally add small noise
                x = rng.uniform(xmin, xmax) + rng.normal(0.0, xy_noise_std)
                y = rng.uniform(ymin, ymax) + rng.normal(0.0, xy_noise_std)
            else:
                raise ValueError("method must be 'center' or 'uniform'")

            if clamp_to_room:
                # keep inside bounds (with tiny margin so you don't sit exactly on a wall)
                eps = 1e-3
                x = float(np.clip(x, xmin + eps, xmax - eps))
                y = float(np.clip(y, ymin + eps, ymax - eps))
            else:
                x, y = float(x), float(y)

            yaw = float(rng.uniform(-np.pi, np.pi))  # alternative: uniform heading


            self.parse["robot"][0]["state"] = [cx, cy, yaw]


            verts = world_kwargs["map_attr"]["verts"]
            
            verts.append(world_kwargs["map_attr"]["verts"][0])

            self.parse["obstacle"].append({
                "shape": {"name": "linestring", "vertices": verts},
                "state": [0.0, 0.0, 0.0],
                "unobstructed": False,   # outline only (no collision)
                "color": "royalblue",   # anything but black
            })


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

        if not isinstance(self.parse.get("obstacle"), list):
            self.parse["obstacle"] = []

        if not isinstance(self.parse.get("robot"), list):
            self.parse["robot"] = []

        if self.house_expo_path is not None:
            world_kwargs.pop("obstacle_map", None)
            with open(os.path.join(self.house_expo_path, "json", self.house_expo_map_name + ".json"), "r", encoding="utf-8") as f:
                world_kwargs["map_attr"]  = json.load(f)

            world_kwargs["width"] = world_kwargs["map_attr"]["bbox"]["min"][0] + world_kwargs["map_attr"]["bbox"]["max"][0]
            world_kwargs["height"] = world_kwargs["map_attr"]["bbox"]["min"][1] + world_kwargs["map_attr"]["bbox"]["max"][1]

            if not isinstance(self.parse.get("obstacle"), list):
                self.parse["obstacle"] = []

            print(world_kwargs["map_attr"]["bbox"])
            print("width: ", world_kwargs["width"], " height: ",world_kwargs["height"])

            rooms = []

            for room_type in world_kwargs["map_attr"]["room_category"].keys():
                for room in world_kwargs["map_attr"]["room_category"][room_type]:
                    rooms.append(room)

            seed = None
            rng = np.random.default_rng(seed)
            method = "center"
            xy_noise_std = 0.15,
            yaw_noise_std = 0.10,  # radians
            clamp_to_room = True

            # pick a room index uniformly
            idx = int(rng.integers(0, len(rooms)))
            box = tuple(map(float, rooms[idx]))  # ensure float
            xmin, ymin, xmax, ymax = box
            if xmax <= xmin or ymax <= ymin:
                raise ValueError(f"Invalid box at idx={idx}: {box}")

            cx, cy = box_center(box)

            if method == "center":
                    x = cx + rng.normal(0.0, xy_noise_std)
                    y = cy + rng.normal(0.0, xy_noise_std)
            elif method == "uniform":
                # sample anywhere inside the room, then optionally add small noise
                x = rng.uniform(xmin, xmax) + rng.normal(0.0, xy_noise_std)
                y = rng.uniform(ymin, ymax) + rng.normal(0.0, xy_noise_std)
            else:
                raise ValueError("method must be 'center' or 'uniform'")

            if clamp_to_room:
                # keep inside bounds (with tiny margin so you don't sit exactly on a wall)
                eps = 1e-3
                x = float(np.clip(x, xmin + eps, xmax - eps))
                y = float(np.clip(y, ymin + eps, ymax - eps))
            else:
                x, y = float(x), float(y)

            yaw = float(rng.uniform(-np.pi, np.pi))  # alternative: uniform heading


            self.parse["robot"][0]["state"] = [cx, cy, yaw]


            verts = world_kwargs["map_attr"]["verts"]
            
            verts.append(world_kwargs["map_attr"]["verts"][0])

            self.parse["obstacle"].append({
                "shape": {"name": "linestring", "vertices": verts},
                "state": [0.0, 0.0, 0.0],
                "unobstructed": False,   # outline only (no collision)
                "color": "royalblue",   # anything but black
            })

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
