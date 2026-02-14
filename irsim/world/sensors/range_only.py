from typing import TYPE_CHECKING, Optional, Literal
import numpy as np
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

from irsim.util.random import rng
from irsim.util.util import transform_point_with_state

if TYPE_CHECKING:
    from irsim.world.object_base import ObjectBase


class RangeOnly:
    def __init__(
        self,
        state: Optional[np.ndarray] = None,
        obj_id: int = 0,
        target_mode: Literal["goal", "pedestrian"] = "goal",
        target_position: Optional[list[float]] = None,
        target_pedestrian_id: Optional[int] = None,
        range_max: float = 30.0,
        noise: bool = True,
        std: float = 0.15,
        offset: Optional[list[float]] = None,
        alpha: float = 0.2,
        color: str = "cyan",
        show_circle: bool = True,
        show_target: bool = True,
        auto_select_pedestrian: bool = True,
        **kwargs,
    ) -> None:
        if offset is None:
            offset = [0, 0, 0]

        self.sensor_type = "range_only"

        self.target_mode = target_mode
        self.auto_select_pedestrian = auto_select_pedestrian
        self._target_pedestrian_id = target_pedestrian_id
        self._target_position = np.array(target_position, dtype=float) if target_position else None
        self._pedestrian_selected = False

        self.range_max = float(range_max)
        self.noise = bool(noise)
        self.std = float(std)
        self.offset = np.c_[offset]

        self.alpha = float(alpha)
        self.color = color
        self.show_circle = bool(show_circle)
        self.show_target = bool(show_target)

        self._state = state
        self.sensor_origin = None
        if state is not None:
            self.sensor_origin = transform_point_with_state(self.offset, self._state)

        self.last_measurement: Optional[float] = None
        self.true_range: Optional[float] = None

        self.obj_id = obj_id
        self.parent: Optional[ObjectBase] = None

        # --- plotting state ---
        self._ax = None
        self.circle_patch: Optional[mpatches.Circle] = None
        self._target_artist = None
        self._sensor_artist = None

        self.plot_patch_list = []
        self.plot_line_list = []
        self.plot_text_list = []

        # Validate configuration
        if self.target_mode == "goal" and self._target_position is None:
            raise ValueError("target_position must be provided when target_mode='goal'")

        if self.target_mode == "pedestrian" and (not auto_select_pedestrian) and (target_pedestrian_id is None):
            raise ValueError("target_pedestrian_id must be provided when target_mode='pedestrian' and auto_select_pedestrian=False")

        # Compute an initial measurement so first draw shows something
        if self._state is not None:
            self.measure()

    @property
    def state(self) -> np.ndarray:
        return self._state

    def get_sensor_position(self) -> np.ndarray:
        if self.sensor_origin is not None:
            return np.array([self.sensor_origin[0, 0], self.sensor_origin[1, 0]], dtype=float)
        return np.zeros(2, dtype=float)

    def _select_random_pedestrian(self):
        if self._pedestrian_selected:
            return
        if self.parent is None or self.parent._env is None:
            return

        # you must ensure your env exposes this attribute if you use pedestrian mode
        jupedsim_mgr = getattr(self.parent._env, "jupedsim_manager", None)
        if jupedsim_mgr is None or len(getattr(jupedsim_mgr, "agent_ids", [])) == 0:
            return

        available = jupedsim_mgr.agent_ids
        idx = rng.integers(0, len(available))
        self._target_pedestrian_id = int(available[idx])
        self._pedestrian_selected = True

    def get_target_position(self) -> Optional[np.ndarray]:
        if self.target_mode == "goal":
            return self._target_position

        # pedestrian mode
        if not self._pedestrian_selected and self.auto_select_pedestrian:
            self._select_random_pedestrian()

        if self._target_pedestrian_id is None:
            return None

        if self.parent is None or self.parent._env is None:
            return None

        jupedsim_mgr = getattr(self.parent._env, "jupedsim_manager", None)
        if jupedsim_mgr is None:
            return None

        try:
            agent = jupedsim_mgr.sim.agent(self._target_pedestrian_id)
            return np.array(agent.position, dtype=float)
        except Exception:
            return None

    def step(self, state: np.ndarray):
        self._state = state
        self.sensor_origin = transform_point_with_state(self.offset, self._state)
        self.measure()

    def measure(self) -> Optional[float]:
        sensor_pos = self.get_sensor_position()
        target_pos = self.get_target_position()

        if target_pos is None:
            self.last_measurement = None
            self.true_range = None
            return None

        self.true_range = float(np.linalg.norm(target_pos - sensor_pos))
        if self.true_range > self.range_max:
            self.last_measurement = None
            return None

        if self.noise:
            val = self.true_range + float(rng.normal(0, self.std))
        else:
            val = self.true_range

        self.last_measurement = max(0.0, float(val))
        return self.last_measurement

    def get_scan(self):
        return {
            "range": self.last_measurement,
            "true_range": self.true_range,
            "range_max": self.range_max,
            "target_mode": self.target_mode,
            "target_position": None if self.get_target_position() is None else self.get_target_position().copy(),
            "sensor_position": self.get_sensor_position().copy(),
            "target_pedestrian_id": self._target_pedestrian_id if self.target_mode == "pedestrian" else None,
        }

    def get_offset(self):
        return np.squeeze(self.offset).tolist()

    # ---------- plotting ----------
    def plot(self, ax, state: Optional[np.ndarray] = None, **kwargs):
        # IR-Sim calls plot() / _init_plot() on first draw; we treat plot() as safe too.
        if state is not None:
            self._state = state
            self.sensor_origin = transform_point_with_state(self.offset, self._state)
            if self.last_measurement is None:
                self.measure()
        if self._ax is None:
            self._init_plot(ax)
        else:
            self._step_plot()

    def _init_plot(self, ax, **kwargs):
        if isinstance(ax, Axes3D):
            return

        self._ax = ax

        # ensure we have a measurement on first draw
        if self.last_measurement is None:
            self.measure()

        sensor_pos = self.get_sensor_position()
        target_pos = self.get_target_position()

        # create circle artist once
        self.circle_patch = mpatches.Circle(
            (sensor_pos[0], sensor_pos[1]),
            radius=0.0,
            fill=False,
            linewidth=1.5,
            linestyle="--",
            alpha=self.alpha,
            color=self.color,
            zorder=5,
        )
        ax.add_patch(self.circle_patch)
        self.plot_patch_list.append(self.circle_patch)

        # create target marker once
        (self._target_artist,) = ax.plot(
            [], [], "x",
            color="red",
            markersize=10,
            markeredgewidth=2.5,
            zorder=10,
        )
        self.plot_line_list.append([self._target_artist])

        # optional sensor dot (if offset)
        (self._sensor_artist,) = ax.plot(
            [], [], "o",
            color=self.color,
            markersize=4,
            zorder=10,
        )
        self.plot_line_list.append([self._sensor_artist])

        # draw initial state
        self._step_plot()

    def step_plot(self):
        self._step_plot()

    def _step_plot(self):
        if self._ax is None or isinstance(self._ax, Axes3D):
            return

        sensor_pos = self.get_sensor_position()
        target_pos = self.get_target_position()

        # update circle
        if self.show_circle and self.last_measurement is not None and self.circle_patch is not None:
            self.circle_patch.set_center((sensor_pos[0], sensor_pos[1]))
            self.circle_patch.set_radius(float(self.last_measurement))
            self.circle_patch.set_visible(True)
        elif self.circle_patch is not None:
            self.circle_patch.set_visible(False)

        # update target marker
        if self.show_target and target_pos is not None and self._target_artist is not None:
            self._target_artist.set_data([target_pos[0]], [target_pos[1]])
            self._target_artist.set_visible(True)
        elif self._target_artist is not None:
            self._target_artist.set_visible(False)

        # update sensor dot only if offset is meaningful
        if self._sensor_artist is not None:
            if float(np.linalg.norm(self.offset[:2])) > 1e-3:
                self._sensor_artist.set_data([sensor_pos[0]], [sensor_pos[1]])
                self._sensor_artist.set_visible(True)
            else:
                self._sensor_artist.set_visible(False)

    def plot_clear(self):
        for patch in self.plot_patch_list:
            try:
                patch.remove()
            except Exception:
                pass
        for line_group in self.plot_line_list:
            try:
                line_group[0].remove()
            except Exception:
                pass
        for text in self.plot_text_list:
            try:
                text.remove()
            except Exception:
                pass

        self.plot_patch_list = []
        self.plot_line_list = []
        self.plot_text_list = []
        self.circle_patch = None
        self._target_artist = None
        self._sensor_artist = None
        self._ax = None
