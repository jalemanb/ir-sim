"""Pedestrian obstacle controlled by JuPedSim."""
from irsim.world.object_base import ObjectBase


class ObstaclePedestrian(ObjectBase):
    def __init__(self,
                 jupedsim_id=None,
                 color="orange",
                 state_dim=3,
                 **kwargs):
        """Create a pedestrian obstacle controlled by JuPedSim.

        This is a lightweight wrapper around ObjectBase that doesn't use
        kinematics or behavior models - position is controlled externally
        by JuPedSim simulation.

        Args:
            jupedsim_id (int): The JuPedSim agent ID this object represents.
            color (str): Display color. Default "orange".
            state_dim (int): State vector dimension (>=3).
            **kwargs: Forwarded to ``ObjectBase``.
                     Common kwargs:
                     - shape: dict with 'name' and 'radius' (e.g., {'name': 'circle', 'radius': 0.3})
                     - description: str, path to image file for visualization
                     - state: [x, y, theta] initial position
        """
        # Don't use kinematics or behavior - JuPedSim controls movement
        kwargs['kinematics'] = None
        kwargs['behavior'] = None

        super().__init__(color=color, role="obstacle", state_dim=state_dim, **kwargs)

        self.jupedsim_id = jupedsim_id
        self.static = False  # Dynamic object for rendering

        # Initialize plot attributes (will be populated by _init_plot)
        if not hasattr(self, 'plot_attr_list'):
            self.plot_attr_list = []

    def step(self, velocity=None, sensor_step=True):
        """Override step to prevent kinematics processing.

        JuPedSim controls movement externally via update_from_jupedsim().
        This method only handles sensors.

        Args:
            velocity: Ignored (JuPedSim controls movement).
            sensor_step: Whether to step sensors.

        Returns:
            Current state.
        """
        # Only step sensors, skip kinematics/behavior
        if sensor_step and hasattr(self, 'sensors'):
            self.sensor_step()

        return self.state

    def update_from_jupedsim(self, position, velocity=None):
        """Update state from JuPedSim agent position.

        Args:
            position: Tuple of (x, y) from JuPedSim agent.
            velocity: Optional velocity vector (vx, vy) for orientation.
        """
        import numpy as np

        # Update position
        self._state[0] = position[0]
        self._state[1] = position[1]

        # Update orientation based on velocity if provided
        if velocity is not None and len(velocity) == 2:
            speed = np.linalg.norm(velocity)
            if speed > 0.01:  # Only update theta if moving
                # Calculate angle from velocity
                # Subtract π/2 (90°) because person.png originally faces 90° (north)
                self._state[2] = np.arctan2(velocity[1], velocity[0]) - np.pi / 2

        # Update geometry to match new state
        if hasattr(self, 'gf') and self.gf is not None:
            self._geometry = self.gf.step(self.state)

        # Record trajectory
        if hasattr(self, 'trajectory'):
            self.trajectory.append(self.state.copy())
