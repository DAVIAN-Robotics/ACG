import collections.abc

import numpy as np
from robosuite.utils.placement_samplers import UniformRandomSampler


class MyUniformRandomSampler(UniformRandomSampler):
    def __init__(self, rng: np.random.Generator = None, *args, **kwargs):
        self.rng = rng
        super().__init__(*args, **kwargs)

    def _sample_x(self, object_horizontal_radius):
        """
        Samples the x location for a given object

        Args:
            object_horizontal_radius (float): Radius of the object currently being sampled for

        Returns:
            float: sampled x position
        """
        minimum, maximum = self.x_range
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        # return np.random.uniform(high=maximum, low=minimum)
        return self.rng.uniform(high=maximum, low=minimum)

    def _sample_y(self, object_horizontal_radius):
        """
        Samples the y location for a given object

        Args:
            object_horizontal_radius (float): Radius of the object currently being sampled for

        Returns:
            float: sampled y position
        """
        minimum, maximum = self.y_range
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        # return np.random.uniform(high=maximum, low=minimum)
        return self.rng.uniform(high=maximum, low=minimum)

    def _sample_quat(self):
        """
        Samples the orientation for a given object

        Returns:
            np.array: sampled object quaternion in (w,x,y,z) form

        Raises:
            ValueError: [Invalid rotation axis]
        """
        if self.rotation is None:
            # rot_angle = np.random.uniform(high=2 * np.pi, low=0)
            rot_angle = self.rng.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, collections.abc.Iterable):
            # rot_angle = np.random.uniform(high=max(self.rotation), low=min(self.rotation))
            rot_angle = self.rng.uniform(high=max(self.rotation), low=min(self.rotation))
        else:
            rot_angle = self.rotation

        # Return angle based on axis requested
        if self.rotation_axis == "x":
            return np.array([np.cos(rot_angle / 2), np.sin(rot_angle / 2), 0, 0])
        elif self.rotation_axis == "y":
            return np.array([np.cos(rot_angle / 2), 0, np.sin(rot_angle / 2), 0])
        elif self.rotation_axis == "z":
            return np.array([np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)])
        else:
            # Invalid axis specified, raise error
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(self.rotation_axis)
            )
