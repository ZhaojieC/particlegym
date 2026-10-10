import jax.numpy as jnp
import numpy as np
import pytest

from particlegym import physics as pp


def test_minimum_image_only_along_periodic_axes():
    box = pp.Box((0.0, 0.0), (10.0, 4.0), periodic=(True, False))
    dr = box.displacement(jnp.array([0.5, 0.5]), jnp.array([9.5, 3.5]))
    np.testing.assert_allclose(dr, [1.0, -3.0])


def test_wrap_only_along_periodic_axes():
    box = pp.Box((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), periodic=(True, False, True))
    wrapped = box.wrap(jnp.array([[10.5, 10.5, -0.5]]))
    np.testing.assert_allclose(wrapped, [[0.5, 10.5, 9.5]])


def test_single_periodic_flag_applies_to_all_axes():
    assert pp.Box((0, 0, 0), (1, 1, 1), periodic=True).periodic == (True, True, True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lo": (0.0,), "hi": (1.0,)},
        {"lo": (0.0, 0.0), "hi": (1.0, 0.0)},
        {"lo": (0.0, 0.0), "hi": (1.0, 1.0), "periodic": (True,)},
    ],
)
def test_invalid_boxes_are_rejected(kwargs):
    with pytest.raises(ValueError):
        pp.Box(**kwargs)
