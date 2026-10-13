"""Pictures and pixel images: shapes appear where the particles are."""

import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from PIL import Image

from particlegym import envs, viz
from particlegym import physics as pp


def _centroid_and_area(frame, color):
    """Centroid (x, y) in image coordinates and pixel count of the pixels of one colour."""
    rows, cols = np.nonzero(np.all(frame == np.asarray(color, dtype=np.uint8), axis=-1))
    return np.array([cols.mean() + 0.5, rows.mean() + 0.5]), len(rows)


@pytest.mark.parametrize("supersample", [1, 3])
def test_disc_is_drawn_at_its_position_with_its_area(supersample):
    canvas = viz.Canvas((0.0, 0.0), (4.0, 2.0), width=400, supersample=supersample)
    canvas.discs([[1.0, 1.5], [3.0, 0.5]], [0.3, 0.2], viz.RED, mask=[True, False])
    frame = canvas.frame()

    assert frame.shape == (200, 400, 3)
    assert frame.dtype == np.uint8
    # 100 pixels per unit length; the y axis points up, so y = 1.5 is row 50.
    # Edge pixels are blends of red and white, weighted here by the red share.
    weight = (255.0 - frame[..., 1]) / (255.0 - viz.RED[1])
    cols, rows = np.meshgrid(np.arange(400) + 0.5, np.arange(200) + 0.5)
    centroid = np.array([(weight * cols).sum(), (weight * rows).sum()]) / weight.sum()
    # Discs are placed on whole pixels of the oversampled image.
    np.testing.assert_allclose(centroid, [100.0, 50.0], atol=0.5 / supersample + 1e-6)
    np.testing.assert_allclose(weight.sum(), np.pi * 30.0**2, rtol=0.01)
    # The masked disc is not drawn.
    assert np.all(frame[100:, 200:] == 255)


def test_shapes_are_drawn_in_place():
    canvas = viz.Canvas((0.0, 0.0), (10.0, 10.0), width=100, supersample=1)
    canvas.shapes(pp.Plates.from_segments([[(2.0, 5.0), (8.0, 5.0)]]), viz.BLACK, thickness=0.2)
    canvas.shapes(pp.Boxes([[1.0, 7.0]], [[3.0, 9.0]]), viz.BLUE)
    canvas.shapes(pp.Spheres([[7.0, 2.0]], [1.0]), viz.GREEN)
    frame = canvas.frame()

    assert tuple(frame[50, 50]) == viz.BLACK  # on the plate
    assert tuple(frame[50, 10]) == viz.WHITE  # beside the plate
    centroid, area = _centroid_and_area(frame, viz.BLUE)
    np.testing.assert_allclose(centroid, [20.0, 20.0], atol=0.6)
    np.testing.assert_allclose(area, 400, rtol=0.1)
    centroid, area = _centroid_and_area(frame, viz.GREEN)
    np.testing.assert_allclose(centroid, [70.0, 80.0], atol=0.6)
    np.testing.assert_allclose(area, np.pi * 100, rtol=0.1)
    with pytest.raises(ValueError, match="only 2D shapes"):
        canvas.shapes(pp.Spheres([[0.0, 0.0, 0.0]], [1.0]))


def test_evacuation_render_matches_the_state():
    env = envs.Evacuation(
        room=(10.0, 10.0),
        exits=[pp.Opening("y+", (4.5, 5.5))],
        obstacles=[pp.Spheres([[5.0, 7.0]], [0.6])],
        n_agents=5,
        n_crowd=20,
    )
    _, state = env.reset(jax.random.key(0))
    alive = np.ones(env.n, dtype=bool)
    alive[[1, 7]] = False
    state = state._replace(particles=state.particles._replace(alive=jnp.asarray(alive)))
    frame = env.render(state, width=480)

    assert frame.shape == (480, 480, 3)
    # The picture shows [-1, 11] along both axes: 40 pixels per metre.
    pos = np.asarray(state.particles.pos)
    cols = np.floor((pos[:, 0] + 1.0) * 40).astype(int)
    rows = np.floor((11.0 - pos[:, 1]) * 40).astype(int)
    for i in range(env.n):
        expected = viz.WHITE if not alive[i] else (viz.RED if i < 5 else viz.BLUE)
        assert tuple(frame[rows[i], cols[i]]) == expected, i
    assert tuple(frame[160, 240]) == viz.GREY  # the pillar at (5, 7)
    assert tuple(frame[440, 240]) == viz.BLACK  # the bottom wall at y = 0
    assert tuple(frame[40, 240]) == viz.WHITE  # the door in the top wall

    room_3d = envs.Evacuation(room=(4.0, 4.0, 3.0), n_agents=2)
    with pytest.raises(NotImplementedError):
        room_3d.render(room_3d.reset(jax.random.key(0))[1])


def test_save_gif_keeps_frames_size_and_rate(tmp_path):
    frames = []
    for k in range(6):
        canvas = viz.Canvas((0.0, 0.0), (6.0, 3.0), width=120)
        canvas.discs([[0.5 + k, 1.5]], 0.4, viz.ORANGE)
        frames.append(canvas.frame())
    path = tmp_path / "moving.gif"
    viz.save_gif(path, frames, fps=20)

    with Image.open(path) as gif:
        assert gif.n_frames == 6
        assert gif.size == (120, 60)
        assert gif.info["duration"] == 50
        gif.seek(4)
        last = np.asarray(gif.convert("RGB"))
    # The colour table reproduces the flat colours of the drawing.
    assert np.abs(last.astype(int) - frames[4]).max() <= 16
    with pytest.raises(ValueError, match="no frames"):
        viz.save_gif(path, [])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_save_mp4_writes_a_video(tmp_path):
    frames = [np.full((45, 81, 3), 40 * k, dtype=np.uint8) for k in range(5)]  # odd sizes
    path = tmp_path / "video.mp4"
    viz.save_mp4(path, frames, fps=10)
    assert path.stat().st_size > 500


def test_upscale_repeats_pixels():
    frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    big = viz.upscale(frame, 4)
    assert big.shape == (8, 12, 3)
    np.testing.assert_array_equal(big[5, 9], frame[1, 2])


def test_pixel_coverage_matches_areas_and_positions():
    lo, hi, shape = (0.0, 0.0), (8.0, 4.0), (40, 80)
    points = viz.pixel_centres(lo, hi, shape)
    pixel = 0.1
    assert points.shape == (40, 80, 2)
    np.testing.assert_allclose(points[0, 0], [0.05, 3.95])  # the top-left pixel

    pos = jnp.array([[2.0, 3.0], [6.0, 1.0]])
    image = jax.jit(viz.disc_coverage)(points, pos, jnp.array([0.7, 0.5]), pixel)
    assert image.shape == (40, 80)
    assert float(image.min()) >= 0.0 and float(image.max()) == 1.0
    left, right = np.asarray(image[:, :40]), np.asarray(image[:, 40:])
    np.testing.assert_allclose(left.sum() * pixel**2, np.pi * 0.7**2, rtol=0.01)
    np.testing.assert_allclose(right.sum() * pixel**2, np.pi * 0.5**2, rtol=0.01)
    centroid = (np.asarray(points[:, :40]) * left[..., None]).sum(axis=(0, 1)) / left.sum()
    np.testing.assert_allclose(centroid, [2.0, 3.0], atol=1e-3)

    masked = viz.disc_coverage(points, pos, 0.5, pixel, mask=jnp.array([False, True]))
    assert float(masked[:, :40].sum()) == 0.0

    boxes = viz.box_coverage(points, jnp.array([[1.0, 1.0]]), jnp.array([[3.0, 2.5]]), pixel)
    np.testing.assert_allclose(float(boxes.sum()) * pixel**2, 2.0 * 1.5, rtol=0.01)
    assert float(boxes[20, 20]) == 1.0  # the point (2.05, 1.95) is inside


def test_pixel_coverage_batches_with_vmap():
    points = viz.pixel_centres((0.0, 0.0), (1.0, 1.0), (16, 16))
    pos = jnp.array([[[0.25, 0.25]], [[0.75, 0.75]]])
    images = jax.vmap(lambda p: viz.disc_coverage(points, p, 0.1, 1 / 16))(pos)
    assert images.shape == (2, 16, 16)
    assert float(images[0, 12, 4]) > 0.5 and float(images[1, 4, 12]) > 0.5  # row 0 is the top
    assert float(images[0, 4, 12]) == 0.0
