from particlegym import physics as pp
from particlegym.io import write_extxyz


def test_write_extxyz_appends_frames_without_dead_particles(tmp_path):
    box = pp.Box((0.0, 0.0), (10.0, 5.0), periodic=(True, False))
    state = pp.create_state(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 1.0]],
        vel=[[0.1, 0.0], [0.0, 0.2], [0.0, 0.0]],
        radius=[0.3, 0.4, 0.5],
        alive=[True, False, True],
        kind=[0, 1, 1],
    )
    path = tmp_path / "trajectory.xyz"
    write_extxyz(path, state, box, time=0.5, type_names=["agent", "crowd"])
    write_extxyz(path, state, box, time=1.0, type_names=["agent", "crowd"])

    lines = path.read_text().splitlines()
    assert len(lines) == 2 * (2 + 2)
    assert lines[0] == "2"
    header = lines[1]
    assert 'Lattice="10 0 0 0 5 0 0 0 1"' in header
    assert 'Origin="0 0 -0.5"' in header
    assert 'pbc="T F F"' in header
    assert "Properties=species:S:1:pos:R:3:velo:R:3:radius:R:1:id:I:1" in header
    assert "Time=0.5" in header
    assert lines[2].split() == ["agent", "1", "2", "0", "0.1", "0", "0", "0.3", "0"]
    assert lines[3].split()[0] == "crowd"
    assert lines[3].split()[-1] == "2"  # original index of the third particle
    assert lines[4] == "2"
    assert "Time=1" in lines[5]
