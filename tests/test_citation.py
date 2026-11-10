"""References to cite: the registry, cite() and the reminder of each environment."""

import re

import pytest

import particlegym
from particlegym import citation, envs
from particlegym import physics as pp

ENVIRONMENTS = [
    envs.Billiards,
    envs.BrickWall,
    envs.CellMigration,
    envs.CounterFlow,
    envs.Evacuation,
    envs.Herding,
    envs.Microswimmer,
    envs.OrbitTransfer,
    envs.RingRoad,
]


@pytest.mark.parametrize("cls", ENVIRONMENTS, ids=lambda cls: cls.__name__)
def test_every_environment_names_known_references(cls):
    assert isinstance(cls.references, tuple)
    assert all(key in citation.REFERENCES for key in cls.references)


def test_cite_lists_particlegym_first_and_every_reference_once():
    text = particlegym.cite()
    assert text.startswith("- Z. Chai, ParticleGym")
    assert f"version {particlegym.__version__}" in text

    lines = particlegym.cite(envs.Evacuation, envs.CounterFlow).split("\n")
    assert len(lines) == 3  # ParticleGym, Zhang et al. 2021, Helbing et al. 2000
    assert "Physica A 571, 125845 (2021)" in lines[1]
    assert "Nature 407, 487 (2000)" in lines[2]


def test_cite_gives_bibtex_entries_with_the_version():
    bibtex = particlegym.cite(envs.Herding, style="bibtex")
    keys = re.findall(r"^@\w+\{(\w+),", bibtex, flags=re.MULTILINE)
    assert keys == ["particlegym", "strombom2014"]
    assert f"version = {{{particlegym.__version__}}}" in bibtex
    assert bibtex.count("{") == bibtex.count("}")
    with pytest.raises(ValueError, match="style"):
        particlegym.cite(style="apa")


def test_the_first_environment_of_each_kind_prints_its_references_once(capsys, monkeypatch):
    monkeypatch.delenv("PARTICLEGYM_CITE", raising=False)
    monkeypatch.setattr(citation, "_reminded", set())
    door = [pp.Opening("y+", (4.5, 5.5))]

    envs.Evacuation(exits=door)
    first = capsys.readouterr().err
    assert "if you publish results obtained with Evacuation, please cite" in first
    assert "Physica A 571, 125845 (2021)" in first and "PARTICLEGYM_CITE=0" in first

    envs.Evacuation(exits=door)
    assert capsys.readouterr().err == ""  # once per kind of environment

    envs.Microswimmer()
    assert "Phys. Rev. Lett. 118, 158004" in capsys.readouterr().err


def test_the_reminder_can_be_turned_off(capsys, monkeypatch):
    monkeypatch.setenv("PARTICLEGYM_CITE", "0")
    monkeypatch.setattr(citation, "_reminded", set())
    envs.Microswimmer()
    assert capsys.readouterr().err == ""
