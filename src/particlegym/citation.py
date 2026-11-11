# ParticleGym, https://github.com/ZhaojieC/particlegym (MIT license).
# Please cite it in published work: particlegym.cite() gives the references.
"""References to cite when using ParticleGym.

Every environment lists the published models it is built on in its
``references`` attribute. :func:`cite` returns the references for
ParticleGym and for the environments passed to it, as text or BibTeX::

    import particlegym

    print(particlegym.cite(env))  # one line per reference
    print(particlegym.cite(env, style="bibtex"))

The first environment of each kind that a process creates prints its
references once, like the citation reminder of LAMMPS. Set the environment
variable ``PARTICLEGYM_CITE=0`` to turn the reminder off.
"""

from __future__ import annotations

import os
import sys
from typing import NamedTuple


class Reference(NamedTuple):
    """A work to cite: one line of text and a BibTeX entry."""

    text: str
    bibtex: str


REFERENCES = {
    "particlegym": Reference(
        "Z. Chai, ParticleGym: fast, physically validated particle-dynamics environments for "
        "reinforcement learning, version {version}, https://github.com/ZhaojieC/particlegym",
        "@software{particlegym,\n"
        "  author = {Chai, Zhaojie},\n"
        "  title = {{ParticleGym}: fast, physically validated particle-dynamics environments "
        "for reinforcement learning},\n"
        "  version = {{version}},\n"
        "  year = {2026},\n"
        "  url = {https://github.com/ZhaojieC/particlegym}\n"
        "}",
    ),
    "zhang2021": Reference(
        "Y. Zhang, Z. Chai and G. Lykotrafitis, Deep reinforcement learning with a particle "
        "dynamics environment applied to emergency evacuation of a room with obstacles, "
        "Physica A 571, 125845 (2021), https://doi.org/10.1016/j.physa.2021.125845",
        "@article{zhang2021,\n"
        "  author = {Zhang, Yihao and Chai, Zhaojie and Lykotrafitis, George},\n"
        "  title = {Deep reinforcement learning with a particle dynamics environment applied "
        "to emergency evacuation of a room with obstacles},\n"
        "  journal = {Physica A: Statistical Mechanics and its Applications},\n"
        "  volume = {571},\n"
        "  pages = {125845},\n"
        "  year = {2021},\n"
        "  doi = {10.1016/j.physa.2021.125845}\n"
        "}",
    ),
    "zhang2020": Reference(
        "Y. Zhang, Z. Chai, Y. Sun and G. Lykotrafitis, A deep reinforcement learning model "
        "based on deterministic policy gradient for collective neural crest cell migration, "
        "arXiv:2007.03190 (2020), https://arxiv.org/abs/2007.03190",
        "@article{zhang2020,\n"
        "  author = {Zhang, Yihao and Chai, Zhaojie and Sun, Yubing and Lykotrafitis, George},\n"
        "  title = {A deep reinforcement learning model based on deterministic policy gradient "
        "for collective neural crest cell migration},\n"
        "  journal = {arXiv preprint arXiv:2007.03190},\n"
        "  year = {2020},\n"
        "  doi = {10.48550/arXiv.2007.03190}\n"
        "}",
    ),
    "helbing2000": Reference(
        "D. Helbing, I. Farkas and T. Vicsek, Simulating dynamical features of escape panic, "
        "Nature 407, 487 (2000), https://doi.org/10.1038/35035023",
        "@article{helbing2000,\n"
        "  author = {Helbing, Dirk and Farkas, Ill{\\'e}s and Vicsek, Tam{\\'a}s},\n"
        "  title = {Simulating dynamical features of escape panic},\n"
        "  journal = {Nature},\n"
        "  volume = {407},\n"
        "  pages = {487--490},\n"
        "  year = {2000},\n"
        "  doi = {10.1038/35035023}\n"
        "}",
    ),
    "colabrese2017": Reference(
        "S. Colabrese, K. Gustavsson, A. Celani and L. Biferale, Flow navigation by smart "
        "microswimmers via reinforcement learning, Phys. Rev. Lett. 118, 158004 (2017), "
        "https://doi.org/10.1103/PhysRevLett.118.158004",
        "@article{colabrese2017,\n"
        "  author = {Colabrese, Simona and Gustavsson, Kristian and Celani, Antonio and "
        "Biferale, Luca},\n"
        "  title = {Flow navigation by smart microswimmers via reinforcement learning},\n"
        "  journal = {Physical Review Letters},\n"
        "  volume = {118},\n"
        "  pages = {158004},\n"
        "  year = {2017},\n"
        "  doi = {10.1103/PhysRevLett.118.158004}\n"
        "}",
    ),
    "strombom2014": Reference(
        "D. Strömbom, R. P. Mann, A. M. Wilson, S. Hailes, A. J. Morton, D. J. T. Sumpter and "
        "A. J. King, Solving the shepherding problem: heuristics for herding autonomous, "
        "interacting agents, J. R. Soc. Interface 11, 20140719 (2014), "
        "https://doi.org/10.1098/rsif.2014.0719",
        "@article{strombom2014,\n"
        '  author = {Str{\\"o}mbom, Daniel and Mann, Richard P. and Wilson, Alan M. and '
        "Hailes, Stephen and Morton, A. Jennifer and Sumpter, David J. T. and King, "
        "Andrew J.},\n"
        "  title = {Solving the shepherding problem: heuristics for herding autonomous, "
        "interacting agents},\n"
        "  journal = {Journal of The Royal Society Interface},\n"
        "  volume = {11},\n"
        "  number = {100},\n"
        "  pages = {20140719},\n"
        "  year = {2014},\n"
        "  doi = {10.1098/rsif.2014.0719}\n"
        "}",
    ),
    "bhatia2021": Reference(
        "J. S. Bhatia, H. Jackson, Y. Tian, J. Xu and W. Matusik, Evolution Gym: A large-scale "
        "benchmark for evolving soft robots, Advances in Neural Information Processing Systems "
        "34 (2021), https://arxiv.org/abs/2201.09863",
        "@inproceedings{bhatia2021,\n"
        "  author = {Bhatia, Jagdeep Singh and Jackson, Holly and Tian, Yunsheng and Xu, Jie "
        "and Matusik, Wojciech},\n"
        "  title = {Evolution Gym: A Large-Scale Benchmark for Evolving Soft Robots},\n"
        "  booktitle = {Advances in Neural Information Processing Systems},\n"
        "  volume = {34},\n"
        "  year = {2021},\n"
        "  eprint = {2201.09863},\n"
        "  archiveprefix = {arXiv}\n"
        "}",
    ),
    "drasdo2005": Reference(
        "D. Drasdo and S. Hoehme, A single-cell-based model of tumor growth in vitro: "
        "monolayers and spheroids, Phys. Biol. 2, 133 (2005), "
        "https://doi.org/10.1088/1478-3975/2/3/001",
        "@article{drasdo2005,\n"
        "  author = {Drasdo, Dirk and Hoehme, Stefan},\n"
        "  title = {A single-cell-based model of tumor growth in vitro: monolayers and "
        "spheroids},\n"
        "  journal = {Physical Biology},\n"
        "  volume = {2},\n"
        "  pages = {133--147},\n"
        "  year = {2005},\n"
        "  doi = {10.1088/1478-3975/2/3/001}\n"
        "}",
    ),
    "gatenby2009": Reference(
        "R. A. Gatenby, A. S. Silva, R. J. Gillies and B. R. Frieden, Adaptive therapy, "
        "Cancer Res. 69, 4894 (2009), https://doi.org/10.1158/0008-5472.CAN-08-3658",
        "@article{gatenby2009,\n"
        "  author = {Gatenby, Robert A. and Silva, Ariosto S. and Gillies, Robert J. and "
        "Frieden, B. Roy},\n"
        "  title = {Adaptive Therapy},\n"
        "  journal = {Cancer Research},\n"
        "  volume = {69},\n"
        "  number = {11},\n"
        "  pages = {4894--4903},\n"
        "  year = {2009},\n"
        "  doi = {10.1158/0008-5472.CAN-08-3658}\n"
        "}",
    ),
    "zhang2017": Reference(
        "J. Zhang, J. J. Cunningham, J. S. Brown and R. A. Gatenby, Integrating evolutionary "
        "dynamics into treatment of metastatic castrate-resistant prostate cancer, Nat. "
        "Commun. 8, 1816 (2017), https://doi.org/10.1038/s41467-017-01968-5",
        "@article{zhang2017,\n"
        "  author = {Zhang, Jingsong and Cunningham, Jessica J. and Brown, Joel S. and "
        "Gatenby, Robert A.},\n"
        "  title = {Integrating evolutionary dynamics into treatment of metastatic "
        "castrate-resistant prostate cancer},\n"
        "  journal = {Nature Communications},\n"
        "  volume = {8},\n"
        "  pages = {1816},\n"
        "  year = {2017},\n"
        "  doi = {10.1038/s41467-017-01968-5}\n"
        "}",
    ),
    "gallaher2018": Reference(
        "J. A. Gallaher, P. M. Enriquez-Navas, K. A. Luddy, R. A. Gatenby and A. R. A. "
        "Anderson, Spatial heterogeneity and evolutionary dynamics modulate time to "
        "recurrence in continuous and adaptive cancer therapies, Cancer Res. 78, 2127 "
        "(2018), https://doi.org/10.1158/0008-5472.CAN-17-2649",
        "@article{gallaher2018,\n"
        "  author = {Gallaher, Jill A. and Enriquez-Navas, Pedro M. and Luddy, Kimberly A. "
        "and Gatenby, Robert A. and Anderson, Alexander R. A.},\n"
        "  title = {Spatial heterogeneity and evolutionary dynamics modulate time to "
        "recurrence in continuous and adaptive cancer therapies},\n"
        "  journal = {Cancer Research},\n"
        "  volume = {78},\n"
        "  number = {8},\n"
        "  pages = {2127--2139},\n"
        "  year = {2018},\n"
        "  doi = {10.1158/0008-5472.CAN-17-2649}\n"
        "}",
    ),
    "sugiyama2008": Reference(
        "Y. Sugiyama, M. Fukui, M. Kikuchi, K. Hasebe, A. Nakayama, K. Nishinari, S. Tadaki and "
        "S. Yukawa, Traffic jams without bottlenecks: experimental evidence for the physical "
        "mechanism of the formation of a jam, New J. Phys. 10, 033001 (2008), "
        "https://doi.org/10.1088/1367-2630/10/3/033001",
        "@article{sugiyama2008,\n"
        "  author = {Sugiyama, Yuki and Fukui, Minoru and Kikuchi, Macoto and Hasebe, Katsuya "
        "and Nakayama, Akihiro and Nishinari, Katsuhiro and Tadaki, Shin-ichi and Yukawa, "
        "Satoshi},\n"
        "  title = {Traffic jams without bottlenecks: experimental evidence for the physical "
        "mechanism of the formation of a jam},\n"
        "  journal = {New Journal of Physics},\n"
        "  volume = {10},\n"
        "  pages = {033001},\n"
        "  year = {2008},\n"
        "  doi = {10.1088/1367-2630/10/3/033001}\n"
        "}",
    ),
    "treiber2000": Reference(
        "M. Treiber, A. Hennecke and D. Helbing, Congested traffic states in empirical "
        "observations and microscopic simulations, Phys. Rev. E 62, 1805 (2000), "
        "https://doi.org/10.1103/PhysRevE.62.1805",
        "@article{treiber2000,\n"
        "  author = {Treiber, Martin and Hennecke, Ansgar and Helbing, Dirk},\n"
        "  title = {Congested traffic states in empirical observations and microscopic "
        "simulations},\n"
        "  journal = {Physical Review E},\n"
        "  volume = {62},\n"
        "  pages = {1805--1824},\n"
        "  year = {2000},\n"
        "  doi = {10.1103/PhysRevE.62.1805}\n"
        "}",
    ),
    "wu2022": Reference(
        "C. Wu, A. R. Kreidieh, K. Parvate, E. Vinitsky and A. M. Bayen, Flow: A modular "
        "learning framework for mixed autonomy traffic, IEEE Trans. Robot. 38, 1270 (2022), "
        "https://doi.org/10.1109/TRO.2021.3087314",
        "@article{wu2022,\n"
        "  author = {Wu, Cathy and Kreidieh, Abdul Rahman and Parvate, Kanaad and Vinitsky, "
        "Eugene and Bayen, Alexandre M.},\n"
        "  title = {Flow: A Modular Learning Framework for Mixed Autonomy Traffic},\n"
        "  journal = {IEEE Transactions on Robotics},\n"
        "  volume = {38},\n"
        "  number = {2},\n"
        "  pages = {1270--1286},\n"
        "  year = {2022},\n"
        "  doi = {10.1109/TRO.2021.3087314}\n"
        "}",
    ),
}

# Kinds of environment whose references have been printed in this process.
_reminded: set[str] = set()


def cite(*environments, style: str = "text") -> str:
    """The references for ParticleGym and for the given environments.

    Args:
        *environments: Environments, or environment classes, whose
            ``references`` follow ParticleGym's own, each listed once.
        style: ``"text"`` for one line per reference, ``"bibtex"`` for BibTeX
            entries.

    Returns:
        The references, ParticleGym first.
    """
    if style not in ("text", "bibtex"):
        raise ValueError("style must be 'text' or 'bibtex'")
    keys = ["particlegym"]
    for env in environments:
        for key in getattr(env, "references", ()):
            if key not in REFERENCES:
                raise KeyError(f"unknown reference {key!r}")
            if key not in keys:
                keys.append(key)
    if style == "text":
        return "\n".join(f"- {_reference(key).text}" for key in keys)
    return "\n\n".join(_reference(key).bibtex for key in keys)


def remind(env) -> None:
    """Print the references of an environment, once per kind and process.

    Environments call this at the end of their constructor. Nothing is
    printed when the environment variable ``PARTICLEGYM_CITE`` is ``0``.
    """
    kind = type(env).__name__
    if kind in _reminded or os.environ.get("PARTICLEGYM_CITE", "1").strip() == "0":
        return
    _reminded.add(kind)
    print(
        f"ParticleGym: if you publish results obtained with {kind}, please cite\n"
        f"{cite(env)}\n"
        "BibTeX: particlegym.cite(env, style='bibtex'). "
        "Set PARTICLEGYM_CITE=0 to hide this note.",
        file=sys.stderr,
    )


def _reference(key: str) -> Reference:
    """A reference, with the version of ParticleGym filled in."""
    if key != "particlegym":
        return REFERENCES[key]
    from particlegym import __version__

    text, bibtex = REFERENCES[key]
    return Reference(
        text.replace("{version}", __version__), bibtex.replace("{version}", __version__)
    )
