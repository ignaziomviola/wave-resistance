"""The CLI must run end to end and refuse contradictory or missing flotation input."""

import pytest

from wave_resistance.cli import main


def test_info_runs(sysser01_path, capsys):
    assert main(["info", str(sysser01_path)]) == 0
    out = capsys.readouterr().out
    assert "type 128 surfaces read: 2" in out
    assert "ignored entities" in out


def test_hydrostatics_with_displacement(sysser01_path, capsys):
    assert main(["hydrostatics", str(sysser01_path), "--displacement", "0.0376136",
                 "--nu", "24", "--nv", "120", "--sections", "60", "--compare"]) == 0
    out = capsys.readouterr().out
    assert "route spread" in out
    assert "comparison with published Sysser 01" in out
    assert "clear the free surface by" in out


def test_hydrostatics_with_draught_and_attitude(sysser01_path, capsys):
    assert main(["hydrostatics", str(sysser01_path), "--draught", "0.127",
                 "--heel", "10", "--trim", "0.5", "--sinkage", "0.002",
                 "--nu", "24", "--nv", "120", "--sections", "60"]) == 0
    out = capsys.readouterr().out
    assert "heel +10.000 deg" in out


def test_both_flotation_flags_is_an_error(sysser01_path):
    assert main(["hydrostatics", str(sysser01_path),
                 "--displacement", "0.03", "--draught", "0.12"]) == 2


def test_neither_flotation_flag_is_an_error(sysser01_path):
    assert main(["hydrostatics", str(sysser01_path)]) == 2
