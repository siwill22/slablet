"""Tests for slablet.gplates.models: config block -> reconstruction files.

The gprm tests use models already in gprm's download cache and are skipped otherwise
(nothing is downloaded).
"""
import os

import pytest

from slablet.gplates import models


def _touch(folder, *names):
    for n in names:
        path = folder / n
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('')


def test_original_top_level_form_is_read_as_files(tmp_path):
    _touch(tmp_path, 'rot/b.rot', 'rot/a.rot', 'top/2.gpml', 'top/1.gpml', 'Continents/c.gpml')
    r = models.load({'model_dir': str(tmp_path), 'rotation_files': ['rot/*.rot'], 'topology_files': ['top/*.gpml']})
    assert [os.path.basename(f) for f in r.rotation_files] == ['a.rot', 'b.rot']      # sorted
    assert [os.path.basename(f) for f in r.topology_files] == ['1.gpml', '2.gpml']
    assert [os.path.basename(f) for f in r.continent_files] == ['c.gpml']            # default layout
    assert r.craton_files is None                                                    # not there
    assert r.frame == 'unknown'


def test_files_block_matches_top_level_form(tmp_path):
    _touch(tmp_path, 'a.rot', 't.gpml')
    top = {'model_dir': str(tmp_path), 'rotation_files': ['*.rot'], 'topology_files': ['*.gpml']}
    blk = {'reconstruction': {'source': 'files', **top, 'frame': 'mantle'}}
    a, b = models.load(top), models.load(blk)
    assert (a.rotation_files, a.topology_files) == (b.rotation_files, b.topology_files)
    assert b.frame == 'mantle'


def test_unmatched_pattern_and_unknown_source_stop(tmp_path):
    _touch(tmp_path, 'a.rot')
    with pytest.raises(SystemExit, match='no file matches'):
        models.load({'model_dir': str(tmp_path), 'rotation_files': ['*.rot'], 'topology_files': ['*.gpml']})
    with pytest.raises(SystemExit, match='source must be one of'):
        models.load({'reconstruction': {'source': 'nowhere'}})


def test_top_level_files_and_a_block_together_stop(tmp_path):
    with pytest.raises(SystemExit, match='put them in the block'):
        models.block({'reconstruction': {'source': 'gprm', 'model': 'Muller2022'}, 'rotation_files': ['x.rot']})


def test_set_rotation_files_goes_where_the_config_keeps_them():
    top = {'rotation_files': ['a.rot']}
    blk = {'reconstruction': {'source': 'gprm', 'model': 'Muller2022'}}
    models.set_rotation_files(top, ['b.rot'])
    models.set_rotation_files(blk, ['b.rot'])
    assert top == {'rotation_files': ['b.rot']}
    assert blk['reconstruction']['rotation_files'] == ['b.rot'] and 'rotation_files' not in blk


def _gprm_cached(folder):
    gprm = pytest.importorskip('gprm')                                   # noqa: F841
    from pooch import os_cache
    if not os.path.isdir(os.path.join(str(os_cache('gprm')), folder)):
        pytest.skip('{} not in gprm\'s cache'.format(folder))
    return os.path.join(str(os_cache('gprm')), folder)


def test_gprm_muller2022_gives_the_same_files_as_the_folder():
    d = _gprm_cached('Muller2022')
    files = models.load({'model_dir': d, 'rotation_files': ['optimisation/1000_0_rotfile_MantleOpt.rot'],
                         'topology_files': ['Topologies/*.gpml']})
    g = models.load({'reconstruction': {'source': 'gprm', 'model': 'Muller2022', 'craton_files': ['Cratons/*.gpml']}})
    for k in ('model_dir', 'rotation_files', 'topology_files', 'continent_files', 'craton_files'):
        assert getattr(files, k) == getattr(g, k), k
    assert g.frame == 'mantle'


def test_gprm_rotation_override_makes_the_frame_unknown():
    _gprm_cached('Muller2022')
    g = models.load({'reconstruction': {'source': 'gprm', 'model': 'Muller2022',
                                        'rotation_files': ['optimisation/no_net_rotation_model.rot']}})
    assert os.path.basename(g.rotation_files[0]) == 'no_net_rotation_model.rot'
    assert g.frame == 'unknown'


def test_gprm_spin_axis_frame_is_refused_unless_allowed():
    _gprm_cached('Merdith2021')
    with pytest.raises(SystemExit, match='spin axis'):
        models.load({'reconstruction': {'source': 'gprm', 'model': 'Merdith2021'}})
    r = models.load({'reconstruction': {'source': 'gprm', 'model': 'Merdith2021', 'allow_spin_axis_frame': True}})
    assert r.frame == 'spin axis'
