"""The seafloor and lithosphere caches are kept apart for runs tracked from different starts."""
import types

from slablet.gplates import seafloor_tracker as st


def test_cache_folder_records_the_start(monkeypatch):
    monkeypatch.setattr(st.CachedTrenchSamples, 'for_config',
                        classmethod(lambda cls, cfg, sample_deg=1.0: types.SimpleNamespace(folder='/cache/model')))
    from_250 = st.folder_for({'start_ma': 250})[1]
    from_1000 = st.folder_for({'start_ma': 1000.0})[1]
    assert from_250 != from_1000
    assert from_1000 == '/cache/model/seafloor_v1_from1000'
    assert st.folder_for({'start_ma': 250}, start=1000)[1] == from_1000   # an explicit start wins
