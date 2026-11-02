"""Which reconstruction a run uses: one config block in, the files slablet needs out.

    reconstruction:
      source: gprm                   # gprm | plate_model_manager | files
      model: Muller2022              # gprm: fetch_<model>; plate_model_manager: its model name
      options: {NNR: false}          # gprm only: keyword arguments of fetch_<model>

source: gprm
    gprm (GPlatesReconstructionModel) downloads the model on first use and keeps it in its own
    cache.  gprm records what each anchor plate of a model is fixed to: slablets sink vertically
    in the anchor plate's frame, so a frame fixed to the spin axis (palaeomagnetic, no true
    polar wander correction) is refused unless allow_spin_axis_frame: true.
source: plate_model_manager
    GPlately's PlateModelManager; model names as it lists them (muller2022, muller2019, ...).
    data_dir: where it keeps the files (default ~/.cache/slablet/plate_model_manager).  It does
    not record what a model's frame is fixed to; the run prints a note.
source: files
    model_dir plus glob patterns relative to it: rotation_files, topology_files and, for the
    lithosphere lid, continent_files and craton_files (each pattern must match a file).

Every source also takes:
    rotation_files   patterns relative to model_dir (for gprm and plate_model_manager: the folder
                     that holds all of the model's files), replacing the model's own rotations:
                     how to choose another frame.  The frame is then 'unknown' unless frame: says
    continent_files, craton_files   the same, replacing the model's polygons (gprm models have no
                     cratons: give craton_files for the lithosphere lid, or [] for none)
    frame            'mantle' or 'spin axis': what the anchor plate is fixed to, when the source
                     cannot tell

Configs without a reconstruction block are read as source: files, from model_dir,
rotation_files and topology_files at the top level (the original form).  The files are
sorted by path within each kind (as the original form's patterns sorted their matches), so
the same files reach the same trench-sample cache whichever source named them.
"""
import dataclasses
import glob
import os

SOURCES = ('files', 'gprm', 'plate_model_manager')
PMM_DATA_DIR = os.path.join('~', '.cache', 'slablet', 'plate_model_manager')
LEGACY_KEYS = ('model_dir', 'rotation_files', 'topology_files')
# files-source defaults for the lithosphere polygons (EarthByte folder layout)
DEFAULT_POLYGONS = {'continent_files': ['Continents/*.gpml'], 'craton_files': ['Cratons/*.gpml']}


@dataclasses.dataclass
class Reconstruction:
    label: str                 # what was asked for, e.g. 'gprm Muller2022 {NNR: False}'
    model_dir: str             # every relative pattern resolves here
    rotation_files: list
    topology_files: list
    continent_files: list      # None: not known (the lithosphere lid then stops with a message)
    craton_files: list
    frame: str                 # what the anchor plate is fixed to: 'mantle', 'spin axis' or 'unknown'
    frame_note: str = ''

    def summary(self):
        return {'label': self.label, 'model_dir': self.model_dir, 'frame': self.frame,
                'frame_note': self.frame_note,
                **{k: [os.path.relpath(f, self.model_dir) for f in getattr(self, k) or []]
                   for k in ('rotation_files', 'topology_files', 'continent_files', 'craton_files')}}


def find(model_dir, patterns, what):
    """Files matching the glob patterns (relative to model_dir, or absolute), sorted within each
    pattern, in pattern order.  Every pattern must match at least one file."""
    found = []
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(model_dir, os.path.expanduser(pat))))
        if not hits:
            raise SystemExit('{}: no file matches {} in {}'.format(what, pat, model_dir))
        found += hits
    return found


def block(cfg):
    """The config's reconstruction block (the original top-level form turned into source: files)."""
    if 'reconstruction' in cfg:
        clash = [k for k in LEGACY_KEYS if k in cfg]
        if clash:
            raise SystemExit('{} at the top level and a reconstruction block: put them in the block'.format(
                ', '.join(clash)))
        return dict(cfg['reconstruction'])
    missing = [k for k in LEGACY_KEYS if k not in cfg]
    if missing:
        raise SystemExit('no reconstruction block and no {} in the config'.format(', '.join(missing)))
    out = {'source': 'files', **{k: cfg[k] for k in LEGACY_KEYS}}
    out.update({k: cfg[k] for k in ('continent_files', 'craton_files', 'frame') if k in cfg})
    return out


_LOADED = {}


def load(cfg):
    """The Reconstruction for a run config (kept in memory, so repeated runs in one process
    resolve it once)."""
    b = block(cfg)
    anchor = int(cfg.get('anchor_plate_id', 0))
    key = repr(sorted(b.items(), key=lambda kv: kv[0])) + '|{}'.format(anchor)
    if key not in _LOADED:
        source = b.get('source', 'files')
        if source not in SOURCES:
            raise SystemExit('reconstruction source must be one of {}, not {!r}'.format(SOURCES, source))
        r = {'files': _from_files, 'gprm': _from_gprm, 'plate_model_manager': _from_pmm}[source](b, anchor)
        _apply_overrides(r, b)
        if r.frame == 'spin axis' and not b.get('allow_spin_axis_frame', False):
            raise SystemExit('{}: anchor plate {} is fixed to the spin axis ({}); slablets need a mantle frame. '
                             'Choose a mantle frame, or set allow_spin_axis_frame: true'.format(
                                 r.label, anchor, r.frame_note))
        _LOADED[key] = r
    return _LOADED[key]


def set_rotation_files(cfg, files):
    """Replace a config's rotation files (in its reconstruction block, or at the top level in the
    original form): how frame tests swap the frame."""
    (cfg['reconstruction'] if 'reconstruction' in cfg else cfg)['rotation_files'] = list(files)


def _apply_overrides(r, b):
    if 'rotation_files' in b and b.get('source', 'files') != 'files':
        r.rotation_files = find(r.model_dir, b['rotation_files'], 'rotation_files')
        r.frame, r.frame_note = 'unknown', 'rotation_files given in the config'
    for k in ('continent_files', 'craton_files'):
        if k in b and b.get('source', 'files') != 'files':
            setattr(r, k, find(r.model_dir, b[k], k))
    if 'frame' in b:
        r.frame, r.frame_note = b['frame'], 'given in the config'


def _from_files(b, anchor):
    d = os.path.expanduser(b['model_dir'])
    polys = {}
    for k, default in DEFAULT_POLYGONS.items():
        pats = b.get(k, default)
        try:
            polys[k] = find(d, pats, k)
        except SystemExit:
            if k in b:
                raise
            polys[k] = None                      # the default layout is not there
    return Reconstruction('files {}'.format(d), d, find(d, b['rotation_files'], 'rotation_files'),
                          find(d, b['topology_files'], 'topology_files'),
                          polys['continent_files'], polys['craton_files'], 'unknown', 'files named in the config')


def _common_dir(files):
    return os.path.commonpath([os.path.dirname(f) for f in files])


def _from_gprm(b, anchor):
    import gprm.datasets.Reconstructions as recs
    name = b['model']
    fetch = getattr(recs, 'fetch_' + name, None)
    if fetch is None:
        known = sorted(n[len('fetch_'):] for n in dir(recs) if n.startswith('fetch_'))
        raise SystemExit('gprm has no model {!r}; it has {}'.format(name, ', '.join(known)))
    options = dict(b.get('options') or {})
    m = fetch(**options)
    rot = sorted(os.path.abspath(f) for f in m.rotation_files)
    top = sorted(os.path.abspath(f) for f in m.dynamic_polygon_files)
    cont = sorted(os.path.abspath(f) for f in m.continent_polygons_files) or None
    if not top:
        raise SystemExit('gprm model {} has no topologies (dynamic polygons): slablet needs them'.format(name))
    frame, note = 'unknown', 'gprm records no frame for anchor plate {}'.format(anchor)
    for fr in getattr(m, 'reference_frames', None) or []:
        if int(fr['plate_id']) == anchor:
            frame, note = fr['reference'], fr['description']
    return Reconstruction(' '.join(['gprm', name] + ([str(options)] if options else [])), _common_dir(rot + top + (cont or [])),
                          rot, top, cont, None, frame, note)


def _from_pmm(b, anchor):
    from plate_model_manager import PlateModelManager
    data_dir = os.path.expanduser(b.get('data_dir', PMM_DATA_DIR))
    os.makedirs(data_dir, exist_ok=True)
    m = PlateModelManager().get_model(b['model'], data_dir=data_dir)
    if m is None:
        raise SystemExit('plate_model_manager has no model {!r}'.format(b['model']))
    rot = sorted(os.path.abspath(f) for f in m.get_rotation_model())
    top = sorted(os.path.abspath(f) for f in (m.get_topologies(return_none_if_not_exist=True) or []))
    if not top:
        raise SystemExit('plate_model_manager model {} has no topologies: slablet needs them'.format(b['model']))
    layers = m.get_avail_layers()
    cont = (sorted(os.path.abspath(f) for f in m.get_continental_polygons())
            if 'ContinentalPolygons' in layers else None)
    crat = sorted(os.path.abspath(f) for f in m.get_layer('Cratons')) if 'Cratons' in layers else None
    return Reconstruction('plate_model_manager {}'.format(b['model']), _common_dir(rot + top + (cont or []) + (crat or [])),
                          rot, top, cont, crat, 'unknown', 'plate_model_manager records no frame')
