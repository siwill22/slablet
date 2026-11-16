"""Show a slablet run in the Geode globe viewer, from a script or a Jupyter notebook.

    from slablet import geode
    archive, model_id = geode.export_run(cfg)          # the run's outputs -> a small Geode Archive
    geode.show(archive, model_id)                       # a local viewer: embedded in a notebook, and a link

Needs a Geode checkout (default ~/GIT/Geode, or $GEODE) with its viewer's node_modules installed,
and node on the PATH.  Nothing is written into the Geode checkout:

  build_viewer()  builds Geode's tomography viewer once into ~/.cache/slablet/geode-viewer, with its
                  Archive at the URL path /archive-notebook (Vite fixes that path at build time).
  export_run()    writes <run output_dir>/geode_archive with Geode's own prep scripts:
                    - the run's dVs volumes (every saved age), as model 'pd-<name>' ('pd-' is what
                      the viewer's "Kinematic model vs OPT1 vs tomography" preset looks for);
                    - coastlines in the run's own reconstruction and frame (rotated in the browser);
                    - links to Geode's surface image and, for comparison, REVEAL and SEMUCB-WM1
                      (present day) and optionally OPT1 (Muller 2022 only: it is in that frame).
  show()          serves the viewer and the Archive from a local web server (one per Archive) and
                  returns the viewer for a notebook cell, with a link to open it in its own tab.
                  A few lines added to the served page pick what it opens on: the run alone, or the
                  run beside the comparison models on a synced depth slice.
"""
import html
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

GEODE = Path(os.environ.get('GEODE', '~/GIT/Geode')).expanduser()
VIEWER_BUILD = Path('~/.cache/slablet/geode-viewer').expanduser()
ARCHIVE_URL = '/archive-notebook'
OPT1_ARCHIVE = Path('~/GIT/slablet/runs_real/geode_archive/models/opt1').expanduser()
PRESET_LABEL = 'Kinematic model vs OPT1 vs tomography'

_VITE_CONFIG = """// written by slablet.geode.build_viewer: Geode's tomography viewer only, without its public/ folder
export default {{
  root: {root!r},
  base: '/',
  publicDir: false,
  logLevel: 'warn',
  build: {{
    target: 'es2022',
    outDir: {out!r},
    emptyOutDir: true,
    chunkSizeWarningLimit: 4000,
    rollupOptions: {{ input: {index!r} }},
  }},
}};
"""


def _geode_commit():
    try:
        return subprocess.run(['git', '-C', str(GEODE), 'rev-parse', '--short', 'HEAD'], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def build_viewer(force=False):
    """Build Geode's tomography viewer once (about a minute); rebuilt when Geode's commit changes."""
    info = VIEWER_BUILD / 'build_info.json'
    commit = _geode_commit()
    if not force and info.exists() and json.loads(info.read_text()).get('geode_commit') == commit:
        return VIEWER_BUILD
    viewer = GEODE / 'viewer'
    if not (viewer / 'node_modules').is_dir():
        raise SystemExit('{} has no node_modules: run "npm install" there first'.format(viewer))
    VIEWER_BUILD.parent.mkdir(parents=True, exist_ok=True)
    config = VIEWER_BUILD.parent / 'geode-viewer.vite.config.mjs'
    config.write_text(_VITE_CONFIG.format(root=str(viewer), out=str(VIEWER_BUILD), index=str(viewer / 'index.html')))
    env = {**os.environ, 'VITE_ARCHIVE_BASE': ARCHIVE_URL}
    print('building the Geode viewer (Geode {}) into {} ...'.format(commit, VIEWER_BUILD), flush=True)
    subprocess.run(['npx', 'vite', 'build', '--config', str(config)], cwd=viewer, env=env, check=True)
    info.write_text(json.dumps({'geode_commit': commit, 'built': time.strftime('%Y-%m-%d %H:%M:%S')}))
    return VIEWER_BUILD


def _slug(text):
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')


def _prep(args):
    """Run one of Geode's prep scripts (they import each other, so from the checkout root)."""
    subprocess.run([sys.executable] + [str(a) for a in args], cwd=GEODE, check=True)


def _link(src, dst):
    dst = Path(dst)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.symlink_to(Path(src).resolve())


def export_run(cfg, variable=None, compare=('reveal', 'semucb'), opt1=False, archive_dir=None,
               coastline_age_step=1.0, clip=3.0):
    """Turn a finished run (its config dict) into a Geode Archive; returns (archive folder, model id).

    variable  the field to show (None: the run's dVs, which runs before 11 Nov 2026 hold as dVs_native where they
              saved it; see slablet.core.current_dvs).
    compare   Geode models to link in for comparison (present-day tomography from Geode's archive).
    opt1      also link Muller 2022 OPT1 (only meaningful for a run in Muller 2022's optimised frame).
    """
    import xarray as xr
    from slablet.gplates.models import load as load_reconstruction
    from slablet.paths import WORK

    run_dir = Path(WORK) / cfg['output_dir']
    files = sorted(run_dir.glob('mantle_temp_*Ma.nc'))
    if not files:
        raise SystemExit('no mantle_temp_*Ma.nc in {}: run the model first'.format(run_dir))
    with xr.open_dataset(files[0]) as ds:
        if variable is None:
            variable = 'dVs_native' if 'dVs_native' in ds else 'dVs'
        if variable not in ds:
            raise SystemExit('{} has no {}'.format(files[0], variable))
        nlon, nlat, ndepth = ds.sizes['lon'], ds.sizes['lat'], ds.sizes['depth']
    ages = [float(re.search(r'_(\d+)Ma', f.name).group(1)) for f in files]
    age_min, age_max = min(ages), max(ages)

    archive = Path(archive_dir) if archive_dir else run_dir / 'geode_archive'
    (archive / 'models').mkdir(parents=True, exist_ok=True)
    shutil.copy(GEODE / 'archive' / 'colormaps.json', archive / 'colormaps.json')
    _link(GEODE / 'archive' / 'surface', archive / 'surface')
    for m in compare:
        _link(GEODE / 'archive' / 'models' / m, archive / 'models' / m)
    if opt1:
        if not OPT1_ARCHIVE.is_dir():
            raise SystemExit('no OPT1 in {} (scripts/build_geode_archive.sh builds it)'.format(OPT1_ARCHIVE))
        _link(OPT1_ARCHIVE, archive / 'models' / 'opt1')

    recon = load_reconstruction(cfg)
    model_id = 'pd-' + _slug(Path(cfg['output_dir']).name)
    label = '{} ({}, {:.0f}-{:.0f} Ma)'.format(Path(cfg['output_dir']).name, recon.label, age_max, age_min)
    source = 'slablet kinematic mantle run {} ({}), {}; anchor plate {}'.format(
        cfg['output_dir'], variable, recon.label, cfg.get('anchor_plate_id', 0))
    _prep(['prep/prep_convection.py', '--input', run_dir, '--volume-var', variable, '--age-regex', r'_(\d+)Ma',
           '--age-min', age_min, '--age-max', age_max, '--id', model_id, '--name', label, '--source', source,
           '--var-id', 'vs', '--var-name', 'dVs (kinematic model)', '--units', '%', '--high-means', 'fast',
           '--colormap', 'RdBu', '--nlon', nlon, '--nlat', nlat + 1, '--ndepth', ndepth,
           '--default-clip', -clip, clip, '--out', archive, '--validate'])

    if recon.coastline_files:
        _prep(['prep/prep_coastlines.py', '--coastlines', *recon.coastline_files, '--rotations', *recon.rotation_files,
               '--age-min', age_min, '--age-max', age_max, '--age-step', coastline_age_step,
               '--anchor', cfg.get('anchor_plate_id', 0), '--out', archive / 'coastlines',
               '--source', '{} coastlines, anchor {}'.format(recon.label, cfg.get('anchor_plate_id', 0))])
    else:
        print('no coastline files for {}: the viewer shows no coastlines (give coastline_files in the '
              'reconstruction block)'.format(recon.label))
    _prep(['prep/build_archive_index.py', '--archive', archive, '--age-min', age_min, '--age-max', age_max])
    return archive, model_id


# ---------------------------------------------------------------------------------- serving

_SERVERS = {}          # site folder -> (server, port)

_BOOT = """<script type="module">
// added by slablet.geode.show: open on the run ({mode})
const want = {want};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
while (!(window.__geode && window.__geode.ready)) await sleep(200);
const g = window.__geode;
if (want.mode === 'compare') {{
  document.getElementById('preset-toggle').click();
  await sleep(100);
  const btn = [...document.querySelectorAll('.preset-btn')].find((b) => b.textContent.startsWith(want.preset));
  if (btn) btn.click(); else {{ document.getElementById('preset-toggle').click(); await g.setModel(want.model); }}
  for (let i = 0; i < 100 && !(g.globeCount() > 1 && g.modelIds()[0] === want.model); i++) await sleep(100);
  await sleep(300);
  if (want.depthKm !== null) g.setDepthSliceOn(0, {{ enabled: true, sinkingEnabled: false, depthKm: want.depthKm }});
}} else {{
  await g.setModel(want.model);
  if (want.depthKm !== null) {{ g.setSurfaceMode('none'); g.setDepthSliceOn(0, {{ enabled: true, sinkingEnabled: false, depthKm: want.depthKm }}); }}
}}
</script>
"""


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def _free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def site(archive, model_id, mode='compare', depth_km=1400.0):
    """The folder the server shows: the built viewer (linked), its page with the opening script added,
    and the Archive at ARCHIVE_URL."""
    build = build_viewer()
    archive = Path(archive).resolve()
    out = archive.parent / 'geode_site_{}'.format(mode)
    out.mkdir(exist_ok=True)
    for item in build.iterdir():
        if item.name not in ('index.html', 'build_info.json'):
            _link(item, out / item.name)
    _link(archive, out / ARCHIVE_URL.strip('/'))
    want = json.dumps({'mode': mode, 'model': model_id, 'preset': PRESET_LABEL,
                       'depthKm': None if depth_km is None else float(depth_km)})
    page = (build / 'index.html').read_text()
    (out / 'index.html').write_text(page.replace('</body>', _BOOT.format(mode=html.escape(mode), want=want) + '</body>'))
    return out


def serve(folder, port=None):
    """A local web server for the folder (reused if already running); returns its URL."""
    folder = str(Path(folder).resolve())
    if folder not in _SERVERS:
        port = port or _free_port()
        server = ThreadingHTTPServer(('127.0.0.1', port), partial(_Quiet, directory=folder))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        _SERVERS[folder] = (server, port)
    return 'http://127.0.0.1:{}/'.format(_SERVERS[folder][1])


def show(archive, model_id, mode='compare', depth_km=1400.0, height=720, open_tab=False):
    """The viewer for a notebook cell (an iframe and a link to open it in a new tab).

    mode      'compare': the run beside the comparison models in the Archive, one globe each, age and
              depth slice synced (the viewer's own preset); 'single': the run alone.
    depth_km  open on a depth slice at this depth (None: the preset's 2700 km, or the outer surface;
              the slice can be moved in the viewer).
    open_tab  also open it in the default browser.
    """
    url = serve(site(archive, model_id, mode, depth_km))
    if open_tab:
        webbrowser.open(url)
    try:
        from IPython.display import HTML
    except ImportError:
        print(url)
        return url
    return HTML('<p><a href="{0}" target="_blank">Open the viewer in a new tab</a> ({0})</p>'
                '<iframe src="{0}" style="width:100%;height:{1}px;border:0" allow="fullscreen"></iframe>'.format(
                    html.escape(url), int(height)))


def stop():
    """Stop every viewer server this session started."""
    for server, _ in _SERVERS.values():
        server.shutdown()
    _SERVERS.clear()
