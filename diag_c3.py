
"""Diagnostic test for Chamber 3 pH trajectory."""
import sys, types, numpy as np

# ── Mock Streamlit ──────────────────────────────────────────────────────────
st_mock = types.ModuleType('streamlit')
class _Noop:
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def __call__(self, *a, **kw): return _Noop()
    def __iter__(self): return iter([])
noop = _Noop()
class FakeColumns:
    def __getitem__(self, i): return noop
    def __iter__(self): return iter([noop]*6)
st_mock.columns = lambda *a, **kw: FakeColumns()
st_mock.session_state = {}
class FakeCacheData:
    def __call__(self, func=None, **kw):
        if func is None: return lambda f: f
        return func
st_mock.cache_data = FakeCacheData()
class FakeTabs:
    def __init__(self, n): self._n = n
    def __iter__(self): return iter([noop]*self._n)
st_mock.tabs = lambda names: FakeTabs(len(names))
st_mock.spinner = noop
for attr in ['set_page_config','markdown','sidebar','header','subheader','slider',
             'selectbox','divider','button','spinner','caption','metric',
             'download_button','write','info','warning','error','success','title']:
    setattr(st_mock, attr, noop)
sys.modules['streamlit'] = st_mock

import app_simulador_3camaras as sim

# Run C3 directly with a known input
r3 = sim.simulate_chamber3(61.4)
tm = r3['t_min']

for mode in ['open', 'pid', 'ff_pid']:
    ph = r3[mode]['pH']
    cm = r3[mode]['C_M']
    # Show pH at key time points
    print(f"\n=== {mode} ===")
    for idx in [0, len(tm)//4, len(tm)//2, 3*len(tm)//4, -1]:
        print(f"  t={tm[idx]:.1f}min  pH={ph[idx]:.3f}  C_M={cm[idx]:.2f} mg/L")
