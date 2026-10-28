"""
Test runner for app_simulador_3camaras.py (no Streamlit server needed).
"""
import sys, types, numpy as np

# ── Mock Streamlit ──────────────────────────────────────────────────────────
st_mock = types.ModuleType('streamlit')

class _Noop:
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def __call__(self, *a, **kw): return _Noop()
    def __iter__(self): return iter([])

noop = _Noop()

for attr in dir(st_mock):
    pass

for name in [
    'set_page_config','markdown','sidebar','header','subheader','slider',
    'selectbox','divider','button','columns','spinner','session_state',
    'cache_data','tabs','plotly_chart','caption','metric','download_button',
    'write', 'info', 'warning', 'error', 'success', 'title', 'text',
]:
    setattr(st_mock, name, noop)

class FakeColumns:
    def __getitem__(self, i): return noop
    def __iter__(self): return iter([noop, noop, noop, noop, noop, noop])

st_mock.columns = lambda *a, **kw: FakeColumns()
st_mock.session_state = {}

class FakeCacheData:
    def __call__(self, func=None, **kw):
        if func is None:
            return lambda f: f
        return func
st_mock.cache_data = FakeCacheData()

class FakeTabs:
    def __init__(self, names):
        self._names = names
    def __iter__(self):
        return iter([noop] * len(self._names))
st_mock.tabs = lambda names: FakeTabs(names)
st_mock.spinner = noop

sys.modules['streamlit'] = st_mock

# ── Import module ──────────────────────────────────────────────────────────
import app_simulador_3camaras as sim
print("Import: OK")

# ── Chamber 1 ──────────────────────────────────────────────────────────────
r1 = sim.simulate_chamber1(250, 250, 2.0)
ph1f  = r1['pH'][-1]
cf1f  = r1['C_F'][-1]
cm1f  = r1['C_M'][-1]
caf2f = r1['CaF2'][-1]
print(f"C1: pH_final={ph1f:.3f} (ref=4.0), F_final={cf1f:.1f} mg/L, M_final={cm1f:.1f} mg/L, CaF2={caf2f:.1f} mg/L")
assert 3.8 <= ph1f <= 4.2, f"pH C1 out of range: {ph1f}"
assert cf1f < 20.0, f"F residual too high: {cf1f}"
assert cm1f > 240.0, f"Metal lost in C1: {cm1f}"
print("C1: PASS")

# ── Chamber 2 ──────────────────────────────────────────────────────────────
r2 = sim.simulate_chamber2(cm1f, pH2_in=ph1f, I_fixed=0.45)
cm2p = r2['C_M_pwm'][-1]
cm2f = r2['C_M_fix'][-1]
md2p = r2['m_dep_pwm'][-1]
ep   = r2['energy_pwm'][-1]
ef   = r2['energy_fix'][-1]
print(f"C2 PWM: C_M_final={cm2p:.1f} mg/L, m_dep={md2p:.1f} mg, energy={ep:.3f} Wh")
print(f"C2 Fix: C_M_final={cm2f:.1f} mg/L, m_dep={r2['m_dep_fix'][-1]:.1f} mg, energy={ef:.3f} Wh")
# Mass balance
m_in  = cm1f * 1.0
m_aq  = cm2p * 1.0
diff  = abs(m_in - md2p - m_aq)
print(f"C2 mass balance: in={m_in:.1f} mg, deposited={md2p:.1f} mg, aq={m_aq:.1f} mg, diff={diff:.2f} mg")
assert diff < 10.0, f"Mass balance error C2: {diff:.2f} mg"
assert ep < ef, "PWM should use less energy than fixed current"
pct_save = (ef - ep) / (ef + 1e-9) * 100.0
print(f"C2 energy saving: {pct_save:.1f}%")
print("C2: PASS")

# ── Chamber 3 ──────────────────────────────────────────────────────────────
r3 = sim.simulate_chamber3(cm2p)
ph3_ol  = r3['open']['pH'][-1]
ph3_pid = r3['pid']['pH'][-1]
ph3_ff  = r3['ff_pid']['pH'][-1]
cm3_bio = r3['ff_pid']['C_M'][-1]
cm3_ab  = r3['abiotic']['C_M'][-1]
ec3f    = r3['ff_pid']['C_EC'][-1]
li3     = r3['C_Li']
print(f"C3 pH: OL={ph3_ol:.3f}, PID={ph3_pid:.3f}, FF+PID={ph3_ff:.3f} (ref=7.0)")
print(f"C3 M: bio={cm3_bio:.2f} mg/L, abiotic={cm3_ab:.2f} mg/L")
print(f"C3 EC_final={ec3f:.1f} mg/L, Li={li3:.0f} mg/L")
assert ph3_ff > ph3_ol, "FF+PID should achieve higher pH than open loop"
assert cm3_bio < cm3_ab, "Biosorption should remove more metal than abiotic"
assert li3 == 200.0, "Li must remain intact"
print("C3: PASS")

# ── Export ─────────────────────────────────────────────────────────────────
df = sim.build_export_df(r1, r2, r3, 'PWM Exponencial')
print(f"CSV export: {len(df)} rows, columns: {list(df.columns)}")
assert len(df) > 0
print("Export: PASS")

print("\n=== ALL TESTS PASSED ===")
