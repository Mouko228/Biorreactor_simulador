"""
auditoria_sensibilidad.py
=========================
Auditoría, análisis de sensibilidad y simulación de incertidumbre (Monte Carlo)
del sistema híbrido de tratamiento de efluentes de baterías ion-litio.

Autor: IBM Bob — uso bajo supervisión de ingeniero especialista
Fecha: 2026-10-30

NOTAS DE AUDITORÍA (Fase 1) — se documentan al final del módulo.
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import json, time, traceback
from dataclasses import dataclass, field, asdict
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES FÍSICAS GLOBALES
# ─────────────────────────────────────────────────────────────────────────────
F_FARADAY  = 96485.0    # C/mol
V_L        = 1.0        # L  (volumen de reactor)
V_M3       = 1.0e-3     # m³
KPS_CAF2   = 3.9e-11    # Ksp CaF₂ [(mol/L)³]  — Stumm & Morgan (1996)
PKA_HF     = 3.17       # pKa HF, agua pura 25°C
PKA_HSO4   = 1.99       # pKa₂ H₂SO₄

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS NOMINALES (punto de operación de diseño)
# ─────────────────────────────────────────────────────────────────────────────
NOM = dict(
    # Efluente de entrada
    C_M_in     = 250.0,   # mg/L  metal (Cu²⁺ modelo)
    C_F_in     = 250.0,   # mg/L  flúor
    pH_in      = 2.0,     # adim
    C_EC_in    = 300.0,   # mg/L  etileno carbonato
    C_Li_in    = 200.0,   # mg/L  litio (conservado)
    # Metal
    M_g_mol    = 63.55,   # g/mol Cu
    z_metal    = 2,       # e⁻
    E0_cell    = 2.00,    # V
    # Electrodo C2
    A_cathode  = 0.044,   # m²
    k_mass     = 1.8e-5,  # m/s
    eta_pwm    = 0.85,    # fracción I_lim
    I_fixed    = 0.45,    # A
    # Precipitación C1
    k_ppt      = 1.5e4,   # (mol/L)⁻²/s
    C_F_eq     = 11.5,    # mg/L  residual equilibrio
    # Biosorción C3 (P. putida)
    q_max      = 2.2,     # mg metal / g biomasa
    K_L        = 0.35,    # L/mg
    m_bio      = 50.0,    # g/L  biomasa soporte
    k_ads      = 0.006,   # 1/s
    pH_pKa_eps = 4.8,     # pKa EPS
    # Biodegradación EC (Monod)
    mu_max     = 2.2e-4,  # 1/s  (0.79 h⁻¹)
    K_s_EC     = 80.0,    # mg/L
    Y_x        = 0.35,    # g_biomasa/g_EC
    K_d        = 5.0e-6,  # 1/s
    X0         = 0.15,    # g/L  biomasa libre inicial
    # Cinética precipitación química C3
    k_ppco     = 8e-5,    # 1/s  Cu(OH)₂ a pH>6.5
    # Control C3 (FF+PID — ganancias nominales)
    Kp_pid     = 1.2,
    Ki_pid     = 0.15,
    Kd_pid     = 0.05,
)

# ─────────────────────────────────────────────────────────────────────────────
# RANGOS DE INCERTIDUMBRE (Fase 2 & 3)
# Justificación bibliográfica resumida para cada parámetro
# ─────────────────────────────────────────────────────────────────────────────
RANGES = {
    # [min, max, distribución, justificación]
    'C_M_in':    [100,   500,  'uniform',     'rango real efluentes baterías Li-ion, Li et al. (2020)'],
    'C_F_in':    [80,    420,  'uniform',     'efluentes LiPF₆, Grützke et al. (2015)'],
    'pH_in':     [1.5,   3.0,  'uniform',     'rango ácido típico efluente, Diekmann et al. (2017)'],
    'A_cathode': [0.030, 0.060,'uniform',     '±30% tolerancia fabricación malla 316L'],
    'k_mass':    [1.0e-5,3.0e-5,'loguniform', 'rango Re–Sc–geom., Pletcher & Walsh (1990)'],
    'eta_pwm':   [0.75,  0.92, 'uniform',     'eficiencia faradaica PWM, Schlesinger (2010)'],
    'q_max':     [1.5,   3.5,  'uniform',     'isoterma Langmuir P. putida Cu²⁺, Pagnanelli et al. (2003)'],
    'K_L':       [0.15,  0.60, 'loguniform',  'constante Langmuir, Pagnanelli et al. (2003)'],
    'mu_max':    [1.5e-4,4.0e-4,'loguniform', 'μ_max P. putida, DSMZ media M9, Eberl et al. (2006)'],
    'K_s_EC':    [40.0,  150.0,'loguniform',  'semisaturación EC, estimado por analogía carbonatos'],
    'k_ppco':    [4e-5,  2e-4, 'loguniform',  'cinética Cu(OH)₂, Stumm & Morgan Tabla 6.1'],
    'C_F_eq':    [8.0,   16.0, 'uniform',     'residual CaF₂: fuerza iónica 0.03–0.08 M, solubilidad condicional'],
    'k_ppt':     [5e3,   5e4,  'loguniform',  'cinética CaF₂ 2° orden, Mpundu et al. (2019)'],
}

# Objetivos de diseño (para cumplimiento)
DESIGN_TARGETS = dict(
    C_M_final_max  = 1.2,    # mg/L  metal salida C3
    C_F_final_max  = 15.0,   # mg/L  flúor salida C1 (→C2→C3 sin cambio F)
    pH_final_min   = 6.5,
    pH_final_max   = 7.5,
    Li_min         = 199.0,  # mg/L  (conservado, tolerancia 0.5%)
    eta_F_min      = 0.75,   # eficiencia faradaica mínima aceptable
)


# ─────────────────────────────────────────────────────────────────────────────
# NÚCLEO DE SIMULACIÓN (independiente de Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def _alpha_HF(pH):
    return 1.0 / (1.0 + 10.0 ** (pH - PKA_HF))

def _beta_buffer(pH, C_F_mgL, C_SO4_mM=5.0):
    ln10 = np.log(10.0)
    bw   = ln10 * (10.0**(-pH) + 10.0**(pH - 14.0))
    aHF  = _alpha_HF(pH)
    CF_m = (C_F_mgL / 1000.0) / 19.0
    bHF  = ln10 * aHF * (1 - aHF) * CF_m
    CSO4 = C_SO4_mM * 1e-3
    aHS  = 1.0 / (1.0 + 10.0 ** (pH - PKA_HSO4))
    bSO4 = ln10 * aHS * (1 - aHS) * CSO4
    return max(bw + bHF + bSO4, 1e-6)


def sim_c1(p: dict) -> dict:
    """Cámara 1: neutralización + precipitación CaF₂."""
    t_span = (0.0, 900.0)   # 15 min
    n = 450
    t_eval = np.linspace(*t_span, n)

    alpha_so4_in = 1.0 / (1.0 + 10.0 ** (PKA_HSO4 - p['pH_in']))
    C_SO4_mM = max((10.0**(-p['pH_in'])) / (1.0 + alpha_so4_in) * 1000.0, 0.1)

    k_pump = 1.2e-4
    u_max  = 2.2e-4

    def odes(t, y):
        pH, CF, CM, intg, Ca2 = y
        pH  = np.clip(pH, 0, 14)
        CF  = max(CF, 0)
        Ca2 = max(Ca2, 0)

        err = 4.0 - pH
        if pH >= 4.0:
            d_intg = 0.0; u = 0.0
        else:
            u_raw  = k_pump * max(p['Kp_pid'] * err + 0.4 * intg, 0.0)
            u      = min(u_raw, u_max)
            d_intg = 0.0 if u_raw >= u_max else err

        beta  = _beta_buffer(pH, CF, C_SO4_mM)
        aHF   = _alpha_HF(pH)
        CF_eff = max(CF - p['C_F_eq'], 0.0)
        Fm    = (1 - aHF) * (CF_eff / 1000.0) / 19.0
        SS    = max(Ca2 * Fm**2 - KPS_CAF2, 0.0)
        dCaF2 = min(p['k_ppt'] * SS, Fm / 2.0)

        H_rel = 2.0 * aHF * dCaF2
        dpH   = (u - H_rel) / beta
        dCF   = -dCaF2 * 2.0 * 19.0 * 1000.0
        dCa2  = u / 2.0 - dCaF2
        return [dpH, dCF, 0.0, d_intg, dCa2]

    sol = solve_ivp(odes, t_span, [p['pH_in'], p['C_F_in'], p['C_M_in'], 0.0, 0.0],
                    t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8, max_step=5.0)

    CF1  = np.maximum(sol.y[1], 0.0)
    pH1  = np.clip(sol.y[0], 0.0, 4.0)
    CaF2 = np.maximum((p['C_F_in'] - CF1) * (78.08 / 38.0), 0.0)
    return dict(
        pH_out  = float(pH1[-1]),
        CF_out  = float(CF1[-1]),
        CM_out  = float(p['C_M_in']),   # metal no precipita a pH<5.5
        CaF2_out= float(CaF2[-1]),
        C_SO4_mM= C_SO4_mM,
        t       = sol.t / 60.0,
        pH_arr  = pH1,
        CF_arr  = CF1,
    )


def sim_c2(p: dict, c1: dict) -> dict:
    """Cámara 2: electrodeposición (modo PWM)."""
    t_span = (0.0, 3600.0)   # 60 min
    n = 1800
    t_eval = np.linspace(*t_span, n)

    M   = p['M_g_mol']
    z   = p['z_metal']
    ff  = M * 1000.0 / (z * F_FARADAY)   # mg/C

    def I_lim(C):
        return max(z * F_FARADAY * p['A_cathode'] * p['k_mass'] * max(C / M, 0.0), 1e-9)

    def odes(t, y):
        CM, m_dep, E_acc = y
        CM   = max(CM, 1e-6)
        IL   = I_lim(CM)
        I    = p['eta_pwm'] * IL
        dm   = p['eta_pwm'] * I * ff
        dCM  = -dm / V_L
        C_mL = max(CM / (M * 1000.0), 1e-12)
        Ec   = p['E0_cell'] - (0.0592 / z) * np.log10(C_mL)
        dE   = I * abs(Ec) / 3600.0
        return [dCM, dm, dE]

    sol = solve_ivp(odes, t_span, [c1['CM_out'], 0.0, 0.0],
                    t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8, max_step=2.0)

    CM2  = np.maximum(sol.y[0], 0.0)
    IL_arr = np.array([I_lim(c) for c in CM2])
    I_arr  = p['eta_pwm'] * IL_arr
    return dict(
        CM_out    = float(CM2[-1]),
        m_dep     = float(sol.y[1][-1]),
        energy_Wh = float(sol.y[2][-1]),
        I_max     = float(I_arr[0]),
        I_min     = float(I_arr[-1]),
        eta_mean  = float(p['eta_pwm']),
        t         = t_eval / 60.0,
        CM_arr    = CM2,
        I_arr     = I_arr,
    )


def sim_c3(p: dict, c1: dict, c2: dict) -> dict:
    """Cámara 3: biorreactor FF+PID."""
    t_span = (0.0, 2700.0)   # 45 min
    n = 1350
    t_eval = np.linspace(*t_span, n)
    t_pert = 20.0 * 60.0   # 20 min dentro de C3 → t=95 min global

    pH3_in  = c1['pH_out']
    C_SO4   = c1['C_SO4_mM']

    def phi(pH):
        return 1.0 / (1.0 + 10.0 ** (p['pH_pKa_eps'] - pH))

    def beta3(pH):
        ln10 = np.log(10.0)
        bw   = ln10 * (10.0**(-pH) + 10.0**(pH - 14.0))
        CSO4 = C_SO4 * 1e-3
        aHS  = 1.0 / (1.0 + 10.0 ** (pH - PKA_HSO4))
        bSO4 = ln10 * aHS * (1 - aHS) * CSO4
        return max(bw + bSO4 + 5e-4, 5e-4)

    k_pump = 4.5e-6; u_max = 1.8e-5

    def odes(t, y):
        pH, CM, CEC, X, intg, dfilt, q = y
        pH  = np.clip(pH, 0, 14)
        CM  = max(CM, 0); CEC = max(CEC, 0)
        X   = max(X, 0.01); q = max(q, 0)

        err  = 7.0 - pH
        beta = beta3(pH)
        q_ap = 1.4e-5 if abs(t - t_pert) < 30.0 else 0.0

        H_in  = 10.0**(-pH3_in)
        ff_OH = (H_in * 5e-6 / V_M3) + 0.85 * q_ap
        u_pid = k_pump * (p['Kp_pid'] * err + p['Ki_pid'] * (intg / 60.0) + p['Kd_pid'] * (dfilt * 60.0))
        u_raw = u_pid + ff_OH
        u_OH  = float(np.clip(u_raw, 0.0, u_max * 1.25))
        if pH >= 7.0 and err <= 0.0:
            u_OH = min(u_OH, q_ap)
        if (u_raw >= u_max * 1.25 and err > 0.0) or (u_raw <= 0 and err < 0) or abs(err) < 0.02:
            d_intg = 0.0
        else:
            d_intg = err

        dpH      = (u_OH - q_ap) / beta
        tau_d    = 5.0
        dd_filt  = (-dpH - dfilt) / tau_d

        q_eq  = p['q_max'] * p['K_L'] * CM / (1.0 + p['K_L'] * CM) * phi(pH)
        dq    = p['k_ads'] * (q_eq - q)
        dCM_b = -dq * p['m_bio']
        k_sm  = p['k_ppco'] / (1.0 + np.exp(-10.0 * (pH - 6.5)))
        dCM_c = -k_sm * CM
        dCM   = dCM_b + dCM_c

        mu   = p['mu_max'] * CEC / (p['K_s_EC'] + CEC + 1e-9)
        dX   = (mu - p['K_d']) * X
        dCEC = -mu * (X * 1000.0) / p['Y_x']

        return [dpH, dCM, dCEC, dX, d_intg, dd_filt, dq]

    y0  = [pH3_in, c2['CM_out'], p['C_EC_in'], p['X0'], 0.0, 0.0, 0.0]
    sol = solve_ivp(odes, t_span, y0, t_eval=t_eval,
                    method='RK45', rtol=1e-6, atol=1e-8, max_step=2.0)

    pH3 = np.clip(sol.y[0], 0, 14)
    CM3 = np.maximum(sol.y[1], 0)
    return dict(
        pH_out   = float(pH3[-1]),
        CM_out   = float(CM3[-1]),
        CEC_out  = float(np.maximum(sol.y[2], 0)[-1]),
        X_out    = float(sol.y[3][-1]),
        t        = t_eval / 60.0,
        pH_arr   = pH3,
        CM_arr   = CM3,
    )


def run_simulation(p_override: dict = None) -> dict:
    """Ejecuta las 3 cámaras con los parámetros dados. Retorna dict con KPIs."""
    p = {**NOM}
    if p_override:
        p.update(p_override)
    # Kp_pid necesario en C1 (reutilizamos para el PI de pH)
    if 'Kp_pid' not in p:
        p['Kp_pid'] = NOM['Kp_pid']

    t0 = time.time()
    try:
        c1 = sim_c1(p)
        c2 = sim_c2(p, c1)
        c3 = sim_c3(p, c1, c2)
        elapsed = time.time() - t0

        # ── KPIs ──────────────────────────────────────────────────────────────
        m_in    = p['C_M_in'] * V_L          # mg metal entrada
        m_dep   = c2['m_dep']                 # mg depositado en malla
        m_aq_C2 = c2['CM_out'] * V_L         # mg en solución tras C2
        m_fin   = c3['CM_out'] * V_L          # mg en solución salida C3

        eta_rec = (m_in - m_fin) / (m_in + 1e-12) * 100.0   # % recuperación total

        # cumplimiento
        T = DESIGN_TARGETS
        cum = dict(
            CM_ok  = c3['CM_out'] <= T['C_M_final_max'],
            CF_ok  = c1['CF_out'] <= T['C_F_final_max'],
            pH_ok  = T['pH_final_min'] <= c3['pH_out'] <= T['pH_final_max'],
            Li_ok  = True,   # Li no tiene vía de pérdida en el modelo
            eta_ok = c2['eta_mean'] >= T['eta_F_min'],
        )
        cum['all_ok'] = all(cum.values())

        return dict(
            # entradas
            **{k: p[k] for k in RANGES},
            # C1
            pH_c1_out   = c1['pH_out'],
            CF_c1_out   = c1['CF_out'],
            CaF2_mgL    = c1['CaF2_out'],
            F_abatido   = p['C_F_in'] - c1['CF_out'],
            # C2
            CM_c2_out   = c2['CM_out'],
            m_dep_mg    = m_dep,
            energy_Wh   = c2['energy_Wh'],
            I_max_A     = c2['I_max'],
            I_min_A     = c2['I_min'],
            eta_F_mean  = c2['eta_mean'],
            # C3
            pH_c3_out   = c3['pH_out'],
            CM_c3_out   = c3['CM_out'],
            CEC_c3_out  = c3['CEC_out'],
            Li_conserved= p['C_Li_in'],
            eta_rec_pct = eta_rec,
            # balance masa
            balance_err = abs((m_in - m_aq_C2) - m_dep),
            # cumplimiento
            **{f'cum_{k}': v for k, v in cum.items()},
            # meta
            elapsed_s   = elapsed,
            error       = None,
        )
    except Exception as e:
        return dict(error=str(e), tb=traceback.format_exc(), elapsed_s=time.time()-t0)


# ─────────────────────────────────────────────────────────────────────────────
# FASE 2 — ANÁLISIS DE SENSIBILIDAD (OAT: One-At-a-Time)
# ─────────────────────────────────────────────────────────────────────────────

def run_sensitivity() -> pd.DataFrame:
    """
    Varía cada parámetro en ±20%, ±40% alrededor del valor nominal.
    Registra variación relativa de cada KPI de salida.
    """
    base = run_simulation()
    if base['error']:
        raise RuntimeError(f"Simulación base falló: {base['error']}")

    kpis_out = ['CM_c3_out', 'CF_c1_out', 'pH_c3_out', 'energy_Wh',
                'eta_rec_pct', 'm_dep_mg', 'CaF2_mgL']

    records = []
    for param, (lo, hi, dist, _) in RANGES.items():
        nom_val = NOM.get(param)
        if nom_val is None:
            continue
        for delta_pct in [-40, -20, -10, +10, +20, +40]:
            factor = 1.0 + delta_pct / 100.0
            new_val = nom_val * factor
            # respetar límites físicos
            new_val = max(lo, min(hi, new_val))
            res = run_simulation({param: new_val})
            if res.get('error'):
                records.append(dict(param=param, delta_pct=delta_pct,
                                    param_value=new_val, error=res['error']))
                continue
            row = dict(param=param, delta_pct=delta_pct, param_value=new_val, error=None)
            for k in kpis_out:
                row[k] = res[k]
                bval   = base[k]
                row[f'd_{k}_pct'] = (res[k] - bval) / (abs(bval) + 1e-12) * 100.0
            records.append(row)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# FASE 3 — SIMULACIÓN DE INCERTIDUMBRE (MONTE CARLO, N=100)
# ─────────────────────────────────────────────────────────────────────────────

def sample_params(rng: np.random.Generator) -> dict:
    """Muestrea un conjunto de parámetros usando las distribuciones definidas."""
    p = {}
    for name, (lo, hi, dist, _) in RANGES.items():
        if dist == 'uniform':
            p[name] = float(rng.uniform(lo, hi))
        elif dist == 'loguniform':
            p[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            p[name] = float(rng.uniform(lo, hi))
    return p


def run_monte_carlo(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    results = []
    for i in range(n):
        params = sample_params(rng)
        res    = run_simulation(params)
        res['sim_id'] = i + 1
        results.append(res)
        if (i + 1) % 10 == 0:
            ok_so_far = sum(1 for r in results if r.get('cum_all_ok'))
            print(f"  MC {i+1:3d}/{n} | errores: {sum(1 for r in results if r.get('error'))} "
                  f"| cum_all_ok: {ok_so_far}/{i+1}")
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# FASE 4 & 5 — ESTADÍSTICA
# ─────────────────────────────────────────────────────────────────────────────

KPI_COLS = [
    'm_dep_mg', 'CM_c3_out', 'CaF2_mgL', 'CF_c1_out', 'CEC_c3_out',
    'Li_conserved', 'pH_c3_out', 'I_max_A', 'I_min_A', 'energy_Wh',
    'eta_rec_pct', 'eta_F_mean',
]
CUM_COLS = ['cum_CM_ok', 'cum_CF_ok', 'cum_pH_ok', 'cum_Li_ok',
            'cum_eta_ok', 'cum_all_ok']

def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df['error'].isna()].copy()
    rows  = []
    for col in KPI_COLS:
        if col not in valid.columns:
            continue
        s = valid[col].dropna()
        rows.append(dict(
            variable   = col,
            n          = len(s),
            mean       = s.mean(),
            median     = s.median(),
            std        = s.std(),
            min        = s.min(),
            p5         = s.quantile(0.05),
            p25        = s.quantile(0.25),
            p75        = s.quantile(0.75),
            p95        = s.quantile(0.95),
            max        = s.max(),
        ))
    return pd.DataFrame(rows)


def compute_compliance(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df['error'].isna()].copy()
    n     = len(valid)
    rows  = []
    labels = {
        'cum_CM_ok':  f'[M²⁺] salida C3 ≤ {DESIGN_TARGETS["C_M_final_max"]} mg/L',
        'cum_CF_ok':  f'[F⁻] salida C1 ≤ {DESIGN_TARGETS["C_F_final_max"]} mg/L',
        'cum_pH_ok':  f'pH C3 ∈ [{DESIGN_TARGETS["pH_final_min"]}, {DESIGN_TARGETS["pH_final_max"]}]',
        'cum_Li_ok':  f'Li⁺ conservado ≥ {DESIGN_TARGETS["Li_min"]} mg/L',
        'cum_eta_ok': f'η_F ≥ {DESIGN_TARGETS["eta_F_min"]*100:.0f}%',
        'cum_all_ok': 'TODOS los objetivos cumplidos',
    }
    for col, label in labels.items():
        if col in valid.columns:
            cnt = valid[col].sum()
            rows.append(dict(objetivo=label, n_cumple=int(cnt),
                             n_total=n, pct=cnt/n*100))
    return pd.DataFrame(rows)


def sensitivity_ranking(sens_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula la elasticidad media |ΔKPI%/Δparam%| por parámetro."""
    kpi_delta_cols = [c for c in sens_df.columns if c.startswith('d_') and c.endswith('_pct')]
    rows = []
    for param in sens_df['param'].unique():
        sub = sens_df[sens_df['param'] == param].dropna(subset=kpi_delta_cols[:1])
        if sub.empty:
            continue
        for kc in kpi_delta_cols:
            if kc not in sub.columns:
                continue
            kpi_name = kc[2:-4]   # strip d_ y _pct
            elasticity = (sub[kc].abs() / (sub['delta_pct'].abs() + 1e-12)).mean()
            rows.append(dict(param=param, kpi=kpi_name, elasticity_mean=elasticity))
    return pd.DataFrame(rows).sort_values('elasticity_mean', ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRO DE ERRORES DE AUDITORÍA (Fase 1)
# ─────────────────────────────────────────────────────────────────────────────

AUDIT_FINDINGS = [
    dict(id='A4',  camara='C1', severidad='Medio',
         descripcion='C_F_eq_mgL = 11.5 mg/L hardcodeado. No depende de [Ca²⁺] dosificado ni de fuerza iónica.',
         causa='Simplificación de equilibrio condicional de CaF₂',
         propuesta='Calcular C_F_eq = f(Ksp_cond, γ_Ca, γ_F, Ca2_acum) en cada paso de integración.'),
    dict(id='A6',  camara='C1', severidad='Medio',
         descripcion='Ca²⁺ inicial en solución = 0. Ignora Ca²⁺ natural del efluente (~0.5–2 mM en agua dura).',
         causa='Condición inicial y0[4]=0 en sim_c1',
         propuesta='Añadir parámetro C_Ca2_in [mM] con default 1.0 mM y sumarlo a la condición inicial.'),
    dict(id='B3',  camara='C2', severidad='Medio',
         descripcion='η_F_max = 0.85 aun cuando I_app << I_lim (régimen cinético). '
                     'Modelo no captura el aumento de eficiencia a baja corriente.',
         causa='Modelo de eficiencia simplificado con cota superior 0.85',
         propuesta='Usar η = η_max·(1 - exp(-I_L/I_app·k_kin)) para modelar régimen cinético.'),
    dict(id='B4',  camara='C2', severidad='Medio',
         descripcion='E_cell → ∞ cuando C_M → 0 por ecuación de Nernst con log(C_molL). '
                     'Produce energía sobreestimada al final del proceso.',
         causa='Sin cota inferior de potencial en la ecuación de Nernst',
         propuesta='Añadir límite: E_cell = min(E0_cell + 0.5, E_nernst). '
                   'Alternativamente usar C_min = 1 mg/L como límite operativo.'),
    dict(id='C3',  camara='C3', severidad='Medio',
         descripcion='max(X, 0.01) impide detectar colapso real de biomasa. '
                     'Si X_real < 0.01 g/L la simulación sigue con biomasa artificial.',
         causa='Cota inferior de seguridad numérica sobreescribe evento físico',
         propuesta='Registrar si X toca el límite inferior y marcarlo como advertencia de colapso.'),
    dict(id='C5',  camara='C3', severidad='Medio',
         descripcion='Filtro derivativo usa -dpH como proxy de de/dt. '
                     'En transitorios con perturbación ácida, de/dt ≠ -dpH.',
         causa='Error en la derivada de la señal de error bajo perturbación externa',
         propuesta='Usar d_filt como estado: dd_filt = (error - error_prev)/dt - d_filt)/tau_d '
                   'donde error_prev es el estado anterior de la señal de error.'),
    dict(id='C6',  camara='C3', severidad='Alto',
         descripcion='k_ppco(Co²⁺) = 1e-5 s⁻¹ implica precipitación de Co(OH)₂ a pH 7. '
                     'Sin embargo Ksp(Co(OH)₂)=5.92e-15 implica [Co²⁺]_eq ≈ 592 mmol/L a pH 7, '
                     'es decir Co(OH)₂ NO precipita en el rango de concentraciones del efluente.',
         causa='Parámetro k_ppco tomado por analogía con Cu sin verificar Ksp del Co',
         propuesta='Poner k_ppco(Co²⁺) = 0.0 a pH ≤ 9. La remoción de Co²⁺ depende '
                   'exclusivamente de la biosorción en este sistema.'),
]


def get_audit_df() -> pd.DataFrame:
    return pd.DataFrame(AUDIT_FINDINGS)


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("AUDITORÍA Y ANÁLISIS DE SENSIBILIDAD — SISTEMA BIOELECTROQUÍMICO")
    print("=" * 70)

    # ── Simulación nominal ────────────────────────────────────────────────────
    print("\n[1/5] Simulación nominal...")
    base = run_simulation()
    if base.get('error'):
        print(f"  ERROR en simulación base: {base['error']}")
    else:
        ok = lambda b: 'OK' if b else 'FAIL'
        print(f"  CM salida C3  : {base['CM_c3_out']:.3f} mg/L  [{ok(base['cum_CM_ok'])} obj <= {DESIGN_TARGETS['C_M_final_max']}]")
        print(f"  CF salida C1  : {base['CF_c1_out']:.2f} mg/L  [{ok(base['cum_CF_ok'])} obj <= {DESIGN_TARGETS['C_F_final_max']}]")
        print(f"  pH salida C3  : {base['pH_c3_out']:.3f}  [{ok(base['cum_pH_ok'])} obj 6.5-7.5]")
        print(f"  Energia C2    : {base['energy_Wh']:.4f} Wh")
        print(f"  Rec. total    : {base['eta_rec_pct']:.1f} %")
        print(f"  Balance masa  : error = {base['balance_err']:.3f} mg")
        print(f"  CUMPLE TODO   : {'SI' if base['cum_all_ok'] else 'NO'}")

    # ── Sensibilidad ──────────────────────────────────────────────────────────
    print("\n[2/5] Análisis de sensibilidad OAT...")
    sens_df = run_sensitivity()
    rank_df = sensitivity_ranking(sens_df)
    print(f"  Parámetros más influyentes (top 10 por elasticidad media):")
    top10 = rank_df.head(10)
    for _, row in top10.iterrows():
        print(f"    {row['param']:15s} → KPI {row['kpi']:20s}  elasticidad = {row['elasticity_mean']:.3f}")

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    print("\n[3/5] Monte Carlo N=100...")
    mc_df = run_monte_carlo(n=100, seed=42)
    n_err = mc_df['error'].notna().sum()
    print(f"  Simulaciones completadas: {len(mc_df) - n_err}/100  |  Errores: {n_err}")

    # ── Estadística ───────────────────────────────────────────────────────────
    print("\n[4/5] Estadística descriptiva...")
    stats_df = compute_stats(mc_df)
    print(stats_df[['variable','mean','std','p5','p95']].to_string(index=False))

    print("\n[5/5] Cumplimiento de objetivos...")
    comp_df = compute_compliance(mc_df)
    for _, row in comp_df.iterrows():
        print(f"  {row['objetivo']:60s}  {row['n_cumple']:3d}/{row['n_total']}  ({row['pct']:.1f}%)")

    # ── Exportar CSV ──────────────────────────────────────────────────────────
    mc_df.to_csv('resultados_montecarlo.csv', index=False)
    sens_df.to_csv('resultados_sensibilidad.csv', index=False)
    stats_df.to_csv('estadistica_kpis.csv', index=False)
    comp_df.to_csv('cumplimiento_objetivos.csv', index=False)
    get_audit_df().to_csv('hallazgos_auditoria.csv', index=False)

    print("\n✓ Archivos exportados:")
    for f in ['resultados_montecarlo.csv', 'resultados_sensibilidad.csv',
              'estadistica_kpis.csv', 'cumplimiento_objetivos.csv', 'hallazgos_auditoria.csv']:
        print(f"  {f}")
    print("\nDone.")
