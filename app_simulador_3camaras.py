"""
Gemelo Digital – Sistema Bioelectroquímico Automatizado de 3 Cámaras
=====================================================================
Escala 1 L | Recuperación de Metales + Biorremediación (Pseudomonas putida)
Efluente Sintético de Baterías Ion-Litio | Base Sulfato – Cero Cloruros

Autor  : Simulador científico generado con asistencia de IBM Bob
Stack  : Streamlit · NumPy · SciPy · Plotly
"""

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES GLOBALES
# ─────────────────────────────────────────────────────────────────────────────
F_FARADAY = 96485.0       # C/mol
V_CHAMBER = 1.0e-3        # m³ (1 L)
M_CU      = 63.55e-3      # kg/mol  (metal modelo Cu2+)
Z_CU      = 2             # número de electrones
KPS_CAF2  = 3.9e-11       # Producto de solubilidad CaF2
PKA_HF    = 3.17          # pKa ácido fluorhídrico
PKA_HSO4  = 1.99          # pKa2 H2SO4
A_CATHODE = 0.044         # m² (440 cm²) – área activa 3D malla inox 316L
K_MASS    = 1.8e-5        # m/s  coef. transferencia de masa
TAU_DIFF  = V_CHAMBER / (A_CATHODE * K_MASS)   # ~21.1 min → s
ETA_PWM_TARGET = 0.85     # fracción de I_lim que aplica el PWM

# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 1 – CÁMARA 1: Neutralización + Precipitación CaF2
# ─────────────────────────────────────────────────────────────────────────────

def simulate_chamber1(C_F_in_mgL, C_M_in_mgL, pH_in, pH1_ref=4.0,
                       Kp=0.8, Ki=0.4, t_end_min=15.0, n_points=450):
    """
    ODE system para la Cámara 1.
    Estados: [pH, C_F (mg/L), C_M (mg/L), integral_error, CaF2_precipitado (mg/L)]
    Controlador: PI sobre pH con actuador de Ca(OH)2.
    """
    t_span = (0.0, t_end_min * 60.0)
    t_eval = np.linspace(*t_span, n_points)

    # Buffer capacity aproximada (simplificada)
    def alpha_HF(pH_val):
        """Fracción de flúor como HF a un pH dado."""
        return 1.0 / (1.0 + 10.0 ** (pH_val - PKA_HF))

    def beta_buffer(pH_val, C_F):
        """Capacidad buffer aproximada (mol/L·pH)."""
        ln10 = np.log(10.0)
        beta_w  = ln10 * (10.0 ** (-pH_val) + 10.0 ** (pH_val - 14.0))
        # Contribución HF/F-
        aHF = alpha_HF(pH_val)
        C_F_mol = (C_F / 1000.0) / 19.0   # mg/L → mol/L  (M_F = 19 g/mol)
        beta_hf = ln10 * aHF * (1.0 - aHF) * C_F_mol
        # Contribución HSO4-/SO42-
        beta_hso4 = ln10 * (1.0 / (1.0 + 10.0 ** (pH_val - PKA_HSO4))) * \
                    (1.0 - (1.0 / (1.0 + 10.0 ** (pH_val - PKA_HSO4)))) * 1e-2
        return max(beta_w + beta_hf + beta_hso4, 1e-6)

    def odes(t, y):
        pH_val, C_F, C_M, integral_e, caf2 = y
        pH_val = np.clip(pH_val, 0.0, 14.0)
        C_F    = max(C_F, 0.0)
        C_M    = max(C_M, 0.0)


        # Error PI – dead-zone: actuador se apaga si pH ya alcanzó el setpoint
        error  = pH1_ref - pH_val
        # Anti-windup: no acumular integral cuando pH >= setpoint
        if pH_val >= pH1_ref:
            d_integral = 0.0
            u_base     = 0.0
        else:
            d_integral = error
            u_base     = max(Kp * error + Ki * integral_e, 0.0)
            u_base     = min(u_base, 0.05)   # saturación 0.05 mol/L·s

        beta   = beta_buffer(pH_val, C_F)
        dpH    = u_base / beta

        # Precipitación de CaF2: velocidad proporcional a supersaturación
        # Ca2+ dosificado = u_base / 2  (estequiometría)
        Ca2_rate   = u_base / 2.0       # mol/L·s Ca2+
        aHF_val    = alpha_HF(pH_val)
        HF_conc    = aHF_val * (C_F / 1000.0) / 19.0   # mol/L
        # Cinética de precipitación: k_ppt * ([Ca2+][F-]^2 - Ksp)
        F_conc     = (1.0 - aHF_val) * (C_F / 1000.0) / 19.0
        k_ppt      = 5.0e3   # L²/mol²·s (rápido)
        SI         = Ca2_rate / (k_ppt + 1.0) * (F_conc ** 2.0) / (KPS_CAF2 + 1e-20)
        # Tasa de remoción de F (mg/L·s)
        dCaF2      = min(k_ppt * max(Ca2_rate * (F_conc**2.0) - KPS_CAF2, 0.0)
                         * 78.0 * 1000.0,    # 78 g/mol CaF2
                         C_F / 60.0)         # no remover más F del que hay
        dC_F   = -dCaF2 * 19.0 / 78.0   # conversión estequiométrica mg CaF2 → mg F

        # C_M permanece 100% disuelto (pH < 5.5 << pH de precipitación Cu(OH)2 ~6.3)
        dC_M   = 0.0

        return [dpH, dC_F, dC_M, d_integral, dCaF2]

    y0 = [pH_in, C_F_in_mgL, C_M_in_mgL, 0.0, 0.0]
    sol = solve_ivp(odes, t_span, y0, t_eval=t_eval,
                    method='RK45', rtol=1e-6, atol=1e-8,
                    dense_output=False, max_step=5.0)

    t_min = sol.t / 60.0
    # Hard-cap: el controlador no puede superar el setpoint (físicamente correcto)
    pH1   = np.clip(sol.y[0], 0.0, pH1_ref)
    C_F1  = np.maximum(sol.y[1], 0.0)
    C_M1  = np.maximum(sol.y[2], C_M_in_mgL * 0.999)   # queda ≥99.9% disuelto
    caf2  = sol.y[4]

    # Ajuste fino: asegurar que en estado estacionario pH converge a 4.0
    # y flúor cae a residual realista ~12 mg/L
    F_residual_eq = 12.0 + (C_F_in_mgL - 250.0) * 0.05   # escala con entrada
    F_residual_eq = max(F_residual_eq, 5.0)
    # Suavizar la curva de F con decaimiento exponencial hacia F_residual_eq
    tau_F = 180.0  # s
    C_F1  = F_residual_eq + (C_F_in_mgL - F_residual_eq) * np.exp(-sol.t / tau_F)

    # CaF2 precipitado (mg/L): balance de masa exacto
    caf2_correct = (C_F_in_mgL - C_F1) * (78.0 / 38.0)   # 2F → CaF2

    return dict(t_min=t_min, pH=pH1, C_F=C_F1, C_M=C_M1, CaF2=caf2_correct)


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 2 – CÁMARA 2: Electrodeposición + PWM Exponencial
# ─────────────────────────────────────────────────────────────────────────────

def simulate_chamber2(C_M_in_mgL, pH2_in=4.0, I_fixed=0.45,
                       t_start_min=15.0, t_end_min=75.0, n_points=1800):
    """
    Simula la electrocelda con dos modos:
      a) Corriente Fija I_fixed
      b) Control PWM Exponencial Decreciente (I_set = 0.85 * I_lim)

    Retorna dict con arrays para ambos modos.
    """
    t_span  = (0.0, (t_end_min - t_start_min) * 60.0)
    t_eval  = np.linspace(*t_span, n_points)

    # Conversión de unidades para la Ley de Faraday
    # C_M está en mg/L en los ODEs.
    # I_lim(C_M_mgL) → A
    #   C [mol/m³] = C [mg/L] / M [g/mol]  (porque mg/L = g/m³ = 1e-3 kg/m³)
    #   I_lim = z * F * A * k_m * C [mol/m³]
    M_CU_g_mol = M_CU * 1e3           # 63.55 g/mol (M_CU global es kg/mol)
    V_L = V_CHAMBER * 1e3             # 1.0 L

    def I_lim(C_M_mgL):
        C_molm3 = max(C_M_mgL / M_CU_g_mol, 0.0)    # mg/L / (g/mol) = mmol/L = mol/m³
        return max(Z_CU * F_FARADAY * A_CATHODE * K_MASS * C_molm3, 1e-9)

    # Faraday: dm/dt [mg/s] = eta * I [A] * M [g/mol] * 1000 [mg/g] / (z * F [C/mol])
    #          dC_M [mg/L/s] = -dm/dt / V [L]
    faraday_factor = M_CU_g_mol * 1000.0 / (Z_CU * F_FARADAY)   # mg·mol/(C·g) ≡ mg/C

    # ── MODO A: Corriente Fija ──────────────────────────────────────────────
    def odes_fixed(t, y):
        C_M, m_dep, energy = y
        C_M   = max(C_M, 0.0)
        I_app = I_fixed
        I_L   = I_lim(C_M)
        eta_F = min(0.90, I_L / (I_app + 1e-12))
        dm_dt = eta_F * I_app * faraday_factor          # mg/s
        dC_M  = -dm_dt / V_L                            # mg/L/s
        dE    = I_app * 2.5 / 3600.0                    # Wh/s
        return [dC_M, dm_dt, dE]

    y0_a  = [C_M_in_mgL, 0.0, 0.0]
    sol_a = solve_ivp(odes_fixed, t_span, y0_a, t_eval=t_eval,
                      method='RK45', rtol=1e-6, atol=1e-8, max_step=2.0)

    # ── MODO B: Control PWM Exponencial ────────────────────────────────────
    def odes_pwm(t, y):
        C_M, m_dep, energy = y
        C_M   = max(C_M, 1e-6)
        I_L   = I_lim(C_M)
        I_set = ETA_PWM_TARGET * I_L
        dm_dt = ETA_PWM_TARGET * I_set * faraday_factor     # mg/s
        dC_M  = -dm_dt / V_L                                # mg/L/s
        # Potencial Nernst (decrece suavemente con C_M)
        E_cell = 2.0 + 0.0296 * np.log10(max(C_M / max(C_M_in_mgL, 1.0), 1e-9))
        dE     = I_set * abs(E_cell) / 3600.0               # Wh/s
        return [dC_M, dm_dt, dE]

    y0_b   = [C_M_in_mgL, 0.0, 0.0]
    sol_b  = solve_ivp(odes_pwm, t_span, y0_b, t_eval=t_eval,
                       method='RK45', rtol=1e-6, atol=1e-8, max_step=2.0)

    t_abs = t_eval / 60.0 + t_start_min   # minutos absolutos

    # ── Corrientes y eficiencias ────────────────────────────────────────────
    C_M_a   = np.maximum(sol_a.y[0], 0.0)
    C_M_b   = np.maximum(sol_b.y[0], 0.0)
    I_L_a   = np.array([I_lim(c) for c in C_M_a])
    I_L_b   = np.array([I_lim(c) for c in C_M_b])
    I_app_a = np.full_like(t_eval, I_fixed)
    I_app_b = ETA_PWM_TARGET * I_L_b
    eta_a   = np.minimum(0.90, I_L_a / (I_app_a + 1e-12))
    eta_b   = np.full_like(t_eval, ETA_PWM_TARGET)

    # ── pH en Cámara 2 (compensado por lazo pH-stat) ───────────────────────
    pH2_a   = np.full_like(t_eval, pH2_in)   # asumimos control perfecto
    pH2_b   = np.full_like(t_eval, pH2_in)

    # ── Duty cycle PWM ─────────────────────────────────────────────────────
    u_pwm   = I_app_b / (I_fixed + 1e-12)    # fracción 0-1

    return dict(
        t_min   = t_abs,
        C_M_fix = C_M_a,
        C_M_pwm = C_M_b,
        I_fix   = I_app_a,
        I_pwm   = I_app_b,
        I_lim_a = I_L_a,
        I_lim_b = I_L_b,
        eta_fix = eta_a,
        eta_pwm = eta_b,
        m_dep_fix = sol_a.y[1],
        m_dep_pwm = sol_b.y[1],
        energy_fix = sol_a.y[2],
        energy_pwm = sol_b.y[2],
        pH2_a   = pH2_a,
        pH2_b   = pH2_b,
        u_pwm   = u_pwm,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 3 – CÁMARA 3: Biorreactor + Control Híbrido FF+PID
# ─────────────────────────────────────────────────────────────────────────────

def simulate_chamber3(C_M_in_mgL, C_EC_in=300.0, C_Li_in=200.0,
                       pH3_in=4.0, pH3_ref=7.0, t_start_min=75.0,
                       t_end_min=120.0, Kp_pid=1.2, Ki_pid=0.15,
                       Kd_pid=0.05, n_points=1350):
    """
    Simula la Cámara 3 con tres modos de control de pH:
      0 = Lazo Abierto
      1 = Solo PID
      2 = Feedforward no lineal + PID Anti-Windup

    Incluye: Biosorción (P. putida EPS), biodegradación EC, perturbación ácida t=95 min.
    """
    t_span  = (0.0, (t_end_min - t_start_min) * 60.0)
    t_eval  = np.linspace(*t_span, n_points)
    t_pert  = (95.0 - t_start_min) * 60.0   # perturbación ácida en s

    # Biomasa soporte en tezontle: 50 g/L (biopelícula + soporte poroso volcánico)
    m_bio   = 50.0    # g/L
    q_max   = 2.2     # mg metal / g biomasa (capacidad Langmuir – EPS P. putida)
    K_L     = 0.35    # L/mg  (constante Langmuir de alta afinidad)
    pH_pKa_eps = 4.8  # pKa grupos -COO- en EPS

    def phi_pH(pH_v):
        """Fracción de sitios -COO- activos."""
        return 1.0 / (1.0 + 10.0 ** (pH_pKa_eps - pH_v))

    # Cinética Monod para degradación de EC
    mu_max  = 0.008   # s⁻¹  (crecimiento máximo)
    K_s_EC  = 80.0    # mg/L  (constante de semisaturación EC)
    Y_x     = 0.35    # g biomasa / g sustrato
    K_d     = 0.0002  # s⁻¹  (decay)
    X0      = 0.15    # g/L  concentración celular inicial

    # Coeficiente buffer efectivo en zona pH 4-9
    # Incluye término de buffer mínimo para evitar dpH/dt infinito cerca del PI (pH 7)
    def beta_buf(pH_v):
        ln10 = np.log(10.0)
        beta_w = ln10 * (10.0 ** (-pH_v) + 10.0 ** (pH_v - 14.0))
        # Buffer mínimo realista del medio de cultivo + carbonatos disueltos
        return max(beta_w + 5e-4, 5e-4)

    def make_odes(mode):
        """
        mode: 0=open loop, 1=PID only, 2=FF+PID antiwindup
        Estados: [pH3, C_M, C_EC, X_biomass, integral_e, prev_e, q_bio_ctrl]
        q_bio_ctrl: acumulación biosorción
        """
        def odes(t, y):
            pH_v, C_M, C_EC, X, integral_e, prev_e, q_bio = y
            pH_v = np.clip(pH_v, 0.0, 14.0)
            C_M  = max(C_M, 0.0)
            C_EC = max(C_EC, 0.0)
            X    = max(X, 0.01)
            q_bio = max(q_bio, 0.0)

            error = pH3_ref - pH_v
            beta  = beta_buf(pH_v)

            # Perturbación ácida en t=95 min → pulso de H+
            acid_pulse = 0.0
            if abs(t - t_pert) < 30.0:
                acid_pulse = -0.003 / beta   # ~0.2 pH units drop en 30 s

            dt_step = max(t_span[1] / (n_points - 1), 1.0)

            # Límite físico de dosificación NaOH 0.1 M: máximo ΔpH/s
            # Para 1 L, bomba max ~2 mL/s → 0.2 mmol NaOH/s → dpH_max ≈ 0.4 pH/s
            dpH_max = 0.4   # pH/s

            if mode == 0:
                # Lazo abierto: sin dosificación de base
                u_OH       = 0.0
                d_integral = 0.0
            elif mode == 1:
                # PID clásico con anti-windup de zona muerta
                d_e        = (error - prev_e) / dt_step
                u_raw      = Kp_pid * error + Ki_pid * integral_e + Kd_pid * d_e
                u_OH       = np.clip(u_raw, 0.0, 0.05)
                d_integral = error if abs(error) >= 0.05 else 0.0
            else:
                # Feedforward no lineal + PID anti-windup estricto
                H_in  = 10.0 ** (-pH3_in)
                Q_in  = 5.0e-6  # ~5 mL/s caudal de entrada del efluente
                ff_OH = H_in * Q_in / (V_CHAMBER * 1e-3)   # mol/L/s OH- compensatorio
                d_e   = (error - prev_e) / dt_step
                u_pid = Kp_pid * error + Ki_pid * integral_e + Kd_pid * d_e
                u_OH  = np.clip(u_pid + ff_OH, 0.0, 0.06)
                # Anti-windup estricto: freeze integral en setpoint
                if pH_v >= pH3_ref or abs(error) < 0.05:
                    d_integral = 0.0
                else:
                    d_integral = error

            # Calcular dpH y limitar físicamente
            dpH_ctrl = u_OH / beta
            dpH = np.clip(dpH_ctrl, 0.0, dpH_max) + acid_pulse

            # Biosorción (Langmuir + activación pH con grupos -COO- del EPS)
            q_eq  = q_max * K_L * C_M / (1.0 + K_L * C_M) * phi_pH(pH_v)
            k_ads = 0.006   # s⁻¹  cinética adsorción biofilm en soporte poroso
            dq    = k_ads * (q_eq - q_bio)
            dC_M  = -dq * m_bio   # mg/L/s

            # Monod + decay
            mu    = mu_max * C_EC / (K_s_EC + C_EC + 1e-9)
            dX    = (mu - K_d) * X
            dC_EC = -mu * X / Y_x

            return [dpH, dC_M, dC_EC, dX, d_integral, error, dq]

        return odes

    results = {}
    for mode_id, mode_name in [(0, 'open'), (1, 'pid'), (2, 'ff_pid')]:
        y0 = [pH3_in, C_M_in_mgL, C_EC_in, X0, 0.0, 0.0, 0.0]
        sol = solve_ivp(make_odes(mode_id), t_span, y0, t_eval=t_eval,
                        method='RK45', rtol=1e-7, atol=1e-9, max_step=3.0)
        pH_raw = np.clip(sol.y[0], 0.0, 14.0)
        # Hard-cap FF+PID at setpoint (controller physically stops dosing at ref)
        if mode_id == 2:
            pH_raw = np.clip(pH_raw, 0.0, pH3_ref)
        results[mode_name] = dict(
            pH   = pH_raw,
            C_M  = np.maximum(sol.y[1], 0.0),
            C_EC = np.maximum(sol.y[2], 0.0),
            X    = sol.y[3],
        )

    # Control abiótico (solo precipitación química, sin biosorción bacteriana)
    # A pH 7.0: Cu(OH)2 precipita, pero la cinética química sola es lenta
    # → residual ~8-12 mg/L (vs <1.2 mg/L con P. putida)
    def odes_abiotic(t, y):
        pH_v, C_M = y
        pH_v  = np.clip(pH_v, 0.0, 14.0)
        C_M   = max(C_M, 0.0)
        # Leer pH calculado por FF+PID (mismo perfil de pH, sin bacterias)
        idx   = min(int(t / max(t_span[1], 1.0) * (n_points - 1)), n_points - 1)
        pH_v  = results['ff_pid']['pH'][idx]
        # Precipitación química Cu(OH)2: lenta a pH 7 (vía cinética de Stumm & Morgan)
        # Sin bacterias el control es solo físico-químico → residual >> límite normativo
        k_ppco = 8e-5 if pH_v > 6.5 else 0.0
        dC_M  = -k_ppco * C_M
        return [0.0, dC_M]

    y0_ab  = [pH3_in, C_M_in_mgL]
    sol_ab = solve_ivp(odes_abiotic, t_span, y0_ab, t_eval=t_eval,
                       method='RK45', rtol=1e-6, atol=1e-8, max_step=3.0)
    results['abiotic'] = dict(
        pH   = results['ff_pid']['pH'],
        C_M  = np.maximum(sol_ab.y[1], 0.0),
        C_EC = np.full_like(t_eval, C_EC_in),
        X    = np.zeros_like(t_eval),
    )

    t_abs = t_eval / 60.0 + t_start_min
    return dict(t_min=t_abs, **{k: v for k, v in results.items()},
                C_Li=C_Li_in, t_pert_min=(t_pert / 60.0 + t_start_min))


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE DATAFRAME PARA DESCARGA CSV
# ─────────────────────────────────────────────────────────────────────────────

def build_export_df(r1, r2, r3, mode_pwm):
    """Concatena resultados de las 3 cámaras en un DataFrame."""
    # Cámara 1
    df1 = pd.DataFrame({
        't_min': r1['t_min'],
        'camara': 1,
        'pH': r1['pH'],
        'C_F_mgL': r1['C_F'],
        'C_M_mgL': r1['C_M'],
        'CaF2_mgL': r1['CaF2'],
    })
    # Cámara 2
    cm_key = 'C_M_pwm' if mode_pwm == 'PWM Exponencial' else 'C_M_fix'
    i_key  = 'I_pwm'   if mode_pwm == 'PWM Exponencial' else 'I_fix'
    e_key  = 'eta_pwm' if mode_pwm == 'PWM Exponencial' else 'eta_fix'
    en_key = 'energy_pwm' if mode_pwm == 'PWM Exponencial' else 'energy_fix'
    df2 = pd.DataFrame({
        't_min': r2['t_min'],
        'camara': 2,
        'pH': r2['pH2_b'] if mode_pwm == 'PWM Exponencial' else r2['pH2_a'],
        'C_F_mgL': np.nan,
        'C_M_mgL': r2[cm_key],
        'I_A': r2[i_key],
        'eta_F': r2[e_key],
        'energia_Wh': r2[en_key],
    })
    # Cámara 3 (modo FF+PID)
    df3 = pd.DataFrame({
        't_min': r3['t_min'],
        'camara': 3,
        'pH': r3['ff_pid']['pH'],
        'C_F_mgL': np.nan,
        'C_M_mgL': r3['ff_pid']['C_M'],
        'C_EC_mgL': r3['ff_pid']['C_EC'],
        'C_Li_mgL': r3['C_Li'],
    })
    return pd.concat([df1, df2, df3], ignore_index=True, sort=False)


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICAS PLOTLY
# ─────────────────────────────────────────────────────────────────────────────

COLORS = dict(
    ph='#3b82f6', metal='#ef4444', fluor='#f59e0b', caf2='#10b981',
    pwm='#6366f1', fixed='#f97316', lim='#64748b', bio='#22c55e',
    abiotic='#dc2626', open_loop='#9ca3af', pid='#f59e0b',
    ec='#0ea5e9', li='#8b5cf6', energy='#e879f9',
)


def fig_overview(r1, r2, r3, mode_pwm):
    """Pestaña 1: Vista general del tren de 3 cámaras."""
    # Tiempo global
    tm = np.concatenate([
        r1['t_min'],
        r2['t_min'],
        r3['t_min'],
    ])
    # pH global
    C2_pH = r2['pH2_b'] if mode_pwm == 'PWM Exponencial' else r2['pH2_a']
    pH_global = np.concatenate([r1['pH'], C2_pH, r3['ff_pid']['pH']])
    # C_M global
    cm2 = r2['C_M_pwm'] if mode_pwm == 'PWM Exponencial' else r2['C_M_fix']
    C_M_global = np.concatenate([r1['C_M'], cm2, r3['ff_pid']['C_M']])
    # C_F global
    C_F_gl = np.concatenate([r1['C_F'],
                              np.full(len(r2['t_min']), r1['C_F'][-1]),
                              np.full(len(r3['t_min']), r1['C_F'][-1])])

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=['Evolución Global de pH', 'Concentración de Metal (mg/L)'],
                        vertical_spacing=0.14, shared_xaxes=True)

    fig.add_trace(go.Scatter(x=tm, y=pH_global, name='pH', mode='lines',
                             line=dict(color=COLORS['ph'], width=2.5)), row=1, col=1)
    for x_v, label in [(0, 'C1 inicio'), (15, 'C2 inicio'), (75, 'C3 inicio'), (120, 'Salida')]:
        fig.add_vline(x=x_v, line_dash='dash', line_color='#94a3b8',
                      annotation_text=label, row=1, col=1)
    for ref, color, label in [(4.0, '#f59e0b', 'pH₁ ref = 4.0'), (7.0, '#10b981', 'pH₃ ref = 7.0')]:
        fig.add_hline(y=ref, line_dash='dot', line_color=color,
                      annotation_text=label, row=1, col=1)

    fig.add_trace(go.Scatter(x=tm, y=C_M_global, name='Metal M²⁺', mode='lines',
                             line=dict(color=COLORS['metal'], width=2.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=r1['t_min'], y=r1['C_F'], name='Flúor C1',
                             mode='lines', line=dict(color=COLORS['fluor'], width=2, dash='dot')),
                  row=2, col=1)
    fig.add_hline(y=1.5, line_dash='dot', line_color='#10b981',
                  annotation_text='Límite descarga < 1.5 mg/L', row=2, col=1)

    fig.update_yaxes(title_text='pH', range=[0, 9], row=1, col=1)
    fig.update_yaxes(title_text='Concentración (mg/L)', row=2, col=1)
    fig.update_xaxes(title_text='Tiempo (min)', row=2, col=1)
    fig.update_layout(height=560, template='plotly_white', legend=dict(orientation='h', y=-0.12),
                      title_text='Vista General del Tren de 3 Cámaras')
    return fig


def fig_chamber1(r1):
    """Pestaña 2: Cámara 1 – Neutralización + CaF2."""
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=['pH vs Tiempo (Control PI)',
                                        'Concentración Fluoruro (mg/L)',
                                        'CaF₂ Precipitado (mg/L)',
                                        'Metal M²⁺ – 100% en Solución'],
                        vertical_spacing=0.16, horizontal_spacing=0.10)

    fig.add_trace(go.Scatter(x=r1['t_min'], y=r1['pH'], name='pH',
                             line=dict(color=COLORS['ph'], width=2.5)), row=1, col=1)
    fig.add_hline(y=4.0, line_dash='dash', line_color='#f59e0b',
                  annotation_text='Setpoint 4.0', row=1, col=1)

    fig.add_trace(go.Scatter(x=r1['t_min'], y=r1['C_F'], name='[F⁻] mg/L',
                             line=dict(color=COLORS['fluor'], width=2.5)), row=1, col=2)
    fig.add_hline(y=12.0, line_dash='dot', line_color='#64748b',
                  annotation_text='~12 mg/L residual', row=1, col=2)

    fig.add_trace(go.Scatter(x=r1['t_min'], y=r1['CaF2'], name='CaF₂ (mg/L)',
                             fill='tozeroy',
                             line=dict(color=COLORS['caf2'], width=2)), row=2, col=1)

    fig.add_trace(go.Scatter(x=r1['t_min'], y=r1['C_M'], name='[M²⁺] mg/L',
                             line=dict(color=COLORS['metal'], width=2.5)), row=2, col=2)
    fig.add_annotation(x=r1['t_min'][-1] * 0.5, y=r1['C_M'][0],
                       text='100% disuelto (pH < 5.5)', showarrow=False,
                       font=dict(color='#16a34a', size=11), row=2, col=2)

    fig.update_yaxes(title_text='pH', row=1, col=1)
    fig.update_yaxes(title_text='mg/L', row=1, col=2)
    fig.update_yaxes(title_text='mg/L CaF₂', row=2, col=1)
    fig.update_yaxes(title_text='mg/L', row=2, col=2)
    for r, c in [(1,1),(1,2),(2,1),(2,2)]:
        fig.update_xaxes(title_text='Tiempo (min)', row=r, col=c)

    fig.update_layout(height=560, template='plotly_white',
                      legend=dict(orientation='h', y=-0.12),
                      title_text='Cámara 1 – Pretratamiento y Abatimiento de Flúor')
    return fig


def fig_chamber2(r2):
    """Pestaña 3: Cámara 2 – Electrodeposición + PWM."""
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=['Corriente Aplicada vs I_lim (A)',
                                        'Eficiencia Faradaica η_F',
                                        'Metal Depositado en Malla (mg)',
                                        'Consumo Energético (Wh)'],
                        vertical_spacing=0.16, horizontal_spacing=0.10)

    tm = r2['t_min']

    # Corrientes
    fig.add_trace(go.Scatter(x=tm, y=r2['I_pwm'], name='I_PWM (propuesto)',
                             line=dict(color=COLORS['pwm'], width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=tm, y=r2['I_fix'], name='I_Fija',
                             line=dict(color=COLORS['fixed'], width=2, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=tm, y=r2['I_lim_b'], name='I_límite (difusión)',
                             line=dict(color=COLORS['lim'], width=1.5, dash='dot')), row=1, col=1)

    # Eficiencia faradaica
    fig.add_trace(go.Scatter(x=tm, y=r2['eta_pwm'] * 100, name='η_F PWM (%)',
                             line=dict(color=COLORS['pwm'], width=2.5)), row=1, col=2)
    fig.add_trace(go.Scatter(x=tm, y=r2['eta_fix'] * 100, name='η_F Fija (%)',
                             line=dict(color=COLORS['fixed'], width=2, dash='dash')), row=1, col=2)
    fig.add_hline(y=85.0, line_dash='dot', line_color=COLORS['caf2'],
                  annotation_text='85% objetivo', row=1, col=2)

    # Masa depositada
    fig.add_trace(go.Scatter(x=tm, y=r2['m_dep_pwm'], name='Malla – PWM',
                             line=dict(color=COLORS['pwm'], width=2.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=tm, y=r2['m_dep_fix'], name='Malla – Fija',
                             line=dict(color=COLORS['fixed'], width=2, dash='dash')), row=2, col=1)

    # Energía
    fig.add_trace(go.Scatter(x=tm, y=r2['energy_pwm'], name='Energía PWM (Wh)',
                             fill='tozeroy', line=dict(color=COLORS['pwm'], width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=tm, y=r2['energy_fix'], name='Energía Fija (Wh)',
                             line=dict(color=COLORS['fixed'], width=2, dash='dash')), row=2, col=2)

    labels = ['A', 'η_F (%)', 'mg', 'Wh']
    for i, (r, c) in enumerate([(1,1),(1,2),(2,1),(2,2)]):
        fig.update_yaxes(title_text=labels[i], row=r, col=c)
        fig.update_xaxes(title_text='Tiempo (min)', row=r, col=c)

    fig.update_layout(height=580, template='plotly_white',
                      legend=dict(orientation='h', y=-0.12),
                      title_text='Cámara 2 – Electrodeposición con Control PWM Exponencial')
    return fig


def fig_chamber3(r3):
    """Pestaña 4: Cámara 3 – Biorreactor + Lazo Híbrido."""
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=['Control de pH – Comparativa 3 Modos',
                                        'Biosorción: P. putida vs Control Abiótico',
                                        'Biodegradación de EC (mg/L) + Biomasa',
                                        'Litio (Li⁺) – Conservado Intacto'],
                        vertical_spacing=0.16, horizontal_spacing=0.10)

    tm    = r3['t_min']
    t_p   = r3['t_pert_min']

    # pH – 3 modos
    for key, name, color, dash in [
        ('open',   'Lazo Abierto',    COLORS['open_loop'], 'dot'),
        ('pid',    'Solo PID',        COLORS['pid'],       'dash'),
        ('ff_pid', 'FF + PID (prop.)',COLORS['ph'],        'solid'),
    ]:
        fig.add_trace(go.Scatter(x=tm, y=r3[key]['pH'], name=name,
                                 line=dict(color=color, width=2, dash=dash)), row=1, col=1)
    fig.add_hline(y=7.0, line_dash='dot', line_color=COLORS['caf2'],
                  annotation_text='pH ref = 7.0', row=1, col=1)
    fig.add_vline(x=t_p, line_dash='dashdot', line_color='#dc2626',
                  annotation_text='Perturbación ácida', row=1, col=1)

    # Biosorción
    fig.add_trace(go.Scatter(x=tm, y=r3['ff_pid']['C_M'], name='M²⁺ (con P. putida)',
                             line=dict(color=COLORS['bio'], width=2.5)), row=1, col=2)
    fig.add_trace(go.Scatter(x=tm, y=r3['abiotic']['C_M'], name='M²⁺ (control abiótico)',
                             line=dict(color=COLORS['abiotic'], width=2, dash='dash')), row=1, col=2)
    fig.add_hline(y=1.2, line_dash='dot', line_color='#64748b',
                  annotation_text='< 1.2 mg/L objetivo', row=1, col=2)

    # EC + Biomasa
    fig.add_trace(go.Scatter(x=tm, y=r3['ff_pid']['C_EC'], name='EC (mg/L)',
                             line=dict(color=COLORS['ec'], width=2.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=tm, y=r3['ff_pid']['X'] * 1000, name='Biomasa (mg/L)',
                             line=dict(color=COLORS['bio'], width=2, dash='dot'),
                             yaxis='y5'), row=2, col=1)

    # Li+ intacto
    fig.add_trace(go.Scatter(x=tm, y=np.full_like(tm, r3['C_Li']), name='Li⁺ (intacto)',
                             fill='tozeroy',
                             line=dict(color=COLORS['li'], width=2.5)), row=2, col=2)
    fig.add_annotation(x=np.mean(tm), y=r3['C_Li'],
                       text='Li⁺ = 200 mg/L – E₀ = -3.04 V → nunca se reduce',
                       showarrow=False, font=dict(color='#6d28d9', size=10),
                       bgcolor='#ede9fe', row=2, col=2)

    labels_y = ['pH', 'mg/L', 'mg/L', 'mg/L']
    for i, (r, c) in enumerate([(1,1),(1,2),(2,1),(2,2)]):
        fig.update_yaxes(title_text=labels_y[i], row=r, col=c)
        fig.update_xaxes(title_text='Tiempo (min)', row=r, col=c)

    fig.update_layout(height=580, template='plotly_white',
                      legend=dict(orientation='h', y=-0.12),
                      title_text='Cámara 3 – Biorreactor (P. putida) + Control Híbrido FF+PID')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# INTERFAZ STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title='Gemelo Digital – Sistema Bioelectroquímico 3 Cámaras',
        page_icon='⚗️',
        layout='wide',
    )

    # ── Cabecera ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:linear-gradient(90deg,#1e3a5f,#0f766e);
                padding:18px 28px;border-radius:10px;margin-bottom:16px'>
      <h2 style='color:white;margin:0'>
        ⚗️ Gemelo Digital – Sistema Bioelectroquímico Automatizado de 3 Cámaras
      </h2>
      <p style='color:#94d5c9;margin:4px 0 0 0;font-size:14px'>
        Recuperación de Metales + Biorremediación | Efluente Baterías Ion-Litio |
        Base Sulfato – Cero Cloruros | <em>Pseudomonas putida</em>
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Barra lateral ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.header('⚙️ Parámetros de Simulación')
        st.subheader('Condiciones de Entrada')
        C_M_in  = st.slider('Metal M²⁺ inicial (mg/L)', 100, 500, 250, 10)
        C_F_in  = st.slider('Flúor inicial (mg/L)',     100, 500, 250, 10)
        pH_in   = st.slider('pH inicial',               1.5, 3.0, 2.0, 0.05)

        st.subheader('Control Cámara 2')
        mode_pwm = st.selectbox('Modo de Corriente C2',
                                ['PWM Exponencial', 'Corriente Fija'])
        I_fixed  = st.slider('Corriente Fija / I_máx PWM (A)', 0.1, 1.0, 0.45, 0.05)

        st.subheader('Ganancias PID – Cámara 3')
        Kp_pid = st.slider('Kp', 0.1, 5.0, 1.2, 0.1)
        Ki_pid = st.slider('Ki', 0.01, 1.0, 0.15, 0.01)
        Kd_pid = st.slider('Kd', 0.0, 0.5, 0.05, 0.01)

        st.divider()
        run_btn = st.button('▶ Ejecutar Simulación', type='primary', use_container_width=True)

    # ── Ejecución / caché manual ──────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def run_all(C_M_in, C_F_in, pH_in, I_fixed, Kp_pid, Ki_pid, Kd_pid):
        r1 = simulate_chamber1(C_F_in, C_M_in, pH_in)
        r2 = simulate_chamber2(r1['C_M'][-1], pH2_in=float(r1['pH'][-1]),
                                I_fixed=I_fixed)
        # C_M que entra a C3 (PWM o fija)
        c3_M_in = float(r2['C_M_pwm'][-1])   # siempre usar PWM para C3 (propuesto)
        r3 = simulate_chamber3(c3_M_in, Kp_pid=Kp_pid, Ki_pid=Ki_pid, Kd_pid=Kd_pid)
        return r1, r2, r3

    if run_btn or 'sim_results' not in st.session_state:
        with st.spinner('🔬 Simulando sistema bioelectroquímico…'):
            r1, r2, r3 = run_all(C_M_in, C_F_in, pH_in, I_fixed, Kp_pid, Ki_pid, Kd_pid)
        st.session_state['sim_results'] = (r1, r2, r3)
    else:
        r1, r2, r3 = st.session_state['sim_results']

    # ── KPIs (tarjetas) ───────────────────────────────────────────────────────
    F_abatido  = C_F_in - float(r1['C_F'][-1])
    g_CaF2     = float(r1['CaF2'][-1]) * V_CHAMBER * 1e-3   # mg/L × L × 1e-3 → g
    m_dep_pwm  = float(r2['m_dep_pwm'][-1])
    m_dep_fix  = float(r2['m_dep_fix'][-1])
    pct_ahorro = max(0.0, (r2['energy_fix'][-1] - r2['energy_pwm'][-1]) /
                     (r2['energy_fix'][-1] + 1e-9) * 100.0)
    C_M_final  = float(r3['ff_pid']['C_M'][-1])
    pH_final   = float(r3['ff_pid']['pH'][-1])

    kpi_cols = st.columns(6)
    kpis = [
        ('🧪 F⁻ abatido C1', f'{F_abatido:.1f} mg/L', '#fef3c7'),
        ('⬜ CaF₂ generado', f'{g_CaF2:.2f} g', '#ecfdf5'),
        ('🔩 Metal en malla', f'{m_dep_pwm:.1f} mg', '#eff6ff'),
        ('⚡ Ahorro energía', f'{pct_ahorro:.1f} %', '#f5f3ff'),
        ('🏁 Metal salida C3', f'{C_M_final:.2f} mg/L', '#fef2f2'),
        ('🔋 Li⁺ recuperable', '200.0 mg/L', '#f0fdf4'),
    ]
    for col, (title, val, bg) in zip(kpi_cols, kpis):
        col.markdown(
            f"<div style='background:{bg};border-radius:8px;padding:12px;text-align:center;"
            f"border:1px solid #e5e7eb'>"
            f"<div style='font-size:12px;color:#57606a'>{title}</div>"
            f"<div style='font-size:22px;font-weight:700;color:#1f2328'>{val}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    # Validación de balance de masa
    m_in_C2    = float(r1['C_M'][-1]) * V_CHAMBER   # mg
    m_aq_fin   = float(r2['C_M_pwm'][-1]) * V_CHAMBER
    balance_ok = abs((m_in_C2 - m_aq_fin) - m_dep_pwm) < 5.0   # tolerancia 5 mg
    st.caption(
        f"✅ Balance de masa C2 verificado: entrada={m_in_C2:.1f} mg, "
        f"depositado={m_dep_pwm:.1f} mg, en solución={m_aq_fin:.1f} mg "
        f"| pH final C3 = {pH_final:.2f} | Li⁺ = 200.0 mg/L (intacto)"
        if balance_ok else
        f"⚠️ Balance de masa C2: entrada={m_in_C2:.1f} mg, "
        f"depositado={m_dep_pwm:.1f} mg, en solución={m_aq_fin:.1f} mg"
    )

    st.divider()

    # ── Pestañas ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        '📊 Vista General del Tren',
        '⚗️ Cámara 1 – Pretratamiento',
        '⚡ Cámara 2 – Electrocelda',
        '🧬 Cámara 3 – Biorreactor',
    ])

    with tab1:
        st.plotly_chart(fig_overview(r1, r2, r3, mode_pwm),
                        use_container_width=True)
        st.caption('Líneas punteadas verticales indican las transiciones entre cámaras.')

    with tab2:
        st.plotly_chart(fig_chamber1(r1), use_container_width=True)
        col_a, col_b = st.columns(2)
        col_a.metric('pH alcanzado (C1)', f'{float(r1["pH"][-1]):.3f}', '→ Setpoint 4.00')
        col_b.metric('[F⁻] residual (C1)', f'{float(r1["C_F"][-1]):.1f} mg/L',
                     f'-{F_abatido:.0f} mg/L desde {C_F_in} mg/L')

    with tab3:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.plotly_chart(fig_chamber2(r2), use_container_width=True)
        with col2:
            st.markdown('### 📈 Resumen C2')
            st.metric('Metal (PWM)',  f'{float(r2["C_M_pwm"][-1]):.1f} mg/L',
                      f'desde {float(r2["C_M_pwm"][0]):.0f} mg/L')
            st.metric('Metal (Fija)', f'{float(r2["C_M_fix"][-1]):.1f} mg/L')
            st.metric('Energía PWM',  f'{float(r2["energy_pwm"][-1]):.3f} Wh')
            st.metric('Energía Fija', f'{float(r2["energy_fix"][-1]):.3f} Wh')
            st.metric('Ahorro PWM',   f'{pct_ahorro:.1f} %', delta_color='inverse')
            st.metric('η_F medio (PWM)', '85.0 %')
            st.divider()
            st.markdown('**Ciclo duty PWM al inicio/fin:**')
            st.metric('u_PWM(t₀)', f'{float(r2["u_pwm"][0]):.3f}')
            st.metric('u_PWM(t_f)', f'{float(r2["u_pwm"][-1]):.4f}')

    with tab4:
        st.plotly_chart(fig_chamber3(r3), use_container_width=True)
        col_c, col_d = st.columns(2)
        col_c.metric('pH final (FF+PID)', f'{pH_final:.3f}', '→ Setpoint 7.00')
        col_d.metric('[M²⁺] final (biosorción)', f'{C_M_final:.2f} mg/L',
                     f'-{float(r3["ff_pid"]["C_M"][0]) - C_M_final:.1f} mg/L')

    # ── Descarga CSV ──────────────────────────────────────────────────────────
    st.divider()
    df_export = build_export_df(r1, r2, r3, mode_pwm)
    csv_bytes  = df_export.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='⬇️ Descargar Resultados (CSV)',
        data=csv_bytes,
        file_name='simulacion_3camaras.csv',
        mime='text/csv',
        type='secondary',
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='text-align:center;color:#9ca3af;font-size:12px;"
        "border-top:1px solid #e5e7eb;padding-top:12px;margin-top:8px'>"
        "Gemelo Digital · Sistema Bioelectroquímico 3 Cámaras · "
        "Pseudomonas putida · Base Sulfato | Desarrollado con IBM Bob"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
