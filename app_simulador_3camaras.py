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

# ── Propiedades por metal ────────────────────────────────────────────────────
# Cada entrada: (M [kg/mol], z [adim], E0_cell [V], K_L_langmuir [L/mg],
#                q_max_langmuir [mg/g], k_ppco [1/s], label)
METAL_PROPS = {
    'Cu²⁺': (63.55e-3, 2, 2.00, 0.35, 2.2, 8e-5, 'Cu²⁺'),
    # E0_cell Cu: E°(Cu²⁺/Cu⁰)=+0.34V + sobrepot. anódico ≈ 2.0 V total
    # K_L, q_max: Pagnanelli et al. (2003), P. putida en sulfato
    # k_ppco: cinética Cu(OH)₂, Ksp=2e-19, pH>6.5

    'Co²⁺': (58.93e-3, 2, 2.60, 0.18, 1.6, 1e-5, 'Co²⁺'),
    # E0_cell Co: E°(Co²⁺/Co⁰)=−0.28V + sobrepot. anódico ≈ 2.6 V total
    # K_L, q_max: Pardo et al. (2003), P. putida, valores medios rango 0.10–0.25 L/mg
    # k_ppco: cinética Co(OH)₂, Ksp=5.9e-15, más soluble → constante menor
}
METAL_DEFAULT = 'Cu²⁺'

KPS_CAF2  = 3.9e-11       # Producto de solubilidad CaF2  [(mol/L)³]
PKA_HF    = 3.17          # pKa HF en agua pura 25°C (en solución sulfato puede ser ~2.9–3.1)
PKA_HSO4  = 1.99          # pKa2 H2SO4
A_CATHODE = 0.044         # m² (440 cm²) – área activa 3D malla inox 316L
K_MASS    = 1.8e-5        # m/s  coef. transferencia de masa (asumido; depende de Re, Sc, geom.)
TAU_DIFF  = V_CHAMBER / (A_CATHODE * K_MASS)   # ~21.1 min → s
ETA_PWM_TARGET = 0.85     # fracción de I_lim aplicada por PWM; η_real ≈ 1 para esa corriente

# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 1 – CÁMARA 1: Neutralización + Precipitación CaF2
# ─────────────────────────────────────────────────────────────────────────────

def simulate_chamber1(C_F_in_mgL, C_M_in_mgL, pH_in, pH1_ref=4.0,
                       Kp=0.8, Ki=0.4, t_end_min=15.0, n_points=450,
                       C_SO4_mM=None):
    """
    C_SO4_mM : concentración de SO₄²⁻ total en el influente [mmol/L].
               Si es None, se estima como (10^(-pH_in)) / 2 × 1000 mM
               (toda la acidez proviene de H₂SO₄).
    u_base   : tasa de adición de OH⁻ [mol·L⁻¹·s⁻¹].
               Límite 0.05 mol/L/s ≈ 3 mol/L/min; para Ca(OH)₂ 50 g/L
               equivale a ~111 mL/min — razonable para bomba peristáltica.
    """
    t_span = (0.0, t_end_min * 60.0)
    t_eval = np.linspace(*t_span, n_points)

    # E02-A: concentración SO₄²⁻ derivada del pH de entrada considerando pKa2(HSO₄⁻) = 1.99
    # H₂SO₄ → H⁺ + HSO₄⁻ (100%) y HSO₄⁻ ⇌ H⁺ + SO₄²⁻ (fracción α_SO4 = 1/(1 + 10^(pKa2 - pH)))
    if C_SO4_mM is None:
        alpha_so4_in = 1.0 / (1.0 + 10.0 ** (PKA_HSO4 - pH_in))
        C_SO4_mM = max((10.0 ** (-pH_in)) / (1.0 + alpha_so4_in) * 1000.0, 0.1)

    def alpha_HF(pH_val):
        return 1.0 / (1.0 + 10.0 ** (pH_val - PKA_HF))

    def beta_buffer(pH_val, C_F):
        # β [mol/L/ΔpH] — Stumm & Morgan (1996) cap. 3
        ln10 = np.log(10.0)
        beta_w  = ln10 * (10.0 ** (-pH_val) + 10.0 ** (pH_val - 14.0))
        aHF = alpha_HF(pH_val)
        C_F_mol = (C_F / 1000.0) / 19.0          # mg/L → mol/L
        beta_hf = ln10 * aHF * (1.0 - aHF) * C_F_mol
        # E02-A corregido: usar C_SO4 derivado del efluente en lugar de 1e-2 fijo
        C_SO4_mol = C_SO4_mM * 1e-3               # mmol/L → mol/L
        alpha_hso4 = 1.0 / (1.0 + 10.0 ** (pH_val - PKA_HSO4))
        beta_hso4 = ln10 * alpha_hso4 * (1.0 - alpha_hso4) * C_SO4_mol
        return max(beta_w + beta_hf + beta_hso4, 1e-6)

    # Escala de dosificación de microbomba peristáltica de Ca(OH)₂ para reactor de 1 L:
    # Convierte la señal adimensional del PI (Kp·e + Ki·∫e) a mol OH⁻ · L⁻¹ · s⁻¹
    k_pump_c1 = 1.2e-4   # mol·L⁻¹·s⁻¹ por unidad de acción de control
    u_max_c1  = 2.2e-4   # mol·L⁻¹·s⁻¹ (~10 mL/min de lechada Ca(OH)₂)

    # Solubilidad condicional de CaF₂ en matriz sulfato (fuerza iónica I ≈ 0.03-0.05 M,
    # coeficientes de actividad γ_Ca ≈ 0.45, γ_F ≈ 0.80 y límite de nucleación secundaria
    # que deja un residual práctico de ~11-12 mg/L de flúor total a pH 4.0)
    C_F_eq_mgL = 11.5    # mg/L de flúor residual de equilibrio práctico en C1

    def odes(t, y):
        # Estados: [pH, C_F (mg/L), C_M (mg/L), integral_PI, Ca2_acum (mol/L)]
        pH_val, C_F, C_M, integral_e, Ca2_acum = y
        pH_val   = np.clip(pH_val, 0.0, 14.0)
        C_F      = max(C_F, 0.0)
        C_M      = max(C_M, 0.0)
        Ca2_acum = max(Ca2_acum, 0.0)

        error = pH1_ref - pH_val
        if pH_val >= pH1_ref:
            d_integral = 0.0
            u_base     = 0.0
        else:
            u_raw  = k_pump_c1 * max(Kp * error + Ki * integral_e, 0.0)
            u_base = min(u_raw, u_max_c1)
            # Anti-windup: no acumular integral si el actuador está saturado
            d_integral = 0.0 if u_raw >= u_max_c1 else error

        beta = beta_buffer(pH_val, C_F)

        # Ca²⁺ añadido: Ca(OH)₂ → Ca²⁺ + 2 OH⁻  → 1 mol Ca²⁺ por 2 mol OH⁻
        dCa2_in = u_base / 2.0                    # mol/L/s de Ca²⁺ ingresando

        # E06: concentración F⁻ libre ionizada [mol/L]
        aHF_val = alpha_HF(pH_val)
        F_eff_mgL = max(C_F - C_F_eq_mgL, 0.0)
        F_conc  = (1.0 - aHF_val) * (F_eff_mgL / 1000.0) / 19.0   # mol/L precipitable

        # Producto iónico vs Ksp condicional: IP = [Ca²⁺][F⁻]²
        k_ppt = 1.5e4                              # (mol/L)⁻² · s⁻¹  (cinética 2º orden)
        supersaturation = max(Ca2_acum * (F_conc ** 2.0) - KPS_CAF2, 0.0)
        dCaF2_molLs = min(k_ppt * supersaturation, F_conc / 2.0)

        # Al precipitar F⁻ a pH ácido, el equilibrio HF ⇌ H⁺ + F⁻ libera 2·α_HF H⁺ por mol de CaF₂
        H_released = 2.0 * aHF_val * dCaF2_molLs
        dpH = (u_base - H_released) / beta

        # E08-A corregido: cada mol CaF₂ consume 2 mol F⁻ → factor 38/78.08
        dC_F  = -dCaF2_molLs * 2.0 * 19.0 * 1000.0   # mol/L/s × 2 × 19 g/mol × 1000 mg/g

        # Balance Ca²⁺ en solución: entra por dosificación, se consume en precipitación
        dCa2_consumed = dCaF2_molLs                # 1 mol Ca²⁺ por mol CaF₂
        dCa2_acum = dCa2_in - dCa2_consumed

        dC_M = 0.0  # metal no precipita a pH < 5.5

        return [dpH, dC_F, dC_M, d_integral, dCa2_acum]

    y0 = [pH_in, C_F_in_mgL, C_M_in_mgL, 0.0, 0.0]
    sol = solve_ivp(odes, t_span, y0, t_eval=t_eval,
                    method='RK45', rtol=1e-6, atol=1e-8,
                    dense_output=False, max_step=5.0)

    t_min = sol.t / 60.0
    pH1   = np.clip(sol.y[0], 0.0, 14.0)
    C_M1  = np.maximum(sol.y[2], 0.0)
    C_F1  = np.maximum(sol.y[1], 0.0)

    # Balance estequiométrico exacto: 2 mol F⁻ (38.00 g) → 1 mol CaF₂ (78.08 g)
    caf2_mgL = np.maximum((C_F_in_mgL - C_F1) * (78.08 / 38.00), 0.0)

    return dict(t_min=t_min, pH=pH1, C_F=C_F1, C_M=C_M1, CaF2=caf2_mgL)


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 2 – CÁMARA 2: Electrodeposición + PWM Exponencial
# ─────────────────────────────────────────────────────────────────────────────

def simulate_chamber2(C_M_in_mgL, pH2_in=4.0, I_fixed=0.45,
                       t_start_min=15.0, t_end_min=75.0, n_points=1800,
                       metal=METAL_DEFAULT):
    """
    metal : clave de METAL_PROPS ('Cu²⁺' o 'Co²⁺').
            Determina M, z, E0_cell y faraday_factor del metal a depositar.
    """
    t_span  = (0.0, (t_end_min - t_start_min) * 60.0)
    t_eval  = np.linspace(*t_span, n_points)

    # Propiedades del metal seleccionado (M en METAL_PROPS está en kg/mol → convertir a g/mol)
    M_kg_mol, z_metal, E0_cell, _, _, _, _ = METAL_PROPS[metal]
    M_g_mol = M_kg_mol * 1000.0   # kg/mol → g/mol (ej. Cu: 63.55 g/mol)
    V_L = V_CHAMBER * 1e3         # m³ → L (1.0 L)

    def I_lim(C_M_mgL):
        # I_lim = z·F·A·k_m·C  [A]; C en mol/m³
        # C_M_mgL [mg/L] / M_g_mol [g/mol] = mmol/L = mol/m³  (1 mmol/L = 1 mol/m³)
        C_molm3 = max(C_M_mgL / M_g_mol, 0.0)
        return max(z_metal * F_FARADAY * A_CATHODE * K_MASS * C_molm3, 1e-9)

    # faraday_factor [mg/C] = M[g/mol] × 1000[mg/g] / (z·F[C/mol])
    faraday_factor = M_g_mol * 1000.0 / (z_metal * F_FARADAY)

    def _E_cell_nernst(C_M, C_M_ref):
        """
        Tensión de celda electrolítica usando Nernst con concentración absoluta [mol/L].
        Al reducirse [M²⁺], el potencial catódico se vuelve más negativo y la tensión
        total requerida por la celda aumenta:
        E_cell = E0_cell - (0.0592/z)·log10([M²⁺])
        """
        C_molL = max(C_M / (M_g_mol * 1000.0), 1e-12)   # mg/L → mol/L
        return E0_cell - (0.0592 / z_metal) * np.log10(C_molL)

    def _eta_fixed(I_app, I_L):
        # Cuando I_app <= I_L, η_F = ETA_PWM_TARGET (85%).
        # Cuando I_app > I_L, el exceso de corriente genera H₂ y las burbujas
        # bloquean parcialmente el área activa catódica (factor (I_L/I_app)^0.35).
        ratio = I_L / (I_app + 1e-12)
        bubble_factor = min(1.0, ratio ** 0.35)
        return ETA_PWM_TARGET * min(1.0, ratio) * bubble_factor

    def odes_fixed(t, y):
        C_M, m_dep, energy = y
        C_M   = max(C_M, 0.0)
        I_app = I_fixed
        I_L   = I_lim(C_M)
        eta_F = _eta_fixed(I_app, I_L)
        # Ley de Faraday: dm/dt = η_F·I·M/(z·F)
        dm_dt = eta_F * I_app * faraday_factor
        dC_M  = -dm_dt / V_L
        E_cell = _E_cell_nernst(C_M, C_M_in_mgL)
        dE     = I_app * abs(E_cell) / 3600.0
        return [dC_M, dm_dt, dE]

    y0_a  = [C_M_in_mgL, 0.0, 0.0]
    sol_a = solve_ivp(odes_fixed, t_span, y0_a, t_eval=t_eval,
                      method='RK45', rtol=1e-6, atol=1e-8, max_step=2.0)

    def odes_pwm(t, y):
        C_M, m_dep, energy = y
        C_M   = max(C_M, 1e-6)
        I_L   = I_lim(C_M)
        # El control PWM modula la corriente siguiendo I_lim sin entrar en evolución
        # masiva de H₂, manteniendo eficiencia faradaica constante η_F = ETA_PWM_TARGET
        I_set = ETA_PWM_TARGET * I_L
        eta_F = ETA_PWM_TARGET
        # Ley de Faraday consistente: dm/dt = η_F·I_set·M/(z·F)
        dm_dt = eta_F * I_set * faraday_factor
        dC_M  = -dm_dt / V_L
        E_cell = _E_cell_nernst(C_M, C_M_in_mgL)
        dE     = I_set * abs(E_cell) / 3600.0
        return [dC_M, dm_dt, dE]

    y0_b   = [C_M_in_mgL, 0.0, 0.0]
    sol_b  = solve_ivp(odes_pwm, t_span, y0_b, t_eval=t_eval,
                       method='RK45', rtol=1e-6, atol=1e-8, max_step=2.0)

    t_abs = t_eval / 60.0 + t_start_min

    C_M_a   = np.maximum(sol_a.y[0], 0.0)
    C_M_b   = np.maximum(sol_b.y[0], 0.0)
    I_L_a   = np.array([I_lim(c) for c in C_M_a])
    I_L_b   = np.array([I_lim(c) for c in C_M_b])
    I_app_a = np.full_like(t_eval, I_fixed)
    I_app_b = ETA_PWM_TARGET * I_L_b
    eta_a   = np.array([_eta_fixed(I_fixed, il) for il in I_L_a])
    eta_b   = np.full_like(t_eval, ETA_PWM_TARGET)
    pH2_a   = np.full_like(t_eval, pH2_in)
    pH2_b   = np.full_like(t_eval, pH2_in)
    u_pwm   = np.clip(I_app_b / (I_fixed + 1e-12), 0.0, 1.0)

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
                       Kd_pid=0.05, n_points=1350,
                       metal=METAL_DEFAULT, C_SO4_mM=None):
    """
    metal     : clave de METAL_PROPS para K_L, q_max y k_ppco correctos por metal.
    C_SO4_mM  : [mmol/L] sulfato residual que llega de C2 para el cálculo de β.
                Si es None, se usa 5 mM como estimado conservador del residual.

    Supuesto de biomasa (E30-A documentado):
      m_bio [g/L] es biomasa inmovilizada en soporte de tezontle (FIJA, no crece).
      X [g/L] es biomasa libre en suspensión (crece con Monod, valor inicial X0).
      La biosorción ocurre sobre m_bio (soporte); X modela el crecimiento libre
      para la biodegradación de EC. Son dos poblaciones distintas y desconectadas
      intencionalmente — simplificación válida para escala de laboratorio con
      soporte saturado al inicio (Pagnanelli et al. 2003).
    """
    t_span  = (0.0, (t_end_min - t_start_min) * 60.0)
    t_eval  = np.linspace(*t_span, n_points)
    t_pert  = (95.0 - t_start_min) * 60.0

    # Parámetros de biosorción — específicos del metal (E26-A, E12-A corregidos)
    _, _, _, K_L, q_max, k_ppco_metal, _ = METAL_PROPS[metal]

    m_bio      = 50.0    # g_biomasa_soporte/L — fija, ver docstring E30-A
    pH_pKa_eps = 4.8     # pKa grupos –COOH del EPS; literatura: 4.5–5.2

    def phi_pH(pH_v):
        # Función logística de activación de grupos –COO⁻ del EPS
        return 1.0 / (1.0 + 10.0 ** (pH_pKa_eps - pH_v))

    mu_max  = 2.2e-4  # 1/s (~0.79 h⁻¹) — tasa máx. crecimiento realista P. putida en EC
    K_s_EC  = 80.0    # mg/L — constante de semisaturación EC (Monod)
    Y_x     = 0.35    # g_biomasa/g_EC (≡ mg_biomasa/mg_EC) — rendimiento celular
    K_d     = 5.0e-6  # 1/s (~0.018 h⁻¹) — tasa de decaimiento endógeno
    X0      = 0.15    # g/L — biomasa libre inicial en suspensión
    k_ads   = 0.006   # 1/s — constante LDF biosorción (0.001–0.05 según agitación)
    tau_d   = 5.0     # s — constante de tiempo del filtro derivativo del PID

    # E22-B corregido: sulfato residual de C2 incluido en β de C3
    if C_SO4_mM is None:
        C_SO4_mM = 5.0   # mM — estimado conservador residual de C2

    def beta_buf(pH_v):
        """
        β [mol/L/ΔpH] — incluye agua y tampón sulfato residual de C2.
        Bicarbonato del CO₂ metabólico no modelado (simplificación).
        """
        ln10   = np.log(10.0)
        beta_w = ln10 * (10.0 ** (-pH_v) + 10.0 ** (pH_v - 14.0))
        C_SO4_mol   = C_SO4_mM * 1e-3
        alpha_hso4  = 1.0 / (1.0 + 10.0 ** (pH_v - PKA_HSO4))
        beta_hso4   = ln10 * alpha_hso4 * (1.0 - alpha_hso4) * C_SO4_mol
        return max(beta_w + beta_hso4 + 5e-4, 5e-4)

    # Escala de microbomba peristáltica de NaOH/Ca(OH)₂ en C3 (1 L):
    # Convierte la salida adimensional del controlador PID a mol OH⁻ · L⁻¹ · s⁻¹
    k_pump_c3 = 4.5e-6   # mol·L⁻¹·s⁻¹ por unidad de señal PID
    u_max_c3  = 1.8e-5   # mol·L⁻¹·s⁻¹ caudal máximo de dosificación

    def make_odes(mode):
        def odes(t, y):
            # Estados: [pH, C_M, C_EC, X_libre, integral_PID, d_filt_PID, q_bio]
            pH_v, C_M, C_EC, X, integral_e, d_filt, q_bio = y
            pH_v  = np.clip(pH_v, 0.0, 14.0)
            C_M   = max(C_M, 0.0)
            C_EC  = max(C_EC, 0.0)
            X     = max(X, 0.01)
            q_bio = max(q_bio, 0.0)

            error = pH3_ref - pH_v
            beta  = beta_buf(pH_v)

            # Perturbación ácida externa [mol H⁺ · L⁻¹ · s⁻¹] en t = 95 min (±30 s)
            q_acid_pert = 1.4e-5 if abs(t - t_pert) < 30.0 else 0.0

            if mode == 0:
                u_OH       = 0.0
                d_integral = 0.0

            elif mode == 1:
                # Control solo PID con filtro derivativo de 1er orden (d_filt ≈ de/dt)
                # Bomba unidireccional de base: si supera el setpoint se apaga y descarga integral
                if pH_v >= pH3_ref:
                    u_OH       = 0.0
                    d_integral = -integral_e / 15.0
                else:
                    u_Signal = Kp_pid * error + Ki_pid * (integral_e / 60.0) + Kd_pid * (d_filt * 60.0)
                    u_raw    = k_pump_c3 * u_Signal
                    u_OH     = float(np.clip(u_raw, 0.0, u_max_c3))
                    if (u_raw >= u_max_c3 and error > 0.0) or abs(error) < 0.02:
                        d_integral = 0.0
                    else:
                        d_integral = error

            else:
                # Control Híbrido Feedforward + PID:
                # Feedforward compensa la acidez inicial de entrada y el 85% de la perturbación
                # ácida medida aguas arriba antes de que desvíe el pH del biorreactor.
                H_in   = 10.0 ** (-pH3_in)
                Q_in   = 5.0e-6                         # m³/s
                ff_OH  = (H_in * Q_in / V_CHAMBER) + 0.85 * q_acid_pert
                u_pid  = k_pump_c3 * (Kp_pid * error + Ki_pid * (integral_e / 60.0) + Kd_pid * (d_filt * 60.0))
                u_raw  = u_pid + ff_OH
                u_OH   = float(np.clip(u_raw, 0.0, u_max_c3 * 1.25))
                if pH_v >= pH3_ref and error <= 0.0:
                    u_OH = min(u_OH, q_acid_pert)
                if (u_raw >= u_max_c3 * 1.25 and error > 0.0) or (u_raw <= 0.0 and error < 0.0) or abs(error) < 0.02:
                    d_integral = 0.0
                else:
                    d_integral = error

            # Balance dinámico de pH sin recortes artificiales post-integración
            dpH = (u_OH - q_acid_pert) / beta

            # Filtro derivativo exacto: d(d_filt)/dt = (de/dt - d_filt) / tau_d, donde de/dt = -dpH
            dd_filt = (-dpH - d_filt) / tau_d

            # Biosorción: Langmuir + cinética LDF
            q_eq  = q_max * K_L * C_M / (1.0 + K_L * C_M) * phi_pH(pH_v)
            dq    = k_ads * (q_eq - q_bio)
            dC_M_bio  = -dq * m_bio
            # Precipitación química (sigmoide suave centrada en pH 6.5)
            k_ppco_smooth = k_ppco_metal / (1.0 + np.exp(-10.0 * (pH_v - 6.5)))
            dC_M_ppco = -k_ppco_smooth * C_M
            dC_M = dC_M_bio + dC_M_ppco

            # Crecimiento Monod (X en g/L; C_EC en mg/L → factor 1000 mg/g en dC_EC)
            mu    = mu_max * C_EC / (K_s_EC + C_EC + 1e-9)
            dX    = (mu - K_d) * X
            dC_EC = -mu * (X * 1000.0) / Y_x

            return [dpH, dC_M, dC_EC, dX, d_integral, dd_filt, dq]

        return odes

    results = {}
    for mode_id, mode_name in [(0, 'open'), (1, 'pid'), (2, 'ff_pid')]:
        y0 = [pH3_in, C_M_in_mgL, C_EC_in, X0, 0.0, 0.0, 0.0]
        sol = solve_ivp(make_odes(mode_id), t_span, y0, t_eval=t_eval,
                        method='RK45', rtol=1e-6, atol=1e-8, max_step=2.0)
        pH_raw = np.clip(sol.y[0], 0.0, 14.0)
        results[mode_name] = dict(
            pH   = pH_raw,
            C_M  = np.maximum(sol.y[1], 0.0),
            C_EC = np.maximum(sol.y[2], 0.0),
            X    = sol.y[3],
        )

    def odes_abiotic(t, y):
        pH_v, C_M = y
        pH_v = np.clip(pH_v, 0.0, 14.0)
        C_M  = max(C_M, 0.0)
        idx  = min(int(t / max(t_span[1], 1.0) * (n_points - 1)), n_points - 1)
        pH_v = results['ff_pid']['pH'][idx]
        # E32-B corregido: sigmoide suave en lugar de escalón en pH 6.5
        k_ppco_smooth = k_ppco_metal / (1.0 + np.exp(-10.0 * (pH_v - 6.5)))
        dC_M = -k_ppco_smooth * C_M
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
    df1 = pd.DataFrame({
        't_min': r1['t_min'],
        'camara': 1,
        'pH': r1['pH'],
        'C_F_mgL': r1['C_F'],
        'C_M_mgL': r1['C_M'],
        'CaF2_mgL': r1['CaF2'],
    })
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
    tm = np.concatenate([r1['t_min'], r2['t_min'], r3['t_min']])
    C2_pH = r2['pH2_b'] if mode_pwm == 'PWM Exponencial' else r2['pH2_a']
    pH_global = np.concatenate([r1['pH'], C2_pH, r3['ff_pid']['pH']])
    cm2 = r2['C_M_pwm'] if mode_pwm == 'PWM Exponencial' else r2['C_M_fix']
    C_M_global = np.concatenate([r1['C_M'], cm2, r3['ff_pid']['C_M']])
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
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=['Corriente Aplicada vs I_lim (A)',
                                        'Eficiencia Faradaica η_F',
                                        'Metal Depositado en Malla (mg)',
                                        'Consumo Energético (Wh)'],
                        vertical_spacing=0.16, horizontal_spacing=0.10)

    tm = r2['t_min']

    fig.add_trace(go.Scatter(x=tm, y=r2['I_pwm'], name='I_PWM (propuesto)',
                             line=dict(color=COLORS['pwm'], width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=tm, y=r2['I_fix'], name='I_Fija',
                             line=dict(color=COLORS['fixed'], width=2, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=tm, y=r2['I_lim_b'], name='I_límite (difusión)',
                             line=dict(color=COLORS['lim'], width=1.5, dash='dot')), row=1, col=1)

    fig.add_trace(go.Scatter(x=tm, y=r2['eta_pwm'] * 100, name='η_F PWM (%)',
                             line=dict(color=COLORS['pwm'], width=2.5)), row=1, col=2)
    fig.add_trace(go.Scatter(x=tm, y=r2['eta_fix'] * 100, name='η_F Fija (%)',
                             line=dict(color=COLORS['fixed'], width=2, dash='dash')), row=1, col=2)
    fig.add_hline(y=85.0, line_dash='dot', line_color=COLORS['caf2'],
                  annotation_text='85% objetivo', row=1, col=2)

    fig.add_trace(go.Scatter(x=tm, y=r2['m_dep_pwm'], name='Malla – PWM',
                             line=dict(color=COLORS['pwm'], width=2.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=tm, y=r2['m_dep_fix'], name='Malla – Fija',
                             line=dict(color=COLORS['fixed'], width=2, dash='dash')), row=2, col=1)

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
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Control de pH – Comparativa 3 Modos',
            'Biosorción: P. putida vs Control Abiótico',
            'Biodegradación de EC (mg/L) + Biomasa',
            'Litio (Li⁺) – Conservado Intacto',
        ],
        specs=[
            [{'secondary_y': False}, {'secondary_y': False}],
            [{'secondary_y': True},  {'secondary_y': False}],
        ],
        vertical_spacing=0.16, horizontal_spacing=0.10,
    )

    tm    = r3['t_min']
    t_p   = r3['t_pert_min']

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

    fig.add_trace(go.Scatter(x=tm, y=r3['ff_pid']['C_M'], name='M²⁺ (con P. putida)',
                             line=dict(color=COLORS['bio'], width=2.5)), row=1, col=2)
    fig.add_trace(go.Scatter(x=tm, y=r3['abiotic']['C_M'], name='M²⁺ (control abiótico)',
                             line=dict(color=COLORS['abiotic'], width=2, dash='dash')), row=1, col=2)
    fig.add_hline(y=1.2, line_dash='dot', line_color='#64748b',
                  annotation_text='< 1.2 mg/L objetivo', row=1, col=2)

    fig.add_trace(go.Scatter(x=tm, y=r3['ff_pid']['C_EC'], name='EC (mg/L)',
                             line=dict(color=COLORS['ec'], width=2.5)),
                  row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=tm, y=r3['ff_pid']['X'] * 1000, name='Biomasa (mg/L)',
                             line=dict(color=COLORS['bio'], width=2, dash='dot')),
                  row=2, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=tm, y=np.full_like(tm, r3['C_Li']), name='Li⁺ (intacto)',
                             fill='tozeroy',
                             line=dict(color=COLORS['li'], width=2.5)), row=2, col=2)
    fig.add_annotation(x=np.mean(tm), y=r3['C_Li'],
                       text='Li⁺ = 200 mg/L – E₀ = -3.04 V → nunca se reduce',
                       showarrow=False, font=dict(color='#6d28d9', size=10),
                       bgcolor='#ede9fe', row=2, col=2)

    fig.update_yaxes(title_text='pH', row=1, col=1)
    fig.update_yaxes(title_text='mg/L', row=1, col=2)
    fig.update_yaxes(title_text='EC (mg/L)', row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text='Biomasa (mg/L)', row=2, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text='mg/L', row=2, col=2)
    for r, c in [(1,1),(1,2),(2,1),(2,2)]:
        fig.update_xaxes(title_text='Tiempo (min)', row=r, col=c)

    fig.update_layout(height=580, template='plotly_white',
                      legend=dict(orientation='h', y=-0.12),
                      title_text='Cámara 3 – Biorreactor (P. putida) + Control Híbrido FF+PID')
    return fig

 
# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAMA DE PROCESO – REPRESENTACIÓN VISUAL DEL TREN DE 3 CÁMARAS
# ─────────────────────────────────────────────────────────────────────────────

def fig_process_diagram(r1, r2, r3, mode_pwm, C_M_in, C_F_in, pH_in):
    """
    Genera un diagrama de proceso (PFD) colorido e interactivo usando Plotly.
    Muestra las 3 cámaras, flujos de entrada/salida y valores clave dinámicos.
    """
    # ── Valores de estado final de cada cámara ──────────────────────────────
    pH_c1_out  = float(r1['pH'][-1])
    CF_c1_out  = float(r1['C_F'][-1])
    CM_c1_out  = float(r1['C_M'][-1])
    CaF2_gen   = float(r1['CaF2'][-1])

    cm2_key    = 'C_M_pwm' if mode_pwm == 'PWM Exponencial' else 'C_M_fix'
    en2_key    = 'energy_pwm' if mode_pwm == 'PWM Exponencial' else 'energy_fix'
    md2_key    = 'm_dep_pwm' if mode_pwm == 'PWM Exponencial' else 'm_dep_fix'
    eta2_key   = 'eta_pwm' if mode_pwm == 'PWM Exponencial' else 'eta_fix'
    CM_c2_out  = float(r2[cm2_key][-1])
    energy_c2  = float(r2[en2_key][-1])
    m_dep_c2   = float(r2[md2_key][-1])
    eta_c2_pct = float(np.mean(r2[eta2_key]) * 100.0)
    pH_c2_out  = float(r2['pH2_b' if mode_pwm == 'PWM Exponencial' else 'pH2_a'][-1])

    pH_c3_out  = float(r3['ff_pid']['pH'][-1])
    CM_c3_out  = float(r3['ff_pid']['C_M'][-1])
    EC_c3_out  = float(r3['ff_pid']['C_EC'][-1])
    Li_c3      = float(r3['C_Li'])

    CM_ok  = CM_c3_out <= 1.2
    pH_ok  = 6.5 <= pH_c3_out <= 7.5
    CF_ok  = CF_c1_out <= 15.0

    # ── Paleta de colores para cada cámara ───────────────────────────────────
    C1_COLOR  = '#0ea5e9'   # azul cielo – neutralización
    C2_COLOR  = '#f97316'   # naranja     – electrodeposición
    C3_COLOR  = '#22c55e'   # verde       – biorreactor
    C1_LIGHT  = '#e0f2fe'
    C2_LIGHT  = '#fff7ed'
    C3_LIGHT  = '#f0fdf4'
    ARROW_CLR = '#64748b'
    BG_CLR    = '#f8fafc'

    fig = go.Figure()

    # ── Fondo general ────────────────────────────────────────────────────────
    fig.add_shape(type='rect', x0=0, y0=0, x1=22, y1=12,
                  fillcolor=BG_CLR, line_width=0, layer='below')

    # ─────────────────────────────────────────────────────────────────────────
    # BLOQUE: ENTRADA (efluente crudo)
    # ─────────────────────────────────────────────────────────────────────────
    fig.add_shape(type='rect', x0=0.2, y0=4.5, x1=2.5, y1=7.5,
                  fillcolor='#f1f5f9', line=dict(color='#94a3b8', width=2),
                  layer='below')
    fig.add_annotation(x=1.35, y=7.9, text='<b>EFLUENTE</b>', showarrow=False,
                       font=dict(size=11, color='#334155'), xanchor='center')
    fig.add_annotation(x=1.35, y=7.2, text='<b>ENTRADA</b>', showarrow=False,
                       font=dict(size=11, color='#334155'), xanchor='center')
    for i, (lbl, val) in enumerate([
        (f'pH = {pH_in:.1f}', '#dc2626'),
        (f'M²⁺ = {C_M_in} mg/L', '#b45309'),
        (f'F⁻ = {C_F_in} mg/L', '#0369a1'),
        ('Li⁺ = 200 mg/L', '#7c3aed'),
    ]):
        fig.add_annotation(x=1.35, y=6.8 - i * 0.6, text=lbl, showarrow=False,
                           font=dict(size=10, color=val), xanchor='center')

    # Flecha entrada → C1
    fig.add_annotation(x=3.7, y=6.0, ax=2.5, ay=6.0,
                       xref='x', yref='y', axref='x', ayref='y',
                       showarrow=True, arrowhead=2, arrowsize=1.4,
                       arrowwidth=3, arrowcolor=ARROW_CLR)
    fig.add_annotation(x=3.1, y=6.3, text='Influente', showarrow=False,
                       font=dict(size=9, color=ARROW_CLR), xanchor='center')

    # ─────────────────────────────────────────────────────────────────────────
    # BLOQUE: CÁMARA 1 – NEUTRALIZACIÓN
    # ─────────────────────────────────────────────────────────────────────────
    fig.add_shape(type='rect', x0=3.7, y0=2.5, x1=7.5, y1=9.5,
                  fillcolor=C1_LIGHT, line=dict(color=C1_COLOR, width=3),
                  layer='below')
    # Badge de número
    fig.add_shape(type='circle', x0=4.1, y0=8.8, x1=5.1, y1=9.8,
                  fillcolor=C1_COLOR, line_width=0)
    fig.add_annotation(x=4.6, y=9.3, text='<b>1</b>', showarrow=False,
                       font=dict(size=16, color='white'), xanchor='center')
    fig.add_annotation(x=5.6, y=9.3, text='<b>CÁMARA 1</b>', showarrow=False,
                       font=dict(size=13, color=C1_COLOR), xanchor='center')
    fig.add_annotation(x=5.6, y=8.7, text='Neutralización', showarrow=False,
                       font=dict(size=11, color='#0369a1'), xanchor='center')
    fig.add_annotation(x=5.6, y=8.3, text='Precipitación CaF₂', showarrow=False,
                       font=dict(size=10, color='#0369a1'), xanchor='center')
    fig.add_annotation(x=5.6, y=7.7, text='0 – 15 min', showarrow=False,
                       font=dict(size=10, color='#64748b', family='monospace'), xanchor='center')

    # Icono representativo (contenedor con ondas)
    fig.add_shape(type='rect', x0=4.1, y0=4.5, x1=7.1, y1=7.0,
                  fillcolor='#bae6fd', line=dict(color=C1_COLOR, width=1.5),
                  layer='below')
    # Ondas simuladas
    for y_w in [5.0, 5.6, 6.2]:
        xs = np.linspace(4.2, 7.0, 40)
        ys = y_w + 0.12 * np.sin(np.linspace(0, 4 * np.pi, 40))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines',
                                 line=dict(color='white', width=1.5),
                                 showlegend=False, hoverinfo='skip'))

    # Burbuja de Ca(OH)₂ dosificada
    fig.add_annotation(x=5.6, y=4.0, text='⬆ Ca(OH)₂ dosificado',
                       showarrow=False, font=dict(size=9, color='#0284c7'), xanchor='center')
    fig.add_annotation(x=5.6, y=3.5, text='Control PI', showarrow=False,
                       font=dict(size=9, color='#0284c7', family='monospace'), xanchor='center')

    # KPIs de salida C1
    CF_color = '#16a34a' if CF_ok else '#dc2626'
    for i, (lbl, val, clr) in enumerate([
        ('pH salida:', f'{pH_c1_out:.2f}',        '#1d4ed8'),
        ('[F⁻]:',     f'{CF_c1_out:.1f} mg/L',   CF_color),
        ('[M²⁺]:',    f'{CM_c1_out:.0f} mg/L',   '#b45309'),
        ('CaF₂:',     f'{CaF2_gen:.0f} mg/L',    '#059669'),
    ]):
        fig.add_annotation(x=4.0 + (i % 2) * 1.7, y=3.2 - (i // 2) * 0.55,
                           text=f'<b>{lbl}</b> {val}', showarrow=False,
                           font=dict(size=9, color=clr), xanchor='left')

    # Descarga de sólidos (CaF₂)
    fig.add_annotation(x=5.6, y=2.1, text='⬇ CaF₂ sólido (filtrado)', showarrow=False,
                       font=dict(size=9, color='#059669'), xanchor='center')
    fig.add_annotation(x=5.6, y=1.6, text=f'≈ {CaF2_gen:.0f} mg/L', showarrow=False,
                       font=dict(size=9, color='#059669'), xanchor='center')
    fig.add_shape(type='line', x0=5.6, y0=2.5, x1=5.6, y1=2.2,
                  line=dict(color='#059669', width=2, dash='dot'))

    # Flecha C1 → C2
    fig.add_annotation(x=10.2, y=6.0, ax=7.5, ay=6.0,
                       xref='x', yref='y', axref='x', ayref='y',
                       showarrow=True, arrowhead=2, arrowsize=1.4,
                       arrowwidth=3, arrowcolor=ARROW_CLR)
    fig.add_annotation(x=8.85, y=6.3, text='Efluente C1→C2', showarrow=False,
                       font=dict(size=9, color=ARROW_CLR), xanchor='center')

    # ─────────────────────────────────────────────────────────────────────────
    # BLOQUE: CÁMARA 2 – ELECTRODEPOSICIÓN
    # ─────────────────────────────────────────────────────────────────────────
    fig.add_shape(type='rect', x0=10.2, y0=2.5, x1=14.0, y1=9.5,
                  fillcolor=C2_LIGHT, line=dict(color=C2_COLOR, width=3),
                  layer='below')
    fig.add_shape(type='circle', x0=10.6, y0=8.8, x1=11.6, y1=9.8,
                  fillcolor=C2_COLOR, line_width=0)
    fig.add_annotation(x=11.1, y=9.3, text='<b>2</b>', showarrow=False,
                       font=dict(size=16, color='white'), xanchor='center')
    fig.add_annotation(x=12.1, y=9.3, text='<b>CÁMARA 2</b>', showarrow=False,
                       font=dict(size=13, color=C2_COLOR), xanchor='center')
    fig.add_annotation(x=12.1, y=8.7, text='Electrodeposición', showarrow=False,
                       font=dict(size=11, color='#c2410c'), xanchor='center')
    fig.add_annotation(x=12.1, y=8.3, text=f'Control {mode_pwm[:3]}', showarrow=False,
                       font=dict(size=10, color='#c2410c'), xanchor='center')
    fig.add_annotation(x=12.1, y=7.7, text='15 – 75 min', showarrow=False,
                       font=dict(size=10, color='#64748b', family='monospace'), xanchor='center')

    # Celda electroquímica (ánodo / cátodo)
    fig.add_shape(type='rect', x0=10.6, y0=4.5, x1=13.6, y1=7.0,
                  fillcolor='#fed7aa', line=dict(color=C2_COLOR, width=1.5),
                  layer='below')
    # Electrodos
    for xp, lbl, clr in [(11.1, '(+)\nÁNODO', '#dc2626'), (13.1, '(−)\nCÁTODO', '#1d4ed8')]:
        fig.add_shape(type='rect', x0=xp - 0.12, y0=4.6, x1=xp + 0.12, y1=6.9,
                      fillcolor=clr, line_width=0)
        fig.add_annotation(x=xp, y=7.15, text=lbl, showarrow=False,
                           font=dict(size=8, color=clr), xanchor='center')
    # Burbujas en el ánodo
    for bx, by in [(11.3, 5.3), (11.5, 5.8), (11.2, 6.2), (11.4, 6.5)]:
        fig.add_shape(type='circle', x0=bx - 0.1, y0=by - 0.1, x1=bx + 0.1, y1=by + 0.1,
                      fillcolor='rgba(147,197,253,0.7)', line=dict(color='#3b82f6', width=1))
    # Flecha de corriente (PWM)
    fig.add_annotation(x=12.1, y=4.0, text=f'⚡ {mode_pwm}',
                       showarrow=False, font=dict(size=9, color='#ea580c'), xanchor='center')
    fig.add_annotation(x=12.1, y=3.5, text=f'E consumida: {energy_c2:.2f} Wh',
                       showarrow=False, font=dict(size=9, color='#ea580c', family='monospace'), xanchor='center')

    # KPIs salida C2
    for i, (lbl, val, clr) in enumerate([
        ('pH:',        f'{pH_c2_out:.2f}',         '#1d4ed8'),
        ('[M²⁺]:',     f'{CM_c2_out:.1f} mg/L',    '#b45309'),
        ('Dep.:',      f'{m_dep_c2:.1f} mg',        '#059669'),
        ('η_F:',       f'{eta_c2_pct:.0f} %',       '#16a34a'),
    ]):
        fig.add_annotation(x=10.7 + (i % 2) * 1.65, y=3.2 - (i // 2) * 0.55,
                           text=f'<b>{lbl}</b> {val}', showarrow=False,
                           font=dict(size=9, color=clr), xanchor='left')

    # Descarga: metal en malla
    fig.add_shape(type='rect', x0=11.3, y0=1.4, x1=12.9, y1=2.2,
                  fillcolor='#fdba74', line=dict(color=C2_COLOR, width=1.5),
                  layer='below')
    fig.add_annotation(x=12.1, y=2.1, text='🔩 M⁰ en malla', showarrow=False,
                       font=dict(size=9, color='#7c2d12'), xanchor='center')
    fig.add_annotation(x=12.1, y=1.65, text=f'{m_dep_c2:.1f} mg depositados', showarrow=False,
                       font=dict(size=9, color='#7c2d12'), xanchor='center')
    fig.add_shape(type='line', x0=12.1, y0=2.5, x1=12.1, y1=2.2,
                  line=dict(color=C2_COLOR, width=2, dash='dot'))

    # Flecha C2 → C3
    fig.add_annotation(x=16.7, y=6.0, ax=14.0, ay=6.0,
                       xref='x', yref='y', axref='x', ayref='y',
                       showarrow=True, arrowhead=2, arrowsize=1.4,
                       arrowwidth=3, arrowcolor=ARROW_CLR)
    fig.add_annotation(x=15.35, y=6.3, text='Efluente C2→C3', showarrow=False,
                       font=dict(size=9, color=ARROW_CLR), xanchor='center')

    # ─────────────────────────────────────────────────────────────────────────
    # BLOQUE: CÁMARA 3 – BIORREACTOR
    # ─────────────────────────────────────────────────────────────────────────
    fig.add_shape(type='rect', x0=16.7, y0=2.5, x1=20.5, y1=9.5,
                  fillcolor=C3_LIGHT, line=dict(color=C3_COLOR, width=3),
                  layer='below')
    fig.add_shape(type='circle', x0=17.1, y0=8.8, x1=18.1, y1=9.8,
                  fillcolor=C3_COLOR, line_width=0)
    fig.add_annotation(x=17.6, y=9.3, text='<b>3</b>', showarrow=False,
                       font=dict(size=16, color='white'), xanchor='center')
    fig.add_annotation(x=18.6, y=9.3, text='<b>CÁMARA 3</b>', showarrow=False,
                       font=dict(size=13, color=C3_COLOR), xanchor='center')
    fig.add_annotation(x=18.6, y=8.7, text='Biorreactor', showarrow=False,
                       font=dict(size=11, color='#15803d'), xanchor='center')
    fig.add_annotation(x=18.6, y=8.3, text='P. putida + FF+PID', showarrow=False,
                       font=dict(size=10, color='#15803d'), xanchor='center')
    fig.add_annotation(x=18.6, y=7.7, text='75 – 120 min', showarrow=False,
                       font=dict(size=10, color='#64748b', family='monospace'), xanchor='center')

    # Representación del biorreactor (bacteria + soporte)
    fig.add_shape(type='rect', x0=17.1, y0=4.5, x1=20.1, y1=7.0,
                  fillcolor='#bbf7d0', line=dict(color=C3_COLOR, width=1.5),
                  layer='below')
    # Partículas de tezontle con bacterias
    for bx, by in [(17.7, 5.0), (18.2, 5.5), (18.8, 5.1), (19.3, 5.8),
                   (17.5, 6.0), (18.5, 6.4), (19.5, 6.3), (18.0, 6.6)]:
        fig.add_shape(type='circle', x0=bx - 0.18, y0=by - 0.18, x1=bx + 0.18, y1=by + 0.18,
                      fillcolor='#4ade80', line=dict(color='#15803d', width=1.5))
    fig.add_annotation(x=18.6, y=7.15, text='🦠 P. putida / tezontle', showarrow=False,
                       font=dict(size=8, color='#15803d'), xanchor='center')

    fig.add_annotation(x=18.6, y=4.0, text='OH⁻ dosificado (FF+PID)',
                       showarrow=False, font=dict(size=9, color='#0d9488'), xanchor='center')
    fig.add_annotation(x=18.6, y=3.5, text='Setpoint pH = 7.0',
                       showarrow=False, font=dict(size=9, color='#0d9488', family='monospace'), xanchor='center')

    # KPIs salida C3
    CM_clr = '#16a34a' if CM_ok else '#dc2626'
    pH_clr = '#16a34a' if pH_ok else '#dc2626'
    for i, (lbl, val, clr) in enumerate([
        ('pH:',    f'{pH_c3_out:.2f}',        pH_clr),
        ('[M²⁺]:', f'{CM_c3_out:.2f} mg/L',  CM_clr),
        ('[EC]:',  f'{EC_c3_out:.0f} mg/L',   '#0284c7'),
        ('Li⁺:',   f'{Li_c3:.0f} mg/L ✓',   '#7c3aed'),
    ]):
        fig.add_annotation(x=17.2 + (i % 2) * 1.65, y=3.2 - (i // 2) * 0.55,
                           text=f'<b>{lbl}</b> {val}', showarrow=False,
                           font=dict(size=9, color=clr), xanchor='left')

    # ─────────────────────────────────────────────────────────────────────────
    # BLOQUE: SALIDA (efluente tratado)
    # ─────────────────────────────────────────────────────────────────────────
    # Flecha C3 → Salida
    fig.add_annotation(x=21.5, y=6.0, ax=20.5, ay=6.0,
                       xref='x', yref='y', axref='x', ayref='y',
                       showarrow=True, arrowhead=2, arrowsize=1.4,
                       arrowwidth=3, arrowcolor='#22c55e' if CM_ok else '#dc2626')

    out_bg = '#f0fdf4' if (CM_ok and pH_ok) else '#fef2f2'
    out_border = '#22c55e' if (CM_ok and pH_ok) else '#dc2626'
    fig.add_shape(type='rect', x0=21.3, y0=4.5, x1=22.0, y1=7.5,
                  fillcolor=out_bg, line=dict(color=out_border, width=2.5),
                  layer='below')
    status_icon = '✅' if (CM_ok and pH_ok) else '⚠️'
    fig.add_annotation(x=21.65, y=7.8, text=f'<b>SALIDA</b>', showarrow=False,
                       font=dict(size=10, color='#1f2328'), xanchor='center')
    for i, (lbl, val, clr) in enumerate([
        (f'{status_icon} pH', f'{pH_c3_out:.2f}', pH_clr),
        ('[M²⁺]', f'{CM_c3_out:.2f}', CM_clr),
        ('Li⁺', f'{Li_c3:.0f} mg/L', '#7c3aed'),
    ]):
        fig.add_annotation(x=21.65, y=7.2 - i * 0.65,
                           text=f'<b>{lbl}</b><br>{val}', showarrow=False,
                           font=dict(size=9, color=clr), xanchor='center')

    # ─────────────────────────────────────────────────────────────────────────
    # LEYENDA DE PROCESOS (parte superior)
    # ─────────────────────────────────────────────────────────────────────────
    for lx, ly, lc, ltxt in [
        (4.5, 11.2, C1_COLOR, '🧪 Neutralización · pH Control PI · CaF₂'),
        (11.0, 11.2, C2_COLOR, '⚡ Electrodeposición · Malla inox 316L · PWM'),
        (17.5, 11.2, C3_COLOR, '🧬 Biorreactor · P. putida · FF+PID'),
    ]:
        fig.add_shape(type='rect', x0=lx - 0.3, y0=10.8, x1=lx + 0.5, y1=11.5,
                      fillcolor=lc, line_width=0, layer='below')
        fig.add_annotation(x=lx + 0.7, y=11.2, text=ltxt, showarrow=False,
                           font=dict(size=10, color='#1e293b'), xanchor='left')

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_xaxes(visible=False, range=[0, 22.5])
    fig.update_yaxes(visible=False, range=[0.8, 12.5])
    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor=BG_CLR,
        paper_bgcolor='white',
        title=dict(
            text='<b>Diagrama de Proceso – Sistema Bioelectroquímico Automatizado de 3 Cámaras</b>',
            font=dict(size=14, color='#1e293b'),
            x=0.5, xanchor='center',
        ),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# INTERFAZ STREAMLIT – MEJORADA CON EXPLICACIONES
# ─────────────────────────────────────────────────────────────────────────────

EXPLAINER_CSS = """
<style>
/* ── Base ─────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    font-size: 14px;
    line-height: 1.6;
}

/* ── Info boxes ──────────────────────────────────────────────────── */
.explainer-box {
    background: #f0f9ff !important;
    border-left: 4px solid #3b82f6 !important;
    border-radius: 8px !important;
    padding: 14px 18px !important;
    margin: 8px 0 16px 0 !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
    color: #1e3a5f !important;
}
.explainer-box b { color: #1d4ed8 !important; }
.explainer-box span { color: inherit !important; }
.warn-box {
    background: #fff7ed !important;
    border-left: 4px solid #f97316 !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    margin: 8px 0 16px 0 !important;
    font-size: 13px !important;
    line-height: 1.65 !important;
    color: #7c2d12 !important;
}
.warn-box b { color: #7c2d12 !important; }
.success-box {
    background: #f0fdf4 !important;
    border-left: 4px solid #22c55e !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    margin: 8px 0 16px 0 !important;
    font-size: 13px !important;
    line-height: 1.65 !important;
    color: #14532d !important;
}
.success-box b { color: #14532d !important; }

/* ── KPI cards ───────────────────────────────────────────────────── */
.kpi-card {
    background: #f7f8fa;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 16px 12px 14px 12px;
    text-align: center;
}
.kpi-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: #57606a;
    text-transform: uppercase;
    margin-bottom: 8px;
    display: block;
}
.kpi-value {
    font-size: 22px;
    font-weight: 800;
    color: #1f2328;
    line-height: 1.15;
    display: block;
    margin-bottom: 6px;
    word-break: break-word;
}
.kpi-hint {
    font-size: 11px;
    color: #6b7280;
    display: block;
    line-height: 1.4;
}

/* ── Flow cards (tab 0) ──────────────────────────────────────────── */
.flow-card {
    border-radius: 8px;
    padding: 14px 16px;
    font-size: 13px;
    line-height: 1.9;
}

/* ── Step badge ──────────────────────────────────────────────────── */
.step-badge {
    display: inline-block;
    background: #1d4ed8;
    color: white;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    text-align: center;
    line-height: 24px;
    font-weight: 700;
    margin-right: 8px;
    font-size: 13px;
}

/* ── Sidebar ─────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
section[data-testid="stSidebar"] .stSlider > label { font-size: 13px; font-weight: 600; }
section[data-testid="stSidebar"] h3 { font-size: 13px; color: #374151; margin-top: 0.4rem; }
</style>
"""


def render_info_box(text: str, kind: str = "info"):
    cls = {"info": "explainer-box", "warn": "warn-box", "ok": "success-box"}.get(kind, "explainer-box")
    st.markdown(f"<div class='{cls}'>{text}</div>", unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title='Gemelo Digital – Sistema Bioelectroquímico 3 Cámaras',
        page_icon='⚗️',
        layout='wide',
    )

    st.markdown(EXPLAINER_CSS, unsafe_allow_html=True)

    # ── Cabecera ─────────────────────────────────────────────────────────────
    st.markdown("""
         <div style='background:linear-gradient(135deg,#1e3a5f 0%,#0f766e 100%);
                padding:26px 32px 22px 32px;border-radius:12px;margin-bottom:16px;
                box-shadow:0 2px 12px rgba(0,0,0,0.15)'>
      <h1 style='color:white;margin:0 0 8px 0;font-size:22px;font-weight:800;
                 letter-spacing:-0.01em;line-height:1.3'>
        ⚗️&nbsp; Gemelo Digital
        <span style='font-weight:400;font-size:18px'>&ndash; Sistema Bioelectroquímico Automatizado de 3 Cámaras</span>
      </h1>
      <div style='display:flex;flex-wrap:wrap;gap:10px;margin-top:4px'>
        <span style='background:rgba(255,255,255,0.15);color:#e0f2fe;border-radius:20px;
                     padding:3px 12px;font-size:12px;font-weight:500'>
          🔩 Recuperación de Metales
        </span>
        <span style='background:rgba(255,255,255,0.15);color:#d1fae5;border-radius:20px;
                     padding:3px 12px;font-size:12px;font-weight:500'>
          🧬 Biorremediación · <em>P. putida</em>
        </span>
        <span style='background:rgba(255,255,255,0.15);color:#fef3c7;border-radius:20px;
                     padding:3px 12px;font-size:12px;font-weight:500'>
          🔋 Efluente Baterías Ion-Litio
        </span>
        <span style='background:rgba(255,255,255,0.15);color:#ede9fe;border-radius:20px;
                     padding:3px 12px;font-size:12px;font-weight:500'>
          ⚗️ Base Sulfato – Cero Cloruros
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Descripción general del sistema ──────────────────────────────────────
    with st.expander("📖  ¿Qué simula esta aplicación? — Guía rápida", expanded=False):
        st.markdown("""
**Este simulador modela un reactor de laboratorio (1 L) que trata el efluente ácido de baterías de ion-litio agotadas.**
El objetivo es recuperar el metal disuelto (Cu²⁺ como modelo), eliminar el flúor tóxico y limpiar el agua antes de descargarla.

El tren de tratamiento tiene **tres etapas en serie**:

| Etapa | Tiempo | Proceso principal | Variable clave |
|-------|--------|-------------------|----------------|
| 🧪 Cámara 1 | 0 – 15 min | Neutralización con Ca(OH)₂ → precipita CaF₂ | pH sube de ~2 a 4 |
| ⚡ Cámara 2 | 15 – 75 min | Electrodeposición del metal en malla inox 316L | Corriente PWM vs fija |
| 🧬 Cámara 3 | 75 – 120 min | Biorreactor (*P. putida*) + control PID híbrido | pH sube a 7, biosorción |

**¿Qué puedes explorar?**
- Cómo cambia el proceso al variar la concentración inicial de metal o flúor.
- Si conviene usar **corriente fija** o **PWM exponencial** en la electrocelda (eficiencia y ahorro energético).
- Cómo responde el biorreactor ante una perturbación ácida repentina a los 95 min.
- La ventaja de usar bacterias (*P. putida*) frente a solo precipitación química.

> 💡 **Litio (Li⁺):** su potencial de reducción es −3.04 V, muy por debajo del rango de la celda. **No se deposita ni se pierde** en ninguna de las etapas — queda 100% disponible para recuperación posterior.
        """)

    # ── Barra lateral ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.header('⚙️ Parámetros de Simulación')

        # --- Condiciones de entrada ---
        st.subheader('📥 Efluente de Entrada')
        st.caption('Concentraciones típicas de un efluente de batería Li-ion agotada en medio sulfato.')

        metal_sel = st.selectbox(
            'Metal a recuperar',
            list(METAL_PROPS.keys()),
            index=0,
            help='Selecciona el metal principal del efluente. Cambia la masa molar, el potencial de celda Nernst, '
                 'los parámetros de biosorción Langmuir (q_max, K_L) y la cinética de precipitación química.'
        )
        _M_g, _z, _E0, _KL, _qmax, _kppco, _ = METAL_PROPS[metal_sel]
        render_info_box(
            f"<b>{metal_sel}</b>: M = {_M_g*1e3:.2f} g/mol · z = {_z} · "
            f"E₀_celda = {_E0:.2f} V · K_L = {_KL} L/mg · q_max = {_qmax} mg/g"
        )

        C_M_in = st.slider(f'{metal_sel} inicial (mg/L)', 100, 500, 250, 10,
                           help=f'Concentración de {metal_sel} disuelto. Rango real en efluentes de baterías: 100–500 mg/L.')
        C_F_in = st.slider('Flúor F⁻ inicial (mg/L)', 100, 500, 250, 10,
                           help='Flúor disuelto como HF/F⁻. Proviene del electrolito LiPF₆. El límite de descarga es 1.5 mg/L.')
        pH_in  = st.slider('pH inicial', 1.5, 3.0, 2.0, 0.05,
                           help='pH muy ácido típico del efluente de baterías. La Cámara 1 lo lleva a 4.0.')

        # Contexto dinámico
        if pH_in < 2.0:
            render_info_box("⚠️ pH muy bajo. El controlador PI necesitará más Ca(OH)₂ y tomará más tiempo en alcanzar el setpoint.", "warn")
        if C_F_in > 350:
            render_info_box("⚠️ Concentración de flúor alta. Se generará más CaF₂ sólido. Verifica la eficiencia de precipitación en la Cámara 1.", "warn")

        st.divider()

        # --- Cámara 2 ---
        st.subheader('⚡ Control Eléctrico – Cámara 2')
        st.caption('La electrocelda deposita el metal sobre una malla de acero inoxidable 316L (440 cm²).')

        mode_pwm = st.selectbox(
            'Modo de Corriente',
            ['PWM Exponencial', 'Corriente Fija'],
            help='**PWM Exponencial (recomendado):** ajusta la corriente al límite de difusión para mantener η_F = 85%. '
                 '**Corriente Fija:** aplica siempre el mismo amperaje, más sencillo pero menos eficiente al final del proceso.'
        )
        I_fixed = st.slider(
            'Corriente de referencia (A)', 0.1, 1.0, 0.45, 0.05,
            help='En modo PWM: es el máximo de la rampa. En modo Fija: es la corriente constante aplicada. '
                 'Valores > I_límite generan burbujas de H₂ (desperdicio energético).'
        )

        if mode_pwm == 'Corriente Fija':
            render_info_box(
                "ℹ️ Con corriente fija, cuando la concentración de metal baja, "
                "la corriente supera el límite de difusión → se reduce H₂O en lugar de Cu²⁺ → "
                "eficiencia faradaica cae por debajo del 85%.", "warn"
            )
        else:
            render_info_box(
                "✅ El PWM exponencial sigue la corriente límite de difusión (I_lim = z·F·A·k_m·C), "
                "manteniendo η_F = 85% durante todo el proceso.", "ok"
            )

        st.divider()

        # --- Cámara 3 ---
        st.subheader('🧬 Control PID – Cámara 3')
        st.caption('Ganancias del controlador PID para el pH del biorreactor. El setpoint es pH = 7.0.')

        with st.expander("ℹ️ ¿Qué hacen Kp, Ki y Kd?"):
            st.markdown("""
- **Kp (proporcional):** reacción inmediata al error de pH. Valores altos = respuesta rápida pero puede oscilar.
- **Ki (integral):** elimina el error en estado estacionario. Valores altos pueden provocar sobreoscilación.
- **Kd (derivativo):** amortigua cambios bruscos (ej. perturbaciones ácidas). Mejora la estabilidad.

La estrategia **FF+PID** añade un término de *feedforward* (compensación anticipada del flujo ácido de entrada),
lo que hace la respuesta más rápida ante perturbaciones que el PID solo.
            """)

        Kp_pid = st.slider('Kp (proporcional)', 0.1, 5.0, 1.2, 0.1)
        Ki_pid = st.slider('Ki (integral)',      0.01, 1.0, 0.15, 0.01)
        Kd_pid = st.slider('Kd (derivativo)',    0.0, 0.5, 0.05, 0.01)

        st.divider()
        run_btn = st.button('▶ Ejecutar Simulación', type='primary', use_container_width=True)

    # ── Ejecución / caché ──────────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def run_all(C_M_in, C_F_in, pH_in, I_fixed, Kp_pid, Ki_pid, Kd_pid, metal, mode_pwm):
        # C_SO4_mM estimado desde pH inicial considerando disociación parcial de HSO₄⁻ (pKa2 = 1.99)
        alpha_so4_in = 1.0 / (1.0 + 10.0 ** (PKA_HSO4 - pH_in))
        C_SO4_mM_est = max((10.0 ** (-pH_in)) / (1.0 + alpha_so4_in) * 1000.0, 0.1)
        r1 = simulate_chamber1(C_F_in, C_M_in, pH_in, C_SO4_mM=C_SO4_mM_est)
        pH_c1_end = float(r1['pH'][-1])
        r2 = simulate_chamber2(r1['C_M'][-1], pH2_in=pH_c1_end,
                                I_fixed=I_fixed, metal=metal)
        c3_M_key = 'C_M_pwm' if mode_pwm == 'PWM Exponencial' else 'C_M_fix'
        c3_M_in  = float(r2[c3_M_key][-1])
        # El ion SO₄²⁻ no se consume en C2; pasa íntegro a C3
        r3 = simulate_chamber3(c3_M_in, pH3_in=pH_c1_end,
                                Kp_pid=Kp_pid, Ki_pid=Ki_pid, Kd_pid=Kd_pid,
                                metal=metal, C_SO4_mM=C_SO4_mM_est)
        return r1, r2, r3

    if run_btn or 'sim_results' not in st.session_state:
        with st.spinner('🔬 Resolviendo ODEs del sistema bioelectroquímico…'):
            r1, r2, r3 = run_all(C_M_in, C_F_in, pH_in, I_fixed, Kp_pid, Ki_pid, Kd_pid, metal_sel, mode_pwm)
        st.session_state['sim_results'] = (r1, r2, r3)
    else:
        r1, r2, r3 = st.session_state['sim_results']

    # ── KPIs ───────────────────────────────────────────────────────────────────
    V_L        = V_CHAMBER * 1e3                                # m³ → L (1.0 L)
    F_abatido  = C_F_in - float(r1['C_F'][-1])
    # CaF2 [mg/L] × V_L [L] × 1e-3 [g/mg] = g de CaF₂
    g_CaF2     = float(r1['CaF2'][-1]) * V_L * 1e-3
    m_dep_pwm  = float(r2['m_dep_pwm'][-1])
    m_dep_fix  = float(r2['m_dep_fix'][-1])
    m_dep_sel  = m_dep_pwm if mode_pwm == 'PWM Exponencial' else m_dep_fix
    pct_ahorro = max(0.0, (r2['energy_fix'][-1] - r2['energy_pwm'][-1]) /
                     (r2['energy_fix'][-1] + 1e-9) * 100.0)
    C_M_final  = float(r3['ff_pid']['C_M'][-1])
    pH_final   = float(r3['ff_pid']['pH'][-1])

    st.markdown(
        "<p class='section-title' style='font-size:17px;font-weight:700;color:#1f2328;"
        "margin:20px 0 4px 0;padding-bottom:8px;border-bottom:2px solid #e5e7eb'>"
        "📊 Indicadores Clave de Desempeño (KPIs)</p>",
        unsafe_allow_html=True
    )
    render_info_box(
        "Resultados <b>al final del proceso completo (t = 120 min)</b>. "
        "Verde = objetivo cumplido."
    )

    # Primera fila — 3 KPIs
    r_a, r_b, r_c = st.columns(3, gap="medium")
    # Segunda fila — 3 KPIs
    r_d, r_e, r_f = st.columns(3, gap="medium")

    def _kpi(col, icon, label, value, hint, accent="#1f2328"):
        col.markdown(
            f"<div class='kpi-card'>"
            f"<span class='kpi-label'>{icon} {label}</span>"
            f"<span class='kpi-value' style='color:{accent}'>{value}</span>"
            f"<span class='kpi-hint'>{hint}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _kpi(r_a, '🧪', 'F⁻ abatido C1',  f'{F_abatido:.1f} mg/L',
         f'de {C_F_in} mg/L iniciales', '#b45309')
    _kpi(r_b, '⬜', 'CaF₂ generado',   f'{g_CaF2:.2f} g',
         'subproducto sólido recuperable', '#059669')
    _kpi(r_c, '🔩', 'Metal en malla',  f'{m_dep_sel:.1f} mg',
         f'electrodeposición ({mode_pwm})', '#1d4ed8')
    _kpi(r_d, '⚡', 'Ahorro energía',  f'{pct_ahorro:.1f} %',
         'PWM vs corriente fija', '#7c3aed')
    _kpi(r_e, '🏁', 'Metal salida C3', f'{C_M_final:.2f} mg/L',
         '✅ < 1.2 mg/L objetivo' if C_M_final <= 1.2 else '⚠️ > 1.2 mg/L',
         '#16a34a' if C_M_final <= 1.2 else '#dc2626')
    _kpi(r_f, '🔋', 'Li⁺ recuperable', '200.0 mg/L',
         '100% conservado (E₀ = −3.04 V)', '#7c3aed')

    # Balance de masa en Cámara 2: C [mg/L] × V_L [L] = masa [mg]
    cm2_sel_key = 'C_M_pwm' if mode_pwm == 'PWM Exponencial' else 'C_M_fix'
    m_in_C2     = float(r1['C_M'][-1]) * V_L
    m_aq_fin    = float(r2[cm2_sel_key][-1]) * V_L
    balance_ok  = abs((m_in_C2 - m_aq_fin) - m_dep_sel) < 1.0
    st.caption(
        f"{'✅' if balance_ok else '⚠️'} Balance de masa C2 — "
        f"entrada: {m_in_C2:.1f} mg | depositado: {m_dep_sel:.1f} mg | "
        f"en solución: {m_aq_fin:.1f} mg | pH final C3: {pH_final:.2f} | Li⁺: 200.0 mg/L (intacto)"
    )

    st.divider()

    # ── Pestañas ───────────────────────────────────────────────────────────────
    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        '🏭 Diagrama del Sistema',
        '📊 Vista General del Tren',
        '⚗️ Cámara 1 – Pretratamiento',
        '⚡ Cámara 2 – Electrocelda',
        '🧬 Cámara 3 – Biorreactor',
    ])

    # ─── TAB 0: Diagrama de proceso ────────────────────────────────────────────
    with tab0:
        render_info_box(
            "<b>Diagrama de Proceso del Sistema (PFD) – valores en tiempo real.</b> "
            "Cada bloque representa una cámara del tren de tratamiento. "
            "Las <b>flechas</b> muestran el flujo del efluente entre etapas. "
            "Los <b>valores de salida</b> se actualizan con cada simulación. "
            "Los colores de los KPIs son <span style='color:#16a34a'><b>verde ✅</b></span> si se cumple el objetivo "
            "o <span style='color:#dc2626'><b>rojo</b></span> si no."
        )
        st.plotly_chart(
            fig_process_diagram(r1, r2, r3, mode_pwm, C_M_in, C_F_in, pH_in),
            use_container_width=True,
        )
        # Tabla resumen del flujo inter-cámaras
        st.markdown("<p style='font-size:15px;font-weight:700;margin:16px 0 10px 0;color:#374151'>"
                    "🔄 Flujo de Variables entre Cámaras</p>",
                    unsafe_allow_html=True)
        col_d1, col_d2, col_d3 = st.columns(3, gap="medium")
        with col_d1:
            st.markdown(
                f"<div style='background:#e0f2fe;border-left:4px solid #0ea5e9;border-radius:8px;"
                f"padding:14px 16px;font-size:13px;line-height:1.9;color:#1e3a5f'>"
                f"<b style='color:#0369a1;font-size:14px'>🧪 CÁMARA 1 — Salida</b><br><br>"
                f"<span style='color:#1e3a5f'>pH &nbsp;→&nbsp; <b>{float(r1['pH'][-1]):.2f}</b> "
                f"<span style='color:#475569;font-size:12px'>(setpoint 4.00)</span></span><br>"
                f"<span style='color:#1e3a5f'>[F⁻] &nbsp;→&nbsp; <b>{float(r1['C_F'][-1]):.1f} mg/L</b></span><br>"
                f"<span style='color:#1e3a5f'>[M²⁺] →&nbsp; <b>{float(r1['C_M'][-1]):.0f} mg/L</b></span><br>"
                f"<span style='color:#1e3a5f'>CaF₂ →&nbsp; <b>{float(r1['CaF2'][-1]):.0f} mg/L</b> precipitado</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_d2:
            cm2_val = float(r2['C_M_pwm' if mode_pwm == 'PWM Exponencial' else 'C_M_fix'][-1])
            en2_val = float(r2['energy_pwm' if mode_pwm == 'PWM Exponencial' else 'energy_fix'][-1])
            md2_val = float(r2['m_dep_pwm' if mode_pwm == 'PWM Exponencial' else 'm_dep_fix'][-1])
            st.markdown(
                f"<div style='background:#fff7ed;border-left:4px solid #f97316;border-radius:8px;"
                f"padding:14px 16px;font-size:13px;line-height:1.9;color:#431407'>"
                f"<b style='color:#c2410c;font-size:14px'>⚡ CÁMARA 2 — Salida</b><br><br>"
                f"<span style='color:#431407'>[M²⁺] →&nbsp; <b>{cm2_val:.1f} mg/L</b></span><br>"
                f"<span style='color:#431407'>Cu⁰ dep. →&nbsp; <b>{md2_val:.1f} mg</b></span><br>"
                f"<span style='color:#431407'>Energía →&nbsp; <b>{en2_val:.3f} Wh</b></span><br>"
                f"<span style='color:#431407'>Modo →&nbsp; <b>{mode_pwm}</b></span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_d3:
            cm3_val = float(r3['ff_pid']['C_M'][-1])
            ph3_val = float(r3['ff_pid']['pH'][-1])
            ec3_val = float(r3['ff_pid']['C_EC'][-1])
            ok_str  = "✅ Cumple objetivos" if (cm3_val <= 1.2 and 6.5 <= ph3_val <= 7.5) else "⚠️ Revisar parámetros"
            ok_clr  = '#15803d' if (cm3_val <= 1.2 and 6.5 <= ph3_val <= 7.5) else '#dc2626'
            st.markdown(
                f"<div style='background:#f0fdf4;border-left:4px solid #22c55e;border-radius:8px;"
                f"padding:14px 16px;font-size:13px;line-height:1.9;color:#14532d'>"
                f"<b style='color:#15803d;font-size:14px'>🧬 CÁMARA 3 — Salida</b><br><br>"
                f"<span style='color:#14532d'>pH &nbsp;→&nbsp; <b>{ph3_val:.2f}</b> "
                f"<span style='color:#475569;font-size:12px'>(setpoint 7.00)</span></span><br>"
                f"<span style='color:#14532d'>[M²⁺] →&nbsp; <b>{cm3_val:.2f} mg/L</b></span><br>"
                f"<span style='color:#14532d'>EC biodeg. →&nbsp; <b>{300 - ec3_val:.0f} mg/L</b></span><br>"
                f"<span style='color:{ok_clr}'><b>{ok_str}</b></span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ─── TAB 1: Vista general ──────────────────────────────────────────────────
    with tab1:
        render_info_box(
            "<b>Vista panorámica del proceso completo (0 → 120 min).</b> "
            "La gráfica superior muestra cómo el pH sube escalonadamente en cada cámara. "
            "La inferior muestra la reducción de metal y flúor a lo largo del tren. "
            "Las líneas verticales punteadas marcan las transiciones entre cámaras."
        )
        st.plotly_chart(fig_overview(r1, r2, r3, mode_pwm), use_container_width=True)
        st.caption(
            "El pH objetivo de C1 es 4.0 (precipitación óptima de CaF₂). "
            "El pH objetivo de C3 es 7.0 (biosorción máxima de P. putida y cumplimiento normativo)."
        )

    # ─── TAB 2: Cámara 1 ──────────────────────────────────────────────────────
    with tab2:
        render_info_box(
            "<b>Cámara 1 – Neutralización y Precipitación de Fluoruro (0 – 15 min)</b><br>"
            "Se dosifica <b>Ca(OH)₂</b> mediante un controlador <b>PI</b> para subir el pH de ~2 a 4.0. "
            "A pH 4, el flúor reacciona con Ca²⁺ y precipita como <b>CaF₂</b> (fluorita), "
            "un sólido filtrable y de valor comercial. "
            "El metal M²⁺ permanece 100% disuelto porque su precipitación ocurre a pH > 6.3."
        )
        st.plotly_chart(fig_chamber1(r1), use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric('pH alcanzado (C1)', f'{float(r1["pH"][-1]):.3f}',
                     '→ Setpoint 4.00',
                     help='El controlador PI converge al setpoint de 4.0 sin sobrepasar (anti-windup activo).')
        col_b.metric('[F⁻] residual (C1)', f'{float(r1["C_F"][-1]):.1f} mg/L',
                     f'−{F_abatido:.0f} mg/L desde {C_F_in} mg/L',
                     help='Residual de ~12 mg/L. Se reducirá más en C3 por adsorción biológica.')
        col_c.metric('CaF₂ precipitado', f'{g_CaF2:.3f} g',
                     'subproducto recuperable',
                     help='El CaF₂ es materia prima industrial (óptica, metalurgia). Tiene valor comercial.')

        render_info_box(
            "💡 <b>¿Por qué pH 4 y no más alto?</b> A pH > 5.5 el Cu²⁺ comenzaría a precipitar como Cu(OH)₂, "
            "lo que contaminaría el CaF₂ y dificultaría la electrodeposición posterior. "
            "pH 4 es el punto óptimo para precipitar flúor sin perder metal.", "ok"
        )

    # ─── TAB 3: Cámara 2 ──────────────────────────────────────────────────────
    with tab3:
        render_info_box(
            "<b>Cámara 2 – Electrodeposición de Metal (15 – 75 min)</b><br>"
            "Se aplica una diferencia de potencial que reduce Cu²⁺ + 2e⁻ → Cu⁰ sobre la malla. "
            "La eficiencia <b>faradaica (η_F)</b> indica qué fracción de la corriente hace trabajo útil "
            "(depositar metal) vs generar H₂ por electrólisis del agua. "
            "El <b>control PWM</b> ajusta la corriente al límite de difusión para mantener η_F ≈ 85% "
            "durante todo el proceso, incluso cuando la concentración de metal baja."
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            st.plotly_chart(fig_chamber2(r2), use_container_width=True)
        with col2:
            st.markdown('#### 📈 Resumen C2')
            st.metric('Metal final (PWM)',  f'{float(r2["C_M_pwm"][-1]):.1f} mg/L',
                      f'desde {float(r2["C_M_pwm"][0]):.0f} mg/L',
                      help='Concentración residual de metal en solución al salir de C2.')
            st.metric('Metal final (Fija)', f'{float(r2["C_M_fix"][-1]):.1f} mg/L',
                      help='Mayor residual que PWM porque la eficiencia cae al final.')
            st.divider()
            st.metric('Energía consumida – PWM',  f'{float(r2["energy_pwm"][-1]):.3f} Wh',
                      help='Integración de P = I·V sobre el tiempo del proceso.')
            st.metric('Energía consumida – Fija', f'{float(r2["energy_fix"][-1]):.3f} Wh')
            st.metric('💡 Ahorro PWM vs Fija',    f'{pct_ahorro:.1f} %',
                      delta_color='inverse',
                      help='A igual concentración tratada, el PWM consume menos energía.')
            st.divider()
            st.markdown('**Duty cycle PWM:**')
            st.metric('u_PWM al inicio (t₀)', f'{float(r2["u_pwm"][0]):.3f}',
                      help='Fracción I_PWM / I_ref al comienzo. Alta concentración → alto I_lim → duty alto.')
            st.metric('u_PWM al final (t_f)', f'{float(r2["u_pwm"][-1]):.4f}',
                      help='Al final hay menos metal → I_lim baja → duty cycle cae exponencialmente.')

        render_info_box(
            "💡 <b>¿Por qué usar PWM?</b> Cuando la concentración de metal baja, la <i>corriente límite de difusión</i> "
            "(I_lim = z·F·A·k_m·C) también baja. Si se aplica corriente fija mayor que I_lim, "
            "el exceso genera H₂ (desperdicio energético y burbujas que interfieren con la deposición). "
            "El PWM evita esto siguiendo exactamente el límite.", "ok"
        )

    # ─── TAB 4: Cámara 3 ──────────────────────────────────────────────────────
    with tab4:
        render_info_box(
            "<b>Cámara 3 – Biorreactor con <i>Pseudomonas putida</i> (75 – 120 min)</b><br>"
            "El metal residual de C2 (< 5 mg/L) se elimina por <b>biosorción</b>: "
            "la bacteria secreta polisacáridos extracelulares (EPS) con grupos –COO⁻ "
            "que capturan iones metálicos a pH > 4.8. "
            "Simultáneamente, la bacteria <b>biodegrada el etileno carbonato (EC)</b> del electrolito. "
            "El controlador <b>FF+PID</b> mantiene pH = 7.0 incluso ante perturbaciones ácidas."
        )
        st.plotly_chart(fig_chamber3(r3), use_container_width=True)

        col_c, col_d, col_e = st.columns(3)
        col_c.metric('pH final (FF+PID)', f'{pH_final:.3f}',
                     '→ Setpoint 7.00',
                     help='El controlador FF+PID converge más rápido que el PID solo ante perturbaciones.')
        col_d.metric('[M²⁺] final (biosorción)', f'{C_M_final:.2f} mg/L',
                     f'−{float(r3["ff_pid"]["C_M"][0]) - C_M_final:.1f} mg/L eliminados',
                     help='Con P. putida se alcanza < 1.2 mg/L. Sin bacterias (abiótico) el residual es mucho mayor.')
        col_e.metric('EC biodegradado', f'{300.0 - float(r3["ff_pid"]["C_EC"][-1]):.0f} mg/L',
                     f'de 300 mg/L iniciales',
                     help='El etileno carbonato del electrolito de la batería es biodegradado por P. putida vía crecimiento Monod.')

        with st.expander("🔬 Detalle: ¿Por qué P. putida y no solo precipitación química?"):
            abiotic_final = float(r3['abiotic']['C_M'][-1])
            render_info_box(
                f"A pH 7.0, la precipitación química de Cu(OH)₂ deja un residual de "
                f"<b>~{abiotic_final:.1f} mg/L</b> (cinética lenta, Stumm & Morgan). "
                f"Con <i>P. putida</i> en soporte de tezontle (50 g/L), se alcanza "
                f"<b>{C_M_final:.2f} mg/L</b> — una reducción adicional del "
                f"{max(0,(abiotic_final-C_M_final)/abiotic_final*100):.0f}%. "
                "Los grupos –COO⁻ del EPS tienen alta afinidad por Cu²⁺ (isoterma Langmuir, K_L = 0.35 L/mg).", "ok"
            )

        render_info_box(
            "💡 <b>Perturbación ácida a t = 95 min:</b> simula un pulso de efluente ácido no esperado. "
            "El gráfico de pH (arriba izquierda) muestra cómo el <b>FF+PID</b> rechaza la perturbación "
            "mucho mejor que el PID solo o el lazo abierto. "
            "El término <i>feedforward</i> compensa el flujo ácido de entrada antes de que afecte el pH.", "warn"
        )

    # ── Descarga CSV ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### ⬇️ Exportar Resultados")
    render_info_box(
        "El archivo CSV contiene todos los datos de las 3 cámaras con resolución temporal completa "
        "(450 + 1800 + 1350 puntos). Columnas: tiempo, pH, [F⁻], [M²⁺], corriente, eficiencia, energía, [EC], [Li⁺]."
    )
    df_export = build_export_df(r1, r2, r3, mode_pwm)
    csv_bytes  = df_export.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='⬇️ Descargar Resultados Completos (CSV)',
        data=csv_bytes,
        file_name='simulacion_3camaras.csv',
        mime='text/csv',
        type='secondary',
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='text-align:center;color:#9ca3af;font-size:12px;"
        "border-top:1px solid #e5e7eb;padding-top:12px;margin-top:16px'>"
        "Gemelo Digital · Sistema Bioelectroquímico 3 Cámaras · "
        "<em>Pseudomonas putida</em> · Base Sulfato | Desarrollado con IBM Bob"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
