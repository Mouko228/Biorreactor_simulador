# -*- coding: utf-8 -*-
"""
auditoria_sensibilidad.py
Auditoria + sensibilidad OAT (sin Monte Carlo).
"""
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import time, traceback

F_FARADAY = 96485.0
V_L       = 1.0
V_M3      = 1.0e-3
KPS_CAF2  = 3.9e-11
PKA_HF    = 3.17
PKA_HSO4  = 1.99

NOM = dict(
    C_M_in=250.0, C_F_in=250.0, pH_in=2.0, C_EC_in=300.0, C_Li_in=200.0,
    M_g_mol=63.55, z_metal=2, E0_cell=2.00,
    A_cathode=0.044, k_mass=1.8e-5, eta_pwm=0.85, I_fixed=0.45,
    k_ppt=1.5e4, C_F_eq=11.5,
    q_max=2.2, K_L=0.35, m_bio=50.0, k_ads=0.006, pH_pKa_eps=4.8,
    mu_max=2.2e-4, K_s_EC=80.0, Y_x=0.35, K_d=5.0e-6, X0=0.15,
    k_ppco=8e-5, Kp_pid=1.2, Ki_pid=0.15, Kd_pid=0.05,
)

TARGETS = dict(CM_max=1.2, CF_max=15.0, pH_min=6.5, pH_max=7.5, eta_min=0.75)

# Parametros que se varian en sensibilidad y sus rangos fisicos
SENS_PARAMS = {
    'C_M_in':   (100,  500),
    'C_F_in':   (80,   420),
    'A_cathode':(0.030,0.060),
    'k_mass':   (1e-5, 3e-5),
    'eta_pwm':  (0.75, 0.92),
    'q_max':    (1.5,  3.5),
    'K_L':      (0.15, 0.60),
    'mu_max':   (1.5e-4, 4e-4),
    'k_ppco':   (4e-5, 2e-4),
    'C_F_eq':   (8.0,  16.0),
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _alpha_HF(pH):
    return 1.0 / (1.0 + 10.0 ** (pH - PKA_HF))

def _beta(pH, CF=0.0, C_SO4_mM=5.0):
    ln10 = np.log(10.0)
    bw   = ln10 * (10**(-pH) + 10**(pH-14))
    aHF  = _alpha_HF(pH)
    bHF  = ln10 * aHF * (1-aHF) * (CF/1000/19)
    aSO4 = 1/(1+10**(pH-PKA_HSO4))
    bSO4 = ln10 * aSO4*(1-aSO4)*(C_SO4_mM*1e-3)
    return max(bw+bHF+bSO4, 1e-6)

# ── Camara 1 ─────────────────────────────────────────────────────────────────

def sim_c1(p):
    a_so4 = 1/(1+10**(PKA_HSO4-p['pH_in']))
    C_SO4 = max((10**(-p['pH_in']))/(1+a_so4)*1000, 0.1)
    k_pump, u_max = 1.2e-4, 2.2e-4

    def odes(t, y):
        pH,CF,CM,intg,Ca2 = y
        pH=np.clip(pH,0,14); CF=max(CF,0); Ca2=max(Ca2,0)
        err = 4.0-pH
        if pH>=4.0:
            u=0.0; di=0.0
        else:
            u_r = k_pump*max(p['Kp_pid']*err+0.4*intg, 0)
            u   = min(u_r, u_max)
            di  = 0.0 if u_r>=u_max else err
        b    = _beta(pH, CF, C_SO4)
        aHF  = _alpha_HF(pH)
        Fm   = (1-aHF)*max(CF-p['C_F_eq'],0)/1000/19
        SS   = max(Ca2*Fm**2 - KPS_CAF2, 0)
        dCaF2= min(p['k_ppt']*SS, Fm/2)
        dpH  = (u - 2*aHF*dCaF2)/b
        dCF  = -dCaF2*2*19*1000
        return [dpH, dCF, 0.0, di, u/2-dCaF2]

    sol = solve_ivp(odes,(0,900),[p['pH_in'],p['C_F_in'],p['C_M_in'],0,0],
                    t_eval=np.linspace(0,900,300),method='RK45',
                    rtol=1e-5,atol=1e-7,max_step=10.0)
    CF1  = np.maximum(sol.y[1],0)
    pH1  = np.clip(sol.y[0],0,4)
    CaF2 = np.maximum((p['C_F_in']-CF1)*(78.08/38.0),0)
    return dict(pH_out=float(pH1[-1]), CF_out=float(CF1[-1]),
                CM_out=float(p['C_M_in']), CaF2_out=float(CaF2[-1]),
                C_SO4_mM=C_SO4)

# ── Camara 2 ─────────────────────────────────────────────────────────────────

def sim_c2(p, c1):
    M=p['M_g_mol']; z=p['z_metal']
    ff = M*1000/(z*F_FARADAY)

    def Ilim(C):
        return max(z*F_FARADAY*p['A_cathode']*p['k_mass']*max(C/M,0), 1e-9)

    def odes(t,y):
        CM,md,E = y; CM=max(CM,1e-6)
        IL=Ilim(CM); I=p['eta_pwm']*IL
        dCM=-p['eta_pwm']*I*ff/V_L
        Ec=p['E0_cell']-(0.0592/z)*np.log10(max(CM/(M*1000),1e-12))
        return [dCM, p['eta_pwm']*I*ff, I*abs(Ec)/3600]

    sol = solve_ivp(odes,(0,3600),[c1['CM_out'],0,0],
                    t_eval=np.linspace(0,3600,600),method='RK45',
                    rtol=1e-5,atol=1e-7,max_step=10.0)
    CM2 = np.maximum(sol.y[0],0)
    IL_arr = np.array([Ilim(c) for c in CM2])
    return dict(CM_out=float(CM2[-1]), m_dep=float(sol.y[1][-1]),
                energy_Wh=float(sol.y[2][-1]),
                I_max=float(p['eta_pwm']*IL_arr[0]),
                I_min=float(p['eta_pwm']*IL_arr[-1]),
                eta_mean=float(p['eta_pwm']))

# ── Camara 3 ─────────────────────────────────────────────────────────────────

def sim_c3(p, c1, c2):
    pH3_in=c1['pH_out']; C_SO4=c1['C_SO4_mM']
    t_pert=1200.0
    k_pump,u_max=4.5e-6,1.8e-5

    def phi(pH):
        return 1/(1+10**(p['pH_pKa_eps']-pH))

    def b3(pH):
        ln10=np.log(10.0)
        bw=ln10*(10**(-pH)+10**(pH-14))
        aS=1/(1+10**(pH-PKA_HSO4))
        return max(bw+ln10*aS*(1-aS)*C_SO4*1e-3+5e-4, 5e-4)

    def odes(t,y):
        pH,CM,CEC,X,intg,df,q = y
        pH=np.clip(pH,0,14); CM=max(CM,0); CEC=max(CEC,0)
        X=max(X,0.01); q=max(q,0)
        err=7.0-pH; beta=b3(pH)
        q_ap=1.4e-5 if abs(t-t_pert)<30 else 0.0
        H_in=10**(-pH3_in)
        ff_OH=(H_in*5e-6/V_M3)+0.85*q_ap
        u_pid=k_pump*(p['Kp_pid']*err+p['Ki_pid']*(intg/60)+p['Kd_pid']*(df*60))
        u_OH=float(np.clip(u_pid+ff_OH, 0, u_max*1.25))
        if pH>=7.0 and err<=0: u_OH=min(u_OH,q_ap)
        di = 0.0 if abs(err)<0.02 else err
        dpH=(u_OH-q_ap)/beta
        dd=(-dpH-df)/5.0
        qeq=p['q_max']*p['K_L']*CM/(1+p['K_L']*CM)*phi(pH)
        dq=p['k_ads']*(qeq-q)
        dCM=-dq*p['m_bio']-p['k_ppco']/(1+np.exp(-10*(pH-6.5)))*CM
        mu=p['mu_max']*CEC/(p['K_s_EC']+CEC+1e-9)
        dX=(mu-p['K_d'])*X
        return [dpH,dCM,-mu*(X*1000)/p['Y_x'],dX,di,dd,dq]

    sol = solve_ivp(odes,(0,2700),[pH3_in,c2['CM_out'],p['C_EC_in'],p['X0'],0,0,0],
                    t_eval=np.linspace(0,2700,450),method='RK45',
                    rtol=1e-5,atol=1e-7,max_step=10.0)
    return dict(pH_out=float(np.clip(sol.y[0],0,14)[-1]),
                CM_out=float(max(sol.y[1][-1],0)),
                CEC_out=float(max(sol.y[2][-1],0)))

# ── Simulacion completa ───────────────────────────────────────────────────────

def run_sim(override=None):
    p = {**NOM}
    if override: p.update(override)
    try:
        c1=sim_c1(p); c2=sim_c2(p,c1); c3=sim_c3(p,c1,c2)
        m_in=p['C_M_in']; m_fin=c3['CM_out']
        rec=(m_in-m_fin)/(m_in+1e-12)*100
        return dict(
            CF_c1=c1['CF_out'], CaF2=c1['CaF2_out'], pH_c1=c1['pH_out'],
            CM_c2=c2['CM_out'], m_dep=c2['m_dep'], energy=c2['energy_Wh'],
            I_max=c2['I_max'], I_min=c2['I_min'], eta_F=c2['eta_mean'],
            pH_c3=c3['pH_out'], CM_c3=c3['CM_out'], CEC_c3=c3['CEC_out'],
            Li=p['C_Li_in'], rec_pct=rec,
            ok_CM=c3['CM_out']<=TARGETS['CM_max'],
            ok_CF=c1['CF_out']<=TARGETS['CF_max'],
            ok_pH=TARGETS['pH_min']<=c3['pH_out']<=TARGETS['pH_max'],
            ok_all=all([c3['CM_out']<=TARGETS['CM_max'],
                        c1['CF_out']<=TARGETS['CF_max'],
                        TARGETS['pH_min']<=c3['pH_out']<=TARGETS['pH_max']]),
            error=None)
    except Exception as e:
        return dict(error=str(e))

# ── Sensibilidad OAT (solo -20% y +20%) ──────────────────────────────────────

def run_sensitivity():
    base = run_sim()
    rows = []
    kpis = ['CM_c3','CF_c1','pH_c3','energy','rec_pct','m_dep']
    for param,(lo,hi) in SENS_PARAMS.items():
        nom = NOM[param]
        for delta in [-20, +20]:
            val = np.clip(nom*(1+delta/100), lo, hi)
            res = run_sim({param: val})
            if res.get('error'): continue
            row = dict(param=param, delta_pct=delta, value=val)
            for k in kpis:
                row[k] = res[k]
                row[f'sens_{k}'] = (res[k]-base[k])/(abs(base[k])+1e-12)*100
            rows.append(row)
    return pd.DataFrame(rows), base

# ── Ranking de elasticidad ────────────────────────────────────────────────────

def ranking(sens_df):
    rows=[]
    for p in sens_df['param'].unique():
        sub=sens_df[sens_df['param']==p]
        for k in ['CM_c3','CF_c1','pH_c3','energy','rec_pct']:
            col=f'sens_{k}'
            if col in sub: rows.append(dict(param=p,kpi=k,
                elast=sub[col].abs().mean()))
    return pd.DataFrame(rows).sort_values('elast',ascending=False)

# ── Hallazgos de auditoria ────────────────────────────────────────────────────

AUDIT = [
    dict(id='A4', camara='C1', sev='Medio',
         desc='C_F_eq_mgL=11.5 fijo; no depende de [Ca2+] ni fuerza ionica.',
         prop='Calcular C_F_eq = f(Ksp_cond, gamma_Ca, gamma_F) en cada paso.'),
    dict(id='A6', camara='C1', sev='Medio',
         desc='Ca2+ inicial=0; ignora Ca2+ natural del efluente (~0.5-2 mM).',
         prop='Agregar parametro C_Ca2_in [mM] con default 1.0 mM.'),
    dict(id='B3', camara='C2', sev='Medio',
         desc='eta_F_max=0.85 incluso con I_app<<I_lim (regimen cinetico).',
         prop='Usar eta=eta_max*(1-exp(-I_L/I_app*k_kin)) para regimen cinetico.'),
    dict(id='B4', camara='C2', sev='Medio',
         desc='E_cell->inf cuando C_M->0 (Nernst). Energia sobreestimada al final.',
         prop='Limitar E_cell <= E0_cell+0.5 V o usar C_min=1 mg/L operativo.'),
    dict(id='C3', camara='C3', sev='Medio',
         desc='max(X,0.01) oculta colapso real de biomasa.',
         prop='Registrar advertencia si X toca el limite inferior.'),
    dict(id='C5', camara='C3', sev='Medio',
         desc='Filtro derivativo usa -dpH como proxy de de/dt; error en transitorios.',
         prop='Usar estado error_prev y calcular dd_filt=(err-err_prev)/dt-dfilt)/tau_d.'),
    dict(id='C6', camara='C3', sev='ALTO',
         desc='k_ppco(Co2+)=1e-5 s-1 implica precipitacion a pH 7. '
              'Pero Ksp(Co(OH)2)=5.92e-15 => [Co2+]_eq=592 mM a pH 7: NO precipita.',
         prop='Poner k_ppco(Co2+)=0. Remocion de Co2+ depende solo de biosorccion.'),
]

# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("="*60)
    print("AUDITORIA + SENSIBILIDAD OAT")
    print("="*60)

    print("\n[1/3] Simulacion nominal...")
    t0=time.time()
    base=run_sim()
    print(f"  Tiempo: {time.time()-t0:.1f}s")
    if base.get('error'):
        print(f"  ERROR: {base['error']}")
    else:
        ok=lambda b:'OK' if b else 'FAIL'
        print(f"  CM_c3  : {base['CM_c3']:.3f} mg/L   [{ok(base['ok_CM'])} obj<={TARGETS['CM_max']}]")
        print(f"  CF_c1  : {base['CF_c1']:.2f} mg/L  [{ok(base['ok_CF'])} obj<={TARGETS['CF_max']}]")
        print(f"  pH_c3  : {base['pH_c3']:.3f}         [{ok(base['ok_pH'])} obj {TARGETS['pH_min']}-{TARGETS['pH_max']}]")
        print(f"  energia: {base['energy']:.4f} Wh")
        print(f"  rec    : {base['rec_pct']:.1f} %")
        print(f"  CUMPLE : {'SI' if base['ok_all'] else 'NO'}")

    print("\n[2/3] Sensibilidad OAT (+/-20%)...")
    t0=time.time()
    sens_df, base2 = run_sensitivity()
    print(f"  Tiempo: {time.time()-t0:.1f}s  |  filas: {len(sens_df)}")
    rank_df=ranking(sens_df)
    print("  Top 8 por elasticidad:")
    for _,r in rank_df.head(8).iterrows():
        print(f"    {r['param']:14s} -> {r['kpi']:10s}  e={r['elast']:.3f}")

    print("\n[3/3] Exportando...")
    pd.DataFrame([{k:v for k,v in base.items() if k!='error'}]).to_csv('nominal.csv',index=False)
    sens_df.to_csv('sensibilidad.csv',index=False)
    rank_df.to_csv('ranking.csv',index=False)
    pd.DataFrame(AUDIT).to_csv('auditoria.csv',index=False)
    print("  nominal.csv  sensibilidad.csv  ranking.csv  auditoria.csv")
    print("\nDone.")
