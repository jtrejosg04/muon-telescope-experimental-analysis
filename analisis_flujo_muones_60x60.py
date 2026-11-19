# ============================================================
# ANÁLISIS DEL FLUJO DE MUONES — MONTAJE 60×60 cm²
# Pipeline definitivo (GitHub)
# ============================================================

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm.auto import tqdm
from pathlib import Path
PICTURES = Path.home() / "Pictures"
PICTURES.mkdir(exist_ok=True)
# ------------------------------------------------------------
# 1. CARGA DE DATOS
# ------------------------------------------------------------
RUTA_BASE = Path(r"D:\2026-10-21 Flujo Muones Placas 60cmX60cm")

CARPETAS = {
    1: "1-1095min-15min",
    2: "2-75min-15min",
    3: "3-110min10min",
    4: "4-10080min-10min"
}

dataframes = {}
for campana, carpeta in CARPETAS.items():
    archivos = sorted((RUTA_BASE / carpeta).rglob("*.txt"))
    print(f"Campaña {campana}: {len(archivos)} archivos")
    partes = []
    for archivo in tqdm(archivos, desc=f"Campaña {campana}"):
        datos = pd.read_csv(archivo, sep=r"\s+", header=None, dtype=str)
        if datos.shape[1] != 9:
            continue
        datos.columns = ["campo_1","campo_2","campo_3","campo_4",
                         "campo_5","campo_6","campo_7","campo_8","timestamp"]
        for col in datos.columns[:8]:
            datos[col] = pd.to_numeric(datos[col].str.replace(",", ".", regex=False), errors="coerce")
        datos["archivo_origen"] = archivo.name
        partes.append(datos)
    dataframes[campana] = pd.concat(partes, ignore_index=True)
    print(f"  → {len(dataframes[campana]):,} eventos")
    del partes

# ------------------------------------------------------------
# 2. VALIDACIÓN DE TIMESTAMPS
# ------------------------------------------------------------
formato = r"\d{2}/\d{2}/\d{2}--\d{2}:\d{2}:\d{2},\d{3}"
for campana in list(dataframes.keys()):
    df = dataframes[campana]
    total = len(df)
    validos = (df["timestamp"].str.fullmatch(formato).fillna(False) &
               pd.to_datetime(df["timestamp"], format="%d/%m/%y--%H:%M:%S,%f", errors="coerce").notna())
    dataframes[campana] = df.loc[validos].reset_index(drop=True)
    print(f"Campaña {campana}: {total:,} → {validos.sum():,} válidos")

# ------------------------------------------------------------
# 3. ACEPTANCIA Y FUNCIONES DE FLUJO
# ------------------------------------------------------------
L = 60.0
G = (L * L * np.pi) * 1e-4
EPS_LV, SIGMA_EPS_LV = 0.952423, 0.002247
FACTOR = G * EPS_LV

def calcular_flujo(N, T_obs):
    return N / (T_obs * FACTOR)

def calcular_incertidumbre(N, T_obs):
    rel = np.sqrt(1/np.maximum(N,1) + (SIGMA_EPS_LV/EPS_LV)**2)
    return calcular_flujo(N, T_obs) * rel

print(f"\nAceptancia: {G:.6f} m² sr")

# ------------------------------------------------------------
# 4. TIEMPO DE OBSERVACIÓN
# ------------------------------------------------------------
def calcular_exposiciones(df, campana):
    nominal = 15*60 if campana in (1,2) else 10*60
    df = df.copy()
    df["fecha_hora"] = pd.to_datetime(df["timestamp"], format="%d/%m/%y--%H:%M:%S,%f")
    tabla = (df.groupby("archivo_origen")
             .agg(inicio=("fecha_hora","min"), fin=("fecha_hora","max"),
                  N_registros=("timestamp","count"))
             .reset_index().sort_values("inicio").reset_index(drop=True))
    tabla["duracion_observada_s"] = (tabla["fin"]-tabla["inicio"]).dt.total_seconds()
    normales = tabla["duracion_observada_s"].between(0.95*nominal, nominal+5)
    tabla["tasa_normal"] = np.where(normales, tabla["N_registros"]/nominal, np.nan)
    tasa_global = tabla["tasa_normal"].median()
    T_obs = []
    for i, fila in tabla.iterrows():
        if normales.iloc[i]:
            T = nominal
        else:
            vecinos = tabla.iloc[max(0,i-5):i+6]
            tasa = vecinos["tasa_normal"].median()
            if not np.isfinite(tasa): tasa = tasa_global
            T = min(fila["duracion_observada_s"] + 2/max(tasa,1e-6), nominal)
        T_obs.append(max(T,1.0))
    tabla["T_obs_s"] = T_obs
    return tabla

exposiciones = {c: calcular_exposiciones(df, c) for c, df in dataframes.items()}

# ------------------------------------------------------------
# 5. CORTES Y FLUJO NATIVO
# ------------------------------------------------------------
def pasa_cortes(df):
    return ((df["campo_1"]<=4095) & (df["campo_3"]<=4095) &
            (df["campo_5"]>=600) & (df["campo_7"]>=600))

serie_flujo = []
for campana in [1,2,3,4]:
    df = dataframes[campana]
    conteos = df.loc[pasa_cortes(df)].groupby("archivo_origen").size().rename("N_mu")
    tabla = exposiciones[campana].copy()
    tabla["N_mu"] = tabla["archivo_origen"].map(conteos).fillna(0).astype(int)
    tabla = tabla.loc[tabla["T_obs_s"]>0].copy()
    tabla["campana"] = campana
    tabla["timestamp"] = tabla["inicio"] + (tabla["fin"]-tabla["inicio"])/2
    tabla["Phi_mu"] = calcular_flujo(tabla["N_mu"], tabla["T_obs_s"])
    tabla["sigma_Phi_mu"] = calcular_incertidumbre(tabla["N_mu"], tabla["T_obs_s"])
    serie_flujo.append(tabla)

serie_flujo = pd.concat(serie_flujo, ignore_index=True).sort_values(["campana","timestamp"]).reset_index(drop=True)
print("Flujo nativo reconstruido.")

# ------------------------------------------------------------
# 6. HISTOGRAMAS
# ------------------------------------------------------------
df_total = pd.concat(dataframes.values(), ignore_index=True)
fig, axes = plt.subplots(2, 1, figsize=(8, 10))
colores = {"C1": "#2563EB", "C2": "#E76F51"}

for ax, campos, titulo, corte, xlim in [
    (axes[0], [("campo_1","C1"),("campo_3","C2")], "Distribuciones TDC — montaje 60×60", 4095, (0,4200)),
    (axes[1], [("campo_5","C1"),("campo_7","C2")], "Distribuciones QDC — montaje 60×60", 600, (0,4100))
]:
    for campo, label in campos:
        ax.hist(df_total[campo].dropna(), bins=150, range=xlim,
                histtype="stepfilled", alpha=0.55, label=label, color=colores[label])
    ax.axvline(corte, color="k", ls="--", lw=1.5, label=f"Corte {'≤' if corte>1000 else '≥'} {corte}")
    ax.set(title=titulo, xlabel="Canal", ylabel="Frecuencia", xlim=xlim)
    ax.legend(); ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("histogramas_TDC_QDC_60x60.pdf", bbox_inches="tight")
plt.show()
del df_total

# ------------------------------------------------------------
# 7. SERIES A RESOLUCIÓN NATIVA
# ------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(14, 12))
for ax, campana in zip(axes, [1,2,3,4]):
    dfp = serie_flujo.loc[serie_flujo["campana"]==campana]
    ax.errorbar(dfp["timestamp"], dfp["Phi_mu"], yerr=dfp["sigma_Phi_mu"],
                fmt="none", ecolor="gray", alpha=0.25, elinewidth=0.6)
    ax.scatter(dfp["timestamp"], dfp["Phi_mu"], s=4, color="#2563EB", alpha=0.85)
    res = "15 min" if campana in [1,2] else "10 min"
    ax.set_title(f"Campaña {campana} — resolución nativa ({res})")
    ax.set_ylabel(r"$\Phi_\mu$ (m$^{-2}$ s$^{-1}$ sr$^{-1}$)")
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n%H:%M"))
    ax.tick_params(axis="x", rotation=30)
axes[-1].set_xlabel("Fecha y hora")
plt.tight_layout()
plt.savefig("flujo_nativo_60x60.pdf", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 8. REFERENCIAS TEMPORALES
# ------------------------------------------------------------
serie_referencia = serie_flujo.copy()
for horas in [4,6,8]:
    for campana in [1,2,3,4]:
        m = serie_referencia["campana"]==campana
        datos = serie_referencia.loc[m].set_index("timestamp")
        serie_referencia.loc[m, f"referencia_{horas}h"] = datos["Phi_mu"].rolling(
            f"{horas}h", center=True, min_periods=1).median().to_numpy()
    serie_referencia[f"desviacion_{horas}h_pct"] = 100*(
        serie_referencia["Phi_mu"]/serie_referencia[f"referencia_{horas}h"] - 1)

# ------------------------------------------------------------
# 9. CANDIDATOS CON ROBUSTEZ ESTRICTA DE VENTANA
# ------------------------------------------------------------
UMBRAL = 3.0
candidatos = []
for campana in [1,2,3,4]:
    df = serie_referencia.loc[serie_referencia["campana"]==campana].copy()
    for h in [4,6,8]:
        sig = np.sqrt(df["sigma_Phi_mu"]**2 + (0.005*df[f"referencia_{h}h"])**2)
        df[f"Z_{h}h"] = (df["Phi_mu"] - df[f"referencia_{h}h"]) / sig
    mask = (df["Z_4h"].abs()>UMBRAL) & (df["Z_6h"].abs()>UMBRAL) & (df["Z_8h"].abs()>UMBRAL)
    for _, row in df.loc[mask].iterrows():
        candidatos.append({
            "campana": campana, "timestamp": row["timestamp"],
            "Phi_mu": row["Phi_mu"], "sigma_Phi_mu": row["sigma_Phi_mu"],
            "Z_4h": row["Z_4h"], "Z_6h": row["Z_6h"], "Z_8h": row["Z_8h"],
            "desv_6h_pct": row["desviacion_6h_pct"]
        })
tabla_candidatos = pd.DataFrame(candidatos)
print(f"\nCandidatos con |Z|>3 en 4h+6h+8h: {len(tabla_candidatos)}")

# ------------------------------------------------------------
# 10. ROBUSTEZ ESTRICTA AL BINEADO + TABLA DEFINITIVA
# ------------------------------------------------------------
def rebinear(df_camp, minutos):
    df = df_camp.sort_values("timestamp").copy()
    df["bin"] = ((df["timestamp"]-df["timestamp"].min()).dt.total_seconds() // (minutos*60))
    g = df.groupby("bin").agg(timestamp=("timestamp","mean"), N_mu=("N_mu","sum"),
                              T_obs_s=("T_obs_s","sum")).reset_index(drop=True)
    g["Phi_mu"] = calcular_flujo(g["N_mu"], g["T_obs_s"])
    g["sigma_Phi_mu"] = calcular_incertidumbre(g["N_mu"], g["T_obs_s"])
    return g

def es_robusto_bineado(ts, campana, ventana=10):
    resoluciones = [20,25] if campana in [1,2] else [15,20]
    ok = 0
    for res in resoluciones:
        reb = rebinear(serie_flujo.loc[serie_flujo["campana"]==campana], res).set_index("timestamp")
        reb["ref"] = reb["Phi_mu"].rolling("6h", center=True, min_periods=1).median()
        reb["Z"] = (reb["Phi_mu"]-reb["ref"]) / np.sqrt(reb["sigma_Phi_mu"]**2 + (0.005*reb["ref"])**2)
        cerca = reb.loc[(reb.index >= ts-pd.Timedelta(minutes=ventana)) &
                        (reb.index <= ts+pd.Timedelta(minutes=ventana))]
        if (cerca["Z"].abs()>3).any():
            ok += 1
    return ok == 2

finales = [ev for _, ev in tabla_candidatos.iterrows() if es_robusto_bineado(ev["timestamp"], ev["campana"])]
tabla_definitiva = pd.DataFrame(finales).reset_index(drop=True)
tabla_definitiva.insert(0, "Evento", [f"E{i+1}" for i in range(len(tabla_definitiva))])

print("\n" + "="*70)
print("TABLA DEFINITIVA")
print("="*70)
print(tabla_definitiva.round(3).to_string(index=False) if len(tabla_definitiva) else "Ningún candidato.")
print("="*70)
tabla_definitiva.to_csv("candidatos_definitivos_60x60.csv", index=False)

# ------------------------------------------------------------
# 11. GRÁFICA FINAL CON EVENTOS MARCADOS
# ------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(15, 14))
for ax, campana in zip(axes, [1,2,3,4]):
    df = serie_referencia.loc[serie_referencia["campana"]==campana].sort_values("timestamp")
    ax.errorbar(df["timestamp"], df["Phi_mu"], yerr=df["sigma_Phi_mu"],
                fmt="none", ecolor="gray", alpha=0.18, elinewidth=0.5)
    ax.scatter(df["timestamp"], df["Phi_mu"], s=3, color="#2563EB", alpha=0.7)
    ax.plot(df["timestamp"], df["referencia_6h"], "k--", lw=1, label="Referencia 6 h")

    cands = tabla_definitiva.loc[tabla_definitiva["campana"]==campana]
    for idx, ev in cands.reset_index(drop=True).iterrows():
        ax.scatter(ev["timestamp"], ev["Phi_mu"], s=45, color="red", zorder=5)
        ox = 12 if idx%2==0 else -35
        oy = 8 if (idx//2)%2==0 else -12
        ax.annotate(f"{ev['Z_6h']:.1f}σ", xy=(ev["timestamp"], ev["Phi_mu"]),
                    xytext=(ox, oy), textcoords="offset points",
                    color="red", fontsize=7, fontweight="bold",
                    ha="left" if ox>0 else "right", va="center")

    res = "15 min" if campana in [1,2] else "10 min"
    ax.set_title(f"Campaña {campana} — resolución nativa ({res})")
    ax.set_ylabel(r"$\Phi_\mu$ (m$^{-2}$ s$^{-1}$ sr$^{-1}$)")
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n%H:%M"))
    ax.tick_params(axis="x", rotation=30)
    if campana==1: ax.legend(fontsize=8)

axes[-1].set_xlabel("Fecha y hora")
fig.suptitle("Candidatos que superaron ambas pruebas de robustez", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("flujo_candidatos_definitivos_60x60.pdf", bbox_inches="tight")
plt.show()


print("\nPipeline completado.")
print("Archivos generados:")
print("  - histogramas_TDC_QDC_60x60.pdf")
print("  - flujo_nativo_60x60.pdf")
print("  - candidatos_definitivos_60x60.csv")
print("  - flujo_candidatos_definitivos_60x60.pdf")