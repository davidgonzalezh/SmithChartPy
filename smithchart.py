#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Alumno: David González Herrera
# Carné: 19221022
# Curso: LTT93 2025-2
# Docente: Willer Ferney Montes Granada
# Laboratorio Integrador Periodo 2025-2

"""
Generador interactivo de Carta de Smith completa con regletas inferiores
y anotaciones por hover.
 
- Impedancia normalizada z_N = ZL / Z0
- Coeficiente de reflexión Γ = (ZL - Z0) / (ZL + Z0)
- SWR, Return Loss, Mismatch Loss, etc. tal como se muestran
  en las cartas de Smith completas clásicas.
"""
# Importaciones necesarias 
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt                     # Librería para gráficos  
import numpy as np                                  # Librería para cálculos numéricos   
from matplotlib.backend_bases import RendererBase   # Base para renderizadores de Matplotlib
from matplotlib.patches import Circle               # Para dibujar círculos
from matplotlib.transforms import Affine2D         # Para transformaciones

# Constantes de diseño de la carta
RADIO_CARTA = 1.0
ESCALA_ANGULO_INTERNA = 1.06
ESCALA_ANGULO_EXTERNA = 1.11
ESCALA_LONGITUD_GENERADOR = 1.18
ESCALA_LONGITUD_CARGA = 1.24
ESCALA_PARAMETROS = 1.34

# Función para envolver ángulos en grados
def _envolver_angulo_deg(valor: float) -> float:
    """Normaliza un ángulo a (-180, 180] grados."""
    return ((valor + 180.0) % 360.0) - 180.0

# Función para leer los parámetros de usuario 
def leer_parametros_usuario():
    """
    Lee desde consola los parámetros básicos del problema:
    - Z0: impedancia característica de la línea [ohmios]
    - ZL: impedancia de carga compleja [ohmios], a partir de su parte real (R_L)
      e imaginaria (X_L).

    Retorna
    -------
    Z0 : float
        Impedancia característica de la línea.
    ZL : complex
        Impedancia de carga compleja R_L + j X_L.
    """
    print("=== Generador interactivo de Carta de Smith (completa) ===")
    print("Alumno: David González Herrera (Carné 19221022)")
    print("Curso: LTT93 - Laboratorio Integrador 2025-2")
    while True:
        try:
            Z0 = float(input("Impedancia característica Z0 [Ω] (por ejemplo 50): "))
            if Z0 <= 0:
                print("Z0 debe ser positiva. Intenta de nuevo.")
                continue
            break
        except ValueError:
            print("Entrada no válida para Z0. Intenta de nuevo.")

    while True:
        try:
            R_L = float(input("Parte real de ZL [Ω] (por ejemplo 75): "))
            X_L = float(input("Parte imaginaria de ZL [Ω] (por ejemplo 25 para +j25, -25 para -j25): "))
            ZL = complex(R_L, X_L)
            break
        except ValueError:
            print("Entrada no válida para ZL. Intenta de nuevo.")

    desplazamientos: List[float]
    while True:
        entrada_delta = input(
            "Longitudes normalizadas ℓ (en múltiplos de λ) hacia el generador;"
            " usa valores negativos si deseas moverte hacia la carga (separa con comas, deja vacío para omitir): "
        ).strip()
        if not entrada_delta:
            desplazamientos = []
            break
        try:
            desplazamientos = [float(token.strip()) for token in entrada_delta.split(',') if token.strip()]
            break
        except ValueError:
            print("Entrada no válida para las longitudes. Usa números separados por comas.")

    return Z0, ZL, desplazamientos

# Función para calcular los parámetros asociados 
def calcular_reflexion_y_parametros(Z0, ZL) -> dict[str, Any]:
    """
    Calcula la impedancia normalizada, el coeficiente de reflexión y todos los
    parámetros asociados a las regletas inferiores de la carta de Smith completa.

    Parámetros
    ----------
    Z0 : float
        Impedancia característica de la línea [ohmios].
    ZL : complex
        Impedancia de carga [ohmios].

    Returns
    -------
    resultados : dict
        Diccionario con los parámetros derivados.
    """
    z_norm = ZL / Z0
    gamma_L = (z_norm - 1) / (z_norm + 1)

    gamma_mag = abs(gamma_L)
    gamma_ang_deg = _envolver_angulo_deg(np.degrees(np.angle(gamma_L)))

    if np.isclose(gamma_mag, 1.0):
        SWR = np.inf
    else:
        SWR = (1 + gamma_mag) / (1 - gamma_mag)

    dBS = np.inf if not np.isfinite(SWR) else 20 * np.log10(SWR)
    RL_dB = np.inf if np.isclose(gamma_mag, 0.0) else -20 * np.log10(gamma_mag)

    Gamma_E = gamma_mag
    Gamma_E_dB = -np.inf if np.isclose(gamma_mag, 0.0) else 20 * np.log10(gamma_mag)

    Gamma_P = gamma_mag ** 2
    P_trans = 1 - Gamma_P

    if P_trans <= 0:
        RFL_LOSS_dB = np.inf
    else:
        RFL_LOSS_dB = -10 * np.log10(P_trans)

    ATTEN_dB = np.inf if np.isclose(gamma_mag, 0.0) else -20 * np.log10(gamma_mag)

    if np.isfinite(SWR) and SWR > 0:
        SW_LOSS_COEFF = (1 + SWR**2) / (2 * SWR)
    else:
        SW_LOSS_COEFF = np.inf
    # Coeficiente de transmisión de potencia y tensión 
    T_P = P_trans
    tau_L = 1 + gamma_L
    tau_ang_deg = _envolver_angulo_deg(np.degrees(np.angle(tau_L)))
    T_E_mag = abs(tau_L)
    fase_coef_deg = gamma_ang_deg
    # Coeficiente de transmisión de tensión
    resultados: dict[str, Any] = dict(
        z_norm=z_norm,                   # Impedancia normalizada
        gamma_L=gamma_L,                 # Coeficiente de reflexión en la carga
        gamma_mag=gamma_mag,             # Magnitud del coeficiente de reflexión
        gamma_ang_deg=gamma_ang_deg,     # Ángulo del coeficiente de reflexión en grados
        fase_coef_deg=fase_coef_deg,     # Ángulo de fase equivalente (coef. de fase)
        SWR=SWR,                         # Relación de onda estacionaria
        dBS=dBS,                         # Relación de onda estacionaria en dB
        RL_dB=RL_dB,                     # Pérdida de retorno en dB
        Gamma_E=Gamma_E,                 # Coeficiente de reflexión en la entrada
        Gamma_E_dB=Gamma_E_dB,           # Coeficiente de reflexión en la entrada en dB
        Gamma_P=Gamma_P,                 # Coeficiente de reflexión de potencia
        RFL_LOSS_dB=RFL_LOSS_dB,         # Pérdida por desajuste en dB
        ATTEN_dB=ATTEN_dB,               # Atenuación en dB
        SW_LOSS_COEFF=SW_LOSS_COEFF,     # Coeficiente de pérdida por ROE
        P_trans=P_trans,                 # Potencia transmitida normalizada
        T_P=T_P,                         # Coeficiente de transmisión de potencia
        tau_L=tau_L,                     # Coeficiente de transmisión de tensión
        tau_ang_deg=tau_ang_deg,         # Ángulo del coeficiente de transmisión
        T_E_mag=T_E_mag,                 # Magnitud del coeficiente de transmisión de tensión
        perfiles_linea=[],               # Perfiles a lo largo de la línea (rellenado después)
    )
    return resultados

# Función para imprimir el procedimiento paso a paso 
def imprimir_procedimiento(Z0, ZL, resultados):
    """Imprime en consola un resumen paso a paso con unidades y lo devuelve como texto."""
    lineas: List[str] = []
    lineas.append("")
    lineas.append("=== Procedimiento paso a paso ===")
    lineas.append("1) Datos de entrada:")
    lineas.append(f"   Z0 = {Z0:.2f} Ω")
    lineas.append(f"   ZL = {ZL.real:.2f} + j{ZL.imag:.2f} Ω")

    z_norm = resultados["z_norm"]
    lineas.append("2) Impedancia normalizada z_N = ZL / Z0:")
    lineas.append(f"   z_N = {z_norm.real:.2f} + j{z_norm.imag:.2f} (adimensional)")

    gamma_L = resultados["gamma_L"]
    lineas.append("3) Coeficiente de reflexión en la carga:")
    lineas.append(f"   Γ_L = (z_N - 1) / (z_N + 1) = {gamma_L.real:.2f} + j{gamma_L.imag:.2f}")
    lineas.append(f"   |Γ_L| = {resultados['gamma_mag']:.2f} (adimensional)")
    lineas.append(f"   ∠Γ_L = {resultados['gamma_ang_deg']:.2f} °")
    lineas.append(f"   Ángulo de fase equivalente (coef. de fase) = {resultados['fase_coef_deg']:.2f} °")

    lineas.append("4) Parámetros derivados basados en |Γ_L|:")
    lineas.append(f"   ROE (SWR) = {resultados['SWR']:.2f} (adimensional)")
    lineas.append(f"   ROE en dB (dBS) = {resultados['dBS']:.2f} dB")
    lineas.append(f"   Atenuación equivalente = {resultados['ATTEN_dB']:.2f} dB")
    lineas.append(f"   Coef. pérdida por ROE = {resultados['SW_LOSS_COEFF']:.2f} (adimensional)")
    lineas.append(f"   Pérdida de retorno = {resultados['RL_dB']:.2f} dB")
    lineas.append(f"   Coef. reflexión potencia |Γ|² = {resultados['Gamma_P']:.2f} (adimensional)")
    lineas.append(f"   Pérdida por desajuste = {resultados['RFL_LOSS_dB']:.2f} dB")
    lineas.append(f"   Coef. reflexión |Γ| = {resultados['Gamma_E']:.2f} (adimensional)")

    lineas.append("5) Parámetros de transmisión:")
    lineas.append(f"   Potencia transmitida normalizada = {resultados['P_trans']:.2f} (adimensional)")
    lineas.append(f"   Coef. transmisión de potencia = {resultados['T_P']:.2f} (adimensional)")
    lineas.append(f"   |Coef. transmisión de tensión| = {resultados['T_E_mag']:.2f} (adimensional)")
    lineas.append(f"   ∠(1 + Γ_L) (ángulo del coef. de transmisión) = {resultados['tau_ang_deg']:.2f} °")

    perfiles = resultados.get("perfiles_linea", [])
    if perfiles:
        lineas.append("6) Desplazamientos a lo largo de la línea:")
        for perfil in perfiles:
            sentido = perfil["direccion"]
            lineas.append(
                f"   ℓ = {perfil['longitud']:+.2f} λ ({sentido}), rotación = {perfil['rotacion_deg']:+.2f} °"
            )
            gamma_d = perfil["gamma"]
            tau_d = perfil["tau"]
            lineas.append(
                f"     Γ(ℓ) = {gamma_d.real:.2f} + j{gamma_d.imag:.2f}; ∠Γ(ℓ) = {perfil['gamma_ang_deg']:.2f} °"
            )
            lineas.append(
                f"     τ(ℓ) = {tau_d.real:.2f} + j{tau_d.imag:.2f}; ∠τ(ℓ) = {perfil['tau_ang_deg']:.2f} °"
            )
    lineas.append("=== Fin del resumen ===")

    texto = "\n".join(lineas)
    print(texto)
    return texto

# Función para determinar la orientación del texto circular
def _ajustar_rotacion_tangencial(angulo_deg: float) -> float:
    """Devuelve una rotación tangencial envuelta a [-90, 90] grados."""
    rot = ((angulo_deg + 180.0) % 360.0) - 180.0
    if rot > 90.0:
        rot -= 180.0
    elif rot < -90.0:
        rot += 180.0
    return rot


def _orientacion_texto_circular(x: float, y: float):
    """Determina alineaciones y rotación tangencial para un texto en la circunferencia."""
    if x > 0.1:
        ha = 'left'
    elif x < -0.1:
        ha = 'right'
    else:
        ha = 'center'

    if y > 0.1:
        va = 'bottom'
    elif y < -0.1:
        va = 'top'
    else:
        va = 'center'

    angulo_deg = np.degrees(np.arctan2(y, x))
    rot = _ajustar_rotacion_tangencial(angulo_deg - 90.0)

    return ha, va, rot

# === NUEVA FUNCIÓN: texto “curvado” sobre un anillo ===
def texto_en_arco(
    ax,
    texto: str,
    radio: float,
    angulo_centro_deg: float,
    ancho_grados: float = 180.0,
    **kwargs
):
    """
    Dibuja 'texto' aproximando texto circular a lo largo de un arco
    de circunferencia de radio 'radio'.

    Cada carácter del texto se coloca en un ángulo distinto dentro de
    un abanico de 'ancho_grados' centrado en 'angulo_centro_deg', y se
    rota tangencialmente usando _orientacion_texto_circular.

    Esto NO deforma el texto, pero visualmente se parece a texto sobre
    un anillo, como en las cartas de Smith impresas.
    """
    if not texto:
        return

    n = len(texto)
    if n == 1:
        angulos = [angulo_centro_deg]
    else:
        # Distribuimos los caracteres dentro del arco definido por ancho_grados
        angulos = [
            angulo_centro_deg - ancho_grados / 2.0 + ancho_grados * (i + 0.5) / n
            for i in range(n)
        ]

    for ch, ang_deg in zip(texto, angulos):
        rad = np.radians(ang_deg)
        x = radio * np.cos(rad)
        y = radio * np.sin(rad)
        ha, va, rot = _orientacion_texto_circular(x, y)
        ax.text(
            x,
            y,
            ch,
            ha=ha,
            va=va,
            rotation=rot,
            rotation_mode='anchor',
            **kwargs,
        )

# Función para calcular perfiles a lo largo de la línea 
def _calcular_perfiles_desplazamiento(gamma_L: complex, desplazamientos: List[float]):
    """Genera los perfiles de Γ y 1+Γ al movernos a lo largo de la línea."""
    perfiles = []
    for desplazamiento in desplazamientos:
        rot_rad = -4.0 * np.pi * desplazamiento
        gamma_d = gamma_L * np.exp(1j * rot_rad)
        tau_d = 1 + gamma_d
        perfiles.append(dict(
            longitud=desplazamiento,
            direccion=(
                "hacia el generador" if desplazamiento > 0
                else "hacia la carga" if desplazamiento < 0
                else "en la carga"
            ),
            rotacion_deg=-720.0 * desplazamiento,
            gamma=gamma_d,
            gamma_ang_deg=_envolver_angulo_deg(np.degrees(np.angle(gamma_d))),
            tau=tau_d,
            tau_ang_deg=_envolver_angulo_deg(np.degrees(np.angle(tau_d))),
        ))
    return perfiles

# Función para dibujar la escala de ángulos 
def _dibujar_escala_angulos(ax):
    """Añade las escalas de ángulos de reflexión y transmisión."""
    minor_step = 10
    major_step = 30
    for ang_deg in range(0, 360, minor_step):
        rad = np.radians(ang_deg)
        r_inner = 1.0
        r_outer = 1.03 if ang_deg % major_step else 1.06
        ax.plot(
            [r_inner * np.cos(rad), r_outer * np.cos(rad)],
            [r_inner * np.sin(rad), r_outer * np.sin(rad)],
            color='silver',
            lw=0.4,
            zorder=2,
        )
    # Etiquetas principales 
    for ang_deg in range(0, 360, major_step):
        rad = np.radians(ang_deg)
        cos_v = np.cos(rad)
        sin_v = np.sin(rad)
        if ang_deg % 60 == 0:
            ha, va, rot_tan = _orientacion_texto_circular(cos_v, sin_v)

            etiqueta_ref = ang_deg if ang_deg <= 180 else ang_deg - 360
            ax.text(
                1.12 * cos_v,
                1.12 * sin_v,
                f"{etiqueta_ref:d}°",
                fontsize=6,
                ha=ha,
                va=va,
                rotation=rot_tan,
                rotation_mode='anchor',
                color='black',
            )

            ang_trans = (ang_deg + 180) % 360
            etiqueta_trans = ang_trans if ang_trans <= 180 else ang_trans - 360
            rot_trans = _ajustar_rotacion_tangencial(rot_tan + 180.0)
            ax.text(
                1.24 * cos_v,
                1.24 * sin_v,
                f"{etiqueta_trans:d}°",
                fontsize=6,
                ha=ha,
                va=va,
                rotation=rot_trans,
                rotation_mode='anchor',
                color='dimgray',
            )
    # Etiquetas generales
    ax.text(
        0.5,
        1.06,
        "Ángulo del coeficiente de reflexión (grados)\nÁngulo del coeficiente de transmisión (grados)",
        fontsize=7,
        ha='center',
        va='bottom',
        transform=ax.transAxes
    )

# Función para dibujar la escala de longitudes de onda 
def _dibujar_escala_longitudes(ax):
    """Añade las escala de longitudes de onda hacia generador y carga."""
    generador_vals = np.arange(0.0, 0.55, 0.05)
    for valor in generador_vals:
        ang_deg = valor * 360.0
        rad = np.radians(ang_deg)
        ax.plot(
            [1.18 * np.cos(rad), 1.22 * np.cos(rad)],
            [1.18 * np.sin(rad), 1.22 * np.sin(rad)],
            color='silver',
            lw=0.35,
            zorder=2,
        )

        if np.isclose(valor % 0.1, 0.0, atol=1e-9) or np.isclose(valor, 0.05):
            cos_v = np.cos(rad)
            sin_v = np.sin(rad)
            ha, va, rot = _orientacion_texto_circular(cos_v, sin_v)
            ax.text(
                1.26 * cos_v,
                1.26 * sin_v,
                f"{valor:.2f}",
                fontsize=6,
                ha=ha,
                va=va,
                rotation=rot,
                rotation_mode='anchor',
            )

    carga_vals = np.arange(0.0, 0.55, 0.05)
    for valor in carga_vals:
        ang_deg = 360.0 - valor * 360.0
        rad = np.radians(ang_deg)
        ax.plot(
            [1.24 * np.cos(rad), 1.28 * np.cos(rad)],
            [1.24 * np.sin(rad), 1.28 * np.sin(rad)],
            color='silver',
            lw=0.35,
            zorder=2,
        )

        if np.isclose(valor % 0.1, 0.0, atol=1e-9) or np.isclose(valor, 0.05):
            cos_v = np.cos(rad)
            sin_v = np.sin(rad)
            ha, va, rot = _orientacion_texto_circular(cos_v, sin_v)
            ax.text(
                1.32 * cos_v,
                1.32 * sin_v,
                f"{valor:.2f}",
                fontsize=6,
                ha=ha,
                va=va,
                color='dimgray',
                rotation=rot,
                rotation_mode='anchor',
            )

    ax.text(
        0.5,
        1.015,
        "Longitudes de onda hacia el generador →",
        fontsize=7,
        ha='center',
        va='bottom',
        transform=ax.transAxes
    )
    ax.text(
        0.0,
        -1.40,
        "← Longitudes de onda hacia la carga",
        fontsize=7,
        ha='center',
        va='top'
    )

# Función para dibujar los anillos exteriores 
def _dibujar_anillos_exteriores(ax):
    """Dibuja los anillos externos asociados a las escalas adicionales."""
    radios = [1.06, 1.18, 1.24, 1.32, 1.40]
    for radio in radios:
        circulo = Circle((0.0, 0.0), radio, fill=False, color='lightgray', lw=0.5, linestyle='--', zorder=1)
        ax.add_patch(circulo)

    ax.text(
        -1.48,
        0.0,
        "Reactancia inductiva (+jX/Z0)\nSusceptancia capacitiva (+jB/Y0)",
        fontsize=7,
        rotation=90,
        ha='center',
        va='center',
    )
    ax.text(
        1.48,
        0.0,
        "Reactancia capacitiva (-jX/Z0)\nSusceptancia inductiva (-jB/Y0)",
        fontsize=7,
        rotation=270,
        ha='center',
        va='center',
    )

    ax.text(
        0.5,
        1.03,
        "Ángulo de reflexión",
        fontsize=7,
        ha='center',
        va='bottom',
        color='black',
        transform=ax.transAxes
    )
    ax.text(
        0.0,
        -1.48,
        "Longitudes de onda",
        fontsize=7,
        ha='center',
        va='top',
        color='dimgray'
    )

# Función para dibujar la carta de Smith completa
def dibujar_carta_smith(ax):
    """Dibuja la retícula normalizada de la carta de Smith."""
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.axhline(0, color='lightgray', lw=0.5)
    ax.axvline(0, color='lightgray', lw=0.5)

    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color='black', lw=1)
    ax.plot(0.0, 0.0, marker='o', color='black', markersize=4, zorder=5)
    ax.text(0.04, 0.04, "1 + j0", fontsize=7, ha='left', va='bottom', color='black')

    resistencias = [0.1, 0.2, 0.3, 0.5, 1, 2, 5, 10]
    for r in resistencias:
        centro = r / (1 + r)
        radio = 1 / (1 + r)
        x = centro + radio * np.cos(theta)
        y = radio * np.sin(theta)
        ax.plot(x, y, color='lightgray', lw=0.7)

    reactancias = [0.1, 0.2, 0.3, 0.5, 1, 2, 5, 10]
    for x_val in reactancias:
        for signo in (+1, -1):
            x_norm = signo * x_val
            centro = (1.0, 1.0 / x_norm)
            radio = 1.0 / abs(x_norm)
            phi = np.linspace(-np.pi, np.pi, 400)
            x = centro[0] + radio * np.cos(phi)
            y = centro[1] + radio * np.sin(phi)
            mask = x**2 + y**2 <= 1.0001
            ax.plot(x[mask], y[mask], color='lightgray', lw=0.7)

    ax.set_xlabel(r'$\Re\{\Gamma\}$', labelpad=16)
    ax.set_ylabel(r'$\Im\{\Gamma\}$', labelpad=16)
    # ax.set_title("Carta de Smith normalizada", pad=28)
    ax.set_xticks([])
    ax.set_yticks([])
    # Marcadores en el eje real para identificar la escala puramente resistiva/admitiva
    valores_r = [0.0] + resistencias + [np.inf]
    for r in valores_r:
        if np.isinf(r):
            x_pos = 1.0
            etiqueta_r = "∞"
            etiqueta_g = "0"
        else:
            x_pos = (r - 1.0) / (r + 1.0)
            etiqueta_r = f"{r:g}"
            if r == 0.0:
                etiqueta_g = "∞"
            else:
                etiqueta_g = f"{(1.0 / r):g}"
        if -1.05 <= x_pos <= 1.05:
            ax.plot([x_pos, x_pos], [0.0, -0.025], color='gray', lw=0.45, zorder=4)
            ax.text(x_pos, -0.06, etiqueta_r, ha='center', va='top', fontsize=6, color='black')
            ax.text(x_pos, 0.06, etiqueta_g, ha='center', va='bottom', fontsize=6, color='dimgray')
    #ax.text(0.5, -0.10, "Resistance Component (R/Z0)", transform=ax.transAxes, ha='center', va='top', fontsize=7)
    #ax.text(0.5, 1.08, "Conductance Component (G/Y0)", transform=ax.transAxes, ha='center', va='bottom', fontsize=7, color='dimgray')
    # Dibujar elementos adicionales 
    _dibujar_anillos_exteriores(ax)
    _dibujar_escala_angulos(ax)
    _dibujar_escala_longitudes(ax)

# Función para calcular gamma desde ROE 
def _calcular_gamma_desde_S(S):
    return (S - 1.0) / (S + 1.0)

# Función para calcular gamma desde pérdida de retorno en dB
def _calcular_gamma_desde_RL(RL_dB):
    return 10.0 ** (-RL_dB / 20.0)

# Función para calcular gamma desde coeficiente de reflexión potencia 
def _calcular_gamma_desde_GammaP(Gamma_P):
    return np.sqrt(Gamma_P)

# Función para calcular gamma desde pérdida por desajuste en dB
def _calcular_gamma_desde_RFL_LOSS(L_dB):
    return np.sqrt(1.0 - 10.0 ** (-L_dB / 10.0))

# Función para calcular gamma desde atenuación en dB 
def _calcular_gamma_desde_ATTEN(A_dB):
    return 10.0 ** (-A_dB / 20.0)

# Función para calcular gamma desde coeficiente de pérdida por ROE 
def _calcular_gamma_desde_SW_LOSS(F):
    F = np.asarray(F)
    # Clamp values to keep the square root argument non-negative
    F_clamped = np.maximum(F, 1.0)
    S = F_clamped + np.sqrt(F_clamped**2 - 1.0)
    return _calcular_gamma_desde_S(S)

# Función para dibujar las regletas inferiores 
def dibujar_regletas(ax, resultados):
    """Dibuja las regletas inferiores con leyendas horizontales a ambos lados."""
    gamma_mag = resultados["gamma_mag"]

    regletas = [
        dict(
            etiqueta="ROE (SWR)",
            valores=[1.1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 4, 5, 10, 20, 40, 100],
            map_gamma=_calcular_gamma_desde_S,
            clave_resultado="SWR",
            formato=lambda v: f"{v:.2f}"
        ),
        dict(
            etiqueta="Pérdida de retorno (dB)",
            valores=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 20, 30],
            map_gamma=_calcular_gamma_desde_RL,
            clave_resultado="RL_dB",
            formato=lambda v: "∞" if not np.isfinite(v) else f"{v:.2f}"
        ),
        dict(
            etiqueta="Coef. reflexión potencia |Γ|²",
            valores=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            map_gamma=_calcular_gamma_desde_GammaP,
            clave_resultado="Gamma_P",
            formato=lambda v: f"{v:.2f}"
        ),
        dict(
            etiqueta="Coef. reflexión |Γ|",
            valores=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            map_gamma=lambda g: g,
            clave_resultado="Gamma_E",
            formato=lambda v: f"{v:.2f}"
        ),
        dict(
            etiqueta="Atenuación (dB)",
            valores=[1, 2, 3, 4, 5, 7, 10, 15],
            map_gamma=_calcular_gamma_desde_ATTEN,
            clave_resultado="ATTEN_dB",
            formato=lambda v: "∞" if not np.isfinite(v) else f"{v:.2f}"
        ),
        dict(
            etiqueta="Coef. pérdida por ROE",
            valores=[1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2, 3, 4, 5, 10, 20],
            map_gamma=_calcular_gamma_desde_SW_LOSS,
            clave_resultado="SW_LOSS_COEFF",
            formato=lambda v: "∞" if not np.isfinite(v) else f"{v:.2f}"
        ),
        dict(
            etiqueta="Pérdida por desajuste (dB)",
            valores=[0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3, 4, 5, 6, 10, 15],
            map_gamma=_calcular_gamma_desde_RFL_LOSS,
            clave_resultado="RFL_LOSS_dB",
            formato=lambda v: "∞" if not np.isfinite(v) else f"{v:.2f}"
        ),
        dict(
            etiqueta="ROE en dB (dBS)",
            valores=[1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40],
            map_gamma=lambda d: _calcular_gamma_desde_S(10.0 ** (d / 20.0)),
            clave_resultado="dBS",
            formato=lambda v: "∞" if not np.isfinite(v) else f"{v:.2f}"
        ),
    ]
    # Determinar número de regletas
    num_reglas = len(regletas)
    # Configurar ejes  
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.7, num_reglas + 0.7)
    ax.axis('off')
    # Dibujar las regleta
    for idx, reg in enumerate(regletas):
        y = num_reglas - idx - 0.5

        ax.hlines(y, 0, 1.0, colors='black', lw=0.6)

        for valor in reg["valores"]:
            x = reg["map_gamma"](valor)
            if np.isfinite(x) and 0 <= x <= 1:
                ax.vlines(x, y - 0.08, y + 0.08, colors='black', lw=0.5)
                ax.text(x, y + 0.1, f"{valor:g}", ha='center', va='bottom', fontsize=6)

        if np.isfinite(gamma_mag):
            ax.vlines(gamma_mag, y - 0.15, y + 0.15, colors='red', lw=1.2)

        valor_actual = resultados.get(reg["clave_resultado"])
        if valor_actual is not None and np.isfinite(valor_actual):
            x_val = reg["map_gamma"](valor_actual)
            if np.isfinite(x_val) and 0 <= x_val <= 1:
                ax.plot(x_val, y, marker='o', color='red', markersize=4, zorder=5)
                ax.text(x_val, y - 0.12, reg["formato"](valor_actual), ha='center', va='top', fontsize=6, color='red')
        elif valor_actual is not None:
            texto_valor = reg["formato"](valor_actual)
            if idx < num_reglas / 2:
                ax.text(-0.25, y, texto_valor, ha='left', va='center', fontsize=6, color='red')
            else:
                ax.text(1.25, y, texto_valor, ha='right', va='center', fontsize=6, color='red')

        if idx < num_reglas / 2:
            x_text = -0.2
            ax.text(x_text, y, reg["etiqueta"], ha='right', va='center', fontsize=7)
            ax.plot([x_text + 0.02, 0.0], [y, y], color='gray', lw=0.5)
        else:
            x_text = 1.2
            ax.text(x_text, y, reg["etiqueta"], ha='left', va='center', fontsize=7)
            ax.plot([1.0, x_text - 0.02], [y, y], color='gray', lw=0.5)

    # ==== BLOQUE NUEVO: título + flechas laterales + texto centrado ====

    # Título general (tipo "RADIALLY SCALED PARAMETERS")
    ax.text(
        0.5,
        1.10,
        "PARÁMETROS ESCALADOS RADIALMENTE",
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='bold',
        transform=ax.transAxes,
    )

    # Texto "HACIA LA CARGA" a la izquierda
    ax.text(
        0.00,
        1.00,
        "HACIA LA CARGA",
        ha='left',
        va='bottom',
        fontsize=8,
        transform=ax.transAxes,
    )

    # Flecha gruesa hacia la derecha (carga)
    ax.annotate(
        "",
        xy=(0.30, 1.025),      # punta de la flecha (derecha)
        xytext=(0.06, 1.025),  # origen (izquierda)
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops=dict(
            arrowstyle="->",
            lw=1.4,
        ),
    )

    # Texto "HACIA EL GENERADOR" a la derecha
    ax.text(
        0.95, # 
        1.00, # 
        "HACIA EL GENERADOR",
        ha='right',
        va='bottom',
        fontsize=8,
        transform=ax.transAxes,
    )

    # Flecha gruesa hacia la izquierda (generador)
    ax.annotate(
        "",
        xy=(0.94, 1.025),      # punta (izquierda)
        xytext=(0.70, 1.025),  # origen (derecha)
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops=dict(
            arrowstyle="<-",
            lw=1.4,
        ),
    )

    # Texto centrado bajo las flechas, como referencia adicional
    ax.text(
        0.5,
        0.98,
        "Hacia la carga →   ← Hacia el generador",
        ha='center',
        va='top',
        fontsize=8,
        transform=ax.transAxes,
    )

# Función principal para crear la gráfica completa con hover y regletas
def crear_grafica_completa(Z0, ZL, desplazamientos: Optional[List[float]] = None):
    """Genera la figura completa con hover y regletas."""
    if desplazamientos is None:
        desplazamientos = []
    resultados = calcular_reflexion_y_parametros(Z0, ZL)
    perfiles_linea = _calcular_perfiles_desplazamiento(resultados["gamma_L"], desplazamientos)
    resultados["perfiles_linea"] = perfiles_linea
    procedimiento_texto = imprimir_procedimiento(Z0, ZL, resultados)

    gamma_L = resultados["gamma_L"]
    gamma_mag = resultados["gamma_mag"]

    fig = plt.figure(figsize=(9, 11), constrained_layout=True)
    fig.suptitle(
        "Laboratorio Integrador 2025-2 - LTT93",
        fontsize=10,
    )
    gestor_ventana = getattr(fig.canvas, "manager", None)
    if gestor_ventana is not None:
        try:
            gestor_ventana.set_window_title(
                "Carta de Smith - David González Herrera (19221022)"
            )
        except Exception:
            pass
    gs = fig.add_gridspec(2, 1, height_ratios=[5.0, 1.6])

    ax_smith = fig.add_subplot(gs[0, 0])
    ax_regletas = fig.add_subplot(gs[1, 0])

    dibujar_carta_smith(ax_smith)
    ax_smith.plot(gamma_L.real, gamma_L.imag, 'ro', label=r'$\Gamma_L$')

    theta = np.linspace(0, 2*np.pi, 400)
    ax_smith.plot(
        gamma_mag * np.cos(theta),
        gamma_mag * np.sin(theta),
        'r--',
        lw=1.0,
        label=r'|$\Gamma$| constante'
    )
    ax_smith.legend(loc='upper left', bbox_to_anchor=(1.15, 1.15), fontsize=8, frameon=False)

    dibujar_regletas(ax_regletas, resultados)

    texto_leyenda = (
        "Leyenda de las regletas (anillos inferiores):\n"
        "• ROE (Relación de onda estacionaria): con S = (1 + |Γ|)/(1 - |Γ|).\n"
        "• ROE en dB (dBS): la misma ROE expresada como dBS = 20·log10(S).\n"
        "• Atenuación (dB): atenuador equivalente frente a reflexión total; |Γ| ≈ 10^(-A/20).\n"
        "• Coef. pérdida por ROE: factor clásico F ≈ (1 + S²)/(2·S).\n"
        "• Pérdida de retorno (dB): RL = -20·log10(|Γ|), potencia reflejada hacia el generador.\n"
        "• Coef. reflexión potencia |Γ|²: razón de potencias reflejada/incidente.\n"
        "• Pérdida por desajuste (dB): -10·log10(1 - |Γ|²), potencia no transferida a la carga.\n"
        "• Coef. reflexión |Γ|: módulo del coeficiente de reflexión de tensión o corriente.\n"
        "• Círculo de impedancia: borde |Γ| = 1 que delimita la carta y define el mapa de impedancias normalizadas."
    )

    fig_procedimiento = plt.figure(figsize=(7, 8), constrained_layout=True)
    fig_procedimiento.suptitle("Procedimiento paso a paso", fontsize=11)
    ax_proc = fig_procedimiento.add_subplot(1, 1, 1)
    ax_proc.axis('off')
    ax_proc.text(
        0.01,
        0.98,
        procedimiento_texto + "\n\n" + texto_leyenda,
        ha='left',
        va='top',
        fontsize=9,
        wrap=True,
    )

    anot = ax_smith.annotate(
        "", xy=(0, 0), xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w", alpha=0.9),
        arrowprops=dict(arrowstyle="->")
    )
    anot.set_visible(False)

    fig.canvas.draw()
    # Función para obtener el renderer actual del canvas 
    def obtener_renderer() -> Optional[RendererBase]:
        renderer = getattr(fig.canvas, "get_renderer", None)
        if callable(renderer):
            renderer = renderer()
        if renderer is None:
            renderer = getattr(fig.canvas, "renderer", None)
        if renderer is None:
            fig.canvas.draw()
            renderer = getattr(fig.canvas, "renderer", None)
        if isinstance(renderer, RendererBase) or renderer is None:
            return renderer
        return None
    # Ajusta la posición de la anotación para que no se salga de los bordes
    def ajustar_a_bordes(eje_destino):
        renderer = obtener_renderer()
        if renderer is None:
            return

        bbox_ax = eje_destino.get_window_extent(renderer)
        bbox_anot = anot.get_window_extent(renderer)

        margen = 4
        delta_x = delta_y = 0.0

        if bbox_anot.x0 < bbox_ax.x0 + margen:
            delta_x = (bbox_ax.x0 + margen) - bbox_anot.x0
        elif bbox_anot.x1 > bbox_ax.x1 - margen:
            delta_x = (bbox_ax.x1 - margen) - bbox_anot.x1

        if bbox_anot.y0 < bbox_ax.y0 + margen:
            delta_y = (bbox_ax.y0 + margen) - bbox_anot.y0
        elif bbox_anot.y1 > bbox_ax.y1 - margen:
            delta_y = (bbox_ax.y1 - margen) - bbox_anot.y1

        if delta_x or delta_y:
            px, py = anot.get_position()
            escala = 72.0 / fig.dpi
            anot.set_position((px + delta_x * escala, py + delta_y * escala))
    
    # Configuración de la posición de la anotación según el eje
    def configurar_anotacion(eje_destino):
        if eje_destino is ax_smith:
            dx = 25 if gamma_L.real <= 0 else -120
            dy = 30 if gamma_L.imag <= 0 else -90
            ha = 'left' if dx >= 0 else 'right'
            va = 'bottom' if dy >= 0 else 'top'
        else:
            dx = 25
            dy = -60
            ha = 'left'
            va = 'top'

        anot.set_position((dx, dy))
        anot.set_horizontalalignment(ha)
        anot.set_verticalalignment(va)
    # Formato base del texto de la anotación 
    def formato_valores_base():
        z = resultados["z_norm"]
        return (
            f"Z0 = {Z0:.2f} Ω\n"
            f"ZL = {ZL.real:.2f} + j{ZL.imag:.2f} Ω\n"
            f"z_N = {z.real:.2f} + j{z.imag:.2f} (Ad)\n"
            f"Γ_L = {gamma_L.real:.2f} + j{gamma_L.imag:.2f}\n"
            f"|Γ_L| = {resultados['gamma_mag']:.2f} (Ad)\n"
            f"∠Γ_L = {resultados['gamma_ang_deg']:.2f}°\n"
            f"Coef. de fase = {resultados['fase_coef_deg']:.2f}°\n"
            f"ROE (SWR) = {resultados['SWR']:.2f} (Ad)\n"
            f"ROE en dB (dBS) = {resultados['dBS']:.2f} dB\n"
            f"Pérdida de retorno = {resultados['RL_dB']:.2f} dB\n"
            f"|Γ| (coef. reflexión de tensión) = {resultados['Gamma_E']:.2f} (Ad)\n"
            f"|Γ| en dB = {resultados['Gamma_E_dB']:.2f} dB\n"
            f"Coef. reflexión potencia |Γ|² = {resultados['Gamma_P']:.2f} (Ad)\n"
            f"Potencia transmitida (1−|Γ|²) = {resultados['P_trans']:.2f} (Ad)\n"
            f"Pérdida por desajuste = {resultados['RFL_LOSS_dB']:.2f} dB\n"
            f"Atenuación equivalente = {resultados['ATTEN_dB']:.2f} dB\n"
            f"Coef. pérdida por ROE = {resultados['SW_LOSS_COEFF']:.2f} (Ad)\n"
            f"Coef. transmisión de potencia = {resultados['T_P']:.2f} (Ad)\n"
            f"|Coef. transmisión de tensión| = {resultados['T_E_mag']:.2f} (Ad)\n"
            f"∠(1 + Γ_L) = {resultados['tau_ang_deg']:.2f}°"
        )
    # Formato completo con perfiles de línea
    def formato_valores_completo():
        base = formato_valores_base()
        perfiles = resultados.get("perfiles_linea", [])
        if not perfiles:
            return base
        # Añadir perfiles de línea
        lineas = [base, "", "Desplazamientos evaluados:"]
        max_mostrar = 3
        for perfil in perfiles[:max_mostrar]:
            simbolo = "→" if perfil["longitud"] > 0 else "←" if perfil["longitud"] < 0 else "•"
            lineas.append(
                f"{simbolo} ℓ = {perfil['longitud']:+.2f} λ ({perfil['direccion']})"
            )
            lineas.append(
                f"   ∠Γ(ℓ) = {perfil['gamma_ang_deg']:.2f}°, ∠τ(ℓ) = {perfil['tau_ang_deg']:.2f}°"
            )
        if len(perfiles) > max_mostrar:
            lineas.append("   …")
        return "\n".join(lineas)
    # Selección del formato de valores según si hay perfiles o no
    def formato_valores():
        return formato_valores_completo()
    # Evento de movimiento del ratón
    def on_move(event):
        if event.inaxes not in (ax_smith, ax_regletas):
            if anot.get_visible():
                anot.set_visible(False)
                fig.canvas.draw_idle()
            return

        mostrar = False
        if event.inaxes is ax_smith and event.xdata is not None and event.ydata is not None:
            dist = np.hypot(event.xdata - gamma_L.real, event.ydata - gamma_L.imag)
            if dist < 0.06:
                anot.xy = (gamma_L.real, gamma_L.imag)
                mostrar = True
        elif event.inaxes is ax_regletas and event.xdata is not None and event.ydata is not None:
            if abs(event.xdata - gamma_mag) < 0.02:
                anot.xy = (gamma_mag, ax_regletas.get_ylim()[1])
                mostrar = True

        if mostrar:
            configurar_anotacion(event.inaxes)
            anot.set_text(formato_valores())
            anot.set_visible(True)
            ajustar_a_bordes(event.inaxes)
        else:
            if anot.get_visible():
                anot.set_visible(False)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)

    plt.show()

# Función principal para ejecutar el script
def main():
    Z0, ZL, desplazamientos = leer_parametros_usuario()
    crear_grafica_completa(Z0, ZL, desplazamientos)


if __name__ == "__main__":
    main()
