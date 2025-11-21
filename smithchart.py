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
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import RendererBase
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection  # Optimización: Para renderizado rápido de líneas
from matplotlib.backends.backend_pdf import PdfPages  # NUEVO: para PDF multipágina

# Constantes de diseño de la carta
RADIO_CARTA = 1.0
ESCALA_ANGULO_INTERNA = 1.06
ESCALA_ANGULO_EXTERNA = 1.24
ESCALA_LONGITUD_GENERADOR = 1.18
ESCALA_LONGITUD_CARGA = 1.24
ESCALA_PARAMETROS = 1.34
VELOCIDAD_LUZ = 299_792_458.0  # m/s

# Función para envolver ángulos en grados
def _envolver_angulo_deg(valor: float) -> float:
    """Normaliza un ángulo a (-180, 180] grados."""
    return ((valor + 180.0) % 360.0) - 180.0

# Expresión regular para analizar números complejos en forma rectangular
_COMPLEJO_RECT_RE = re.compile(
    r"^\s*(?P<real>[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)?"
    r"(?:(?P<imag_sign>[+-]?)[ji](?P<imag>\d*(?:\.\d+)?(?:e[+-]?\d+)?))?\s*$",
    re.IGNORECASE,
)

# Funciones para analizar y formatear números complejos en forma rectangular
def _parse_complejo_rectangular(texto: str) -> complex:
    """Parses strings like '50', '50+j25', '-j10' into complex numbers."""
    # Optimización: Limpieza previa y soporte para 'i'
    s = texto.strip().replace(',', '.').replace(' ', '').lower()
    if not s:
        raise ValueError("La cadena está vacía.")
    
    match = _COMPLEJO_RECT_RE.match(s)
    if not match:
        # Intento secundario usando el parser nativo de Python si el regex falla
        try:
            return complex(s.replace('i', 'j'))
        except ValueError:
            raise ValueError("Formato rectangular inválido.")

    # Extraer partes
    real_str = match.group('real')
    imag_sign = match.group('imag_sign')
    imag_str = match.group('imag')
    
    # Determinar si hay parte imaginaria
    tiene_imaginaria = ('j' in s or 'i' in s) or (imag_sign is not None and imag_str is not None)
    
    if real_str is None and not tiene_imaginaria:
        raise ValueError("Formato rectangular inválido.")
    
    # Convertir a float
    real = float(real_str) if real_str else 0.0
    
    # Parte imaginaria
    if tiene_imaginaria:
        signo = imag_sign if imag_sign else '+'
        if imag_str is None or imag_str == '':
            imag_val = 1.0
        else:
            imag_val = float(imag_str)
        imag = imag_val if signo != '-' else -imag_val
    else:
        imag = 0.0

    return complex(real, imag)


_FRECUENCIA_SUFIJOS = (
    ("ghz", 1e9),
    ("g", 1e9),
    ("mhz", 1e6),
    ("m", 1e6),
    ("khz", 1e3),
    ("k", 1e3),
    ("hz", 1.0),
)


def _parse_frecuencia(texto: str) -> float:
    """Convierte una cadena opcionalmente con sufijos (Hz, kHz, MHz, GHz) a Hz."""
    s = texto.strip().lower().replace(' ', '')
    s = s.replace(',', '.')
    if not s:
        raise ValueError("La cadena está vacía.")

    for sufijo, factor in _FRECUENCIA_SUFIJOS:
        if s.endswith(sufijo) and len(s) > len(sufijo):
            base = s[:-len(sufijo)]
            return float(base) * factor

    return float(s)

def _parse_porcentaje(texto: str) -> float:
    """
    Convierte cadenas como '75', '75%', '0.75' en un número entre 0 y 1.

    - '75' o '75%' -> 0.75
    - '0.75'       -> 0.75

    Se usa para el factor de velocidad (VNP) de la LT.
    """
    s = texto.strip().lower().replace(' ', '').replace(',', '.')
    if not s:
        raise ValueError("Cadena vacía para porcentaje.")
    if s.endswith('%'):
        base = s[:-1]
        return float(base) / 100.0
    val = float(s)
    if val > 1.0:      # se asume que viene en %
        val = val / 100.0
    return val


def _formatear_frecuencia_hz(valor_hz: float) -> str:
    """Devuelve la frecuencia con el sufijo más legible (Hz, kHz, MHz o GHz)."""
    magnitudes = (
        (1e9, "GHz"),
        (1e6, "MHz"),
        (1e3, "kHz"),
    )
    for factor, sufijo in magnitudes:
        if valor_hz >= factor:
            return f"{valor_hz / factor:.3f} {sufijo}"
    return f"{valor_hz:.3f} Hz"


def _formatear_longitud_m(valor_m: float, *, signo: bool = True) -> str:
    """Formatea longitudes en metros con resolución adaptable."""
    abs_val = abs(valor_m)
    if abs_val >= 10:
        fmt = "{:+.3f}" if signo else "{:.3f}"
    elif abs_val >= 1:
        fmt = "{:+.4f}" if signo else "{:.4f}"
    elif abs_val >= 0.01:
        fmt = "{:+.5f}" if signo else "{:.5f}"
    elif abs_val == 0:
        fmt = "{:+.4f}" if signo else "{:.4f}"
    else:
        fmt = "{:+.2e}" if signo else "{:.2e}"
    return f"{fmt.format(valor_m)} m"

# Función para formatear números complejos en forma rectangular
def _formatear_complejo_rectangular(valor: complex, decimales: int = 2) -> str:
    """Devuelve una representación 'a ± j b' con los decimales deseados."""
    valor_c = complex(valor)
    if not np.isfinite(valor_c):
        return "∞"

    real = valor_c.real
    imag = valor_c.imag
    tol = 10 ** (-(decimales + 2))
    if abs(real) < tol:
        real = 0.0
    if abs(imag) < tol:
        imag = 0.0

    signo = '+' if imag >= 0 else '-'
    return f"{real:.{decimales}f} {signo} j{abs(imag):.{decimales}f}"

# Función para leer los parámetros de usuario 
def leer_parametros_usuario():
    """
    Solicita por consola todos los datos necesarios:

    - Z0 (siempre).
    - Tipo de carga / escenario:
        1) Carga general ZL (compleja).
        2) Circuito abierto (ZL = ∞).
        3) Cortocircuito (ZL = 0).
        4) Línea con carga resistiva pura conocida por ROE (se calculan ZL1 y ZL2).

    - Opcionalmente:
        * Frecuencia (con sufijos Hz/kHz/MHz/GHz).
        * Factor de velocidad VNP (%) para la línea.
        * Longitud física total de la línea.
        * Distancias físicas donde se quiere evaluar Γ y ZL′.
        * Longitudes eléctricas normalizadas (en múltiplos de λ).

    Además prepara un diccionario `parametros_linea` con:
        - frecuencia_hz
        - lambda_m
        - vp_m_s
        - vnp (tanto por uno)
        - tipo_carga
        - roe_resistiva (si aplica)
        - cargas_resistivas_equivalentes (si aplica)
    y la lista `perfiles_config` con las longitudes eléctricas solicitadas.
    """
    print("=== Generador interactivo de Carta de Smith (completa) ===")
    print("Alumno: David González Herrera (Carné 19221022)")
    print("Curso: LTT93 - Laboratorio Integrador 2025-2\n")
    print("Ingresa cada valor solicitado en formato rectangular (ej. 50, 50+j25, 75-j10)")
    print("o en las unidades indicadas para frecuencia, longitudes y ROE.\n")

    # --- Z0 (siempre obligatorio) ---
    while True:
        entrada_Z0 = input("Impedancia característica Z0 [Ω] (ej. 50, 75, 50+j25): ")
        try:
            Z0 = _parse_complejo_rectangular(entrada_Z0)
        except ValueError:
            print("Entrada no válida para Z0. Usa el formato a+jb.")
            continue

        if np.isclose(abs(Z0), 0.0, atol=1e-12):
            print("Z0 no puede ser cero. Intenta de nuevo.")
            continue
        break

    # --- Tipo de carga / escenario ---
    print("\nSelecciona el tipo de carga / escenario:")
    print("  1) Carga general ZL (compleja).")
    print("  2) Circuito abierto (CTO. ABIERTO, ZL = ∞).")
    print("  3) Cortocircuito (CTO, ZL = 0).")
    print("  4) Línea con carga resistiva pura conocida por ROE (se calculan ZL1 y ZL2).")

    tipo_carga = "general"
    roe_resistiva = None
    cargas_resistivas_equivalentes = None

    while True:
        modo = input("Opción [1/2/3/4]: ").strip()
        if modo not in ("1", "2", "3", "4"):
            print("Opción no válida. Elige 1, 2, 3 o 4.")
            continue
        modo = int(modo)
        break

    # --- ZL según el modo escogido ---
    if modo == 1:
        # Carga general ZL compleja
        while True:
            entrada_ZL = input("Impedancia de carga ZL [Ω] (ej. 200-j50): ")
            try:
                ZL = _parse_complejo_rectangular(entrada_ZL)
                break
            except ValueError:
                print("Entrada no válida para ZL. Usa formato a+jb.")
        tipo_carga = "general"

    elif modo == 2:
        # Circuito abierto
        ZL = complex(np.inf)  # solo para guardar, el cálculo usará tipo_carga
        tipo_carga = "abierto"
        print(">> Se trabajará con circuito abierto: ZL = ∞, Γ_L = +1 ∠0°.")

    elif modo == 3:
        # Cortocircuito
        ZL = 0.0 + 0.0j
        tipo_carga = "corto"
        print(">> Se trabajará con cortocircuito: ZL = 0, Γ_L = -1 ∠180°.")

    else:
        # ROE conocida, carga resistiva pura (dos posibles ZL).
        tipo_carga = "roe_resistiva"
        while True:
            entrada_roe = input("Valor de ROE (SWR) (>1, ej. 1.5, 2.65): ").strip()
            try:
                roe_resistiva = float(entrada_roe.replace(',', '.'))
                if roe_resistiva <= 1.0:
                    print("La ROE debe ser mayor que 1. Intenta de nuevo.")
                    continue
                break
            except ValueError:
                print("Entrada no válida para ROE.")
        # Para carga resistiva pura hay dos soluciones clásicas:
        #   ZL1 = Z0 * S   (ZL1 > Z0, fase en fase)
        #   ZL2 = Z0 / S   (ZL2 < Z0, fase en contrafase)
        ZL1 = Z0.real * roe_resistiva  # asumimos Z0 real para este caso
        ZL2 = Z0.real / roe_resistiva
        cargas_resistivas_equivalentes = (ZL1, ZL2)
        print(f">> Carga resistiva pura con ROE = {roe_resistiva:.3f}:")
        print(f"   ZL1 (en fase, ZL > Z0)  ≈ {ZL1:.3f} Ω")
        print(f"   ZL2 (contra fase, ZL < Z0) ≈ {ZL2:.3f} Ω")
        # Para dibujar la carta tomamos por defecto la solución en fase (ZL1),
        # de manera que Γ_L sea positivo real.
        ZL = complex(ZL1, 0.0)

    # --- Longitudes eléctricas normalizadas directamente en λ ---
    desplazamientos_norm: List[float]
    while True:
        entrada_delta = input(
            "\nLongitudes normalizadas ℓ (en múltiplos de λ) hacia el generador;\n"
            "usa valores negativos si deseas moverte hacia la carga\n"
            "(separa con comas, deja vacío para omitir): "
        ).strip()
        if not entrada_delta:
            desplazamientos_norm = []
            break
        try:
            desplazamientos_norm = [
                float(token.strip())
                for token in entrada_delta.split(',')
                if token.strip()
            ]
            break
        except ValueError:
            print("Entrada no válida. Usa números separados por comas (ej. 0.25, 0.5, -0.1).")

    perfiles_config: List[Dict[str, Any]] = [
        dict(longitud_norm=valor, fuente="ℓ normalizada")
        for valor in desplazamientos_norm
    ]

    # --- Frecuencia y factor de velocidad VNP ---
    frecuencia_hz: Optional[float] = None
    lambda_m: Optional[float] = None
    vnp: Optional[float] = None
    vp_m_s: Optional[float] = None

    while True:
        entrada_f = input(
            "\nFrecuencia de operación f [Hz] (acepta sufijos k/M/G, deja vacío para omitir): "
        ).strip()
        if not entrada_f:
            break
        try:
            frecuencia_hz = _parse_frecuencia(entrada_f)
            if frecuencia_hz <= 0:
                print("La frecuencia debe ser positiva. Intenta de nuevo.")
                frecuencia_hz = None
                continue
        except ValueError:
            print("Entrada no válida para la frecuencia. Ejemplos: 915e6, 2.45GHz, 60MHz.")
            continue

        # Factor de velocidad VNP (opcional pero MUY útil)
        entrada_vnp = input(
            "Factor de velocidad de la LT VNP [%] (ej. 75, 80; ENTER para asumir 100% c): "
        ).strip()
        if entrada_vnp:
            try:
                vnp = _parse_porcentaje(entrada_vnp)
                if not (0 < vnp <= 1.0):
                    print("VNP debe estar entre 0 y 100%. Se asumirá 100%.")
                    vnp = 1.0
            except ValueError:
                print("No se pudo interpretar VNP, se asume 100%.")
                vnp = 1.0
        else:
            vnp = 1.0

        vp_m_s = vnp * VELOCIDAD_LUZ
        lambda_m = vp_m_s / frecuencia_hz
        break

    # --- Longitud física total de la línea ---
    longitud_total_m: Optional[float] = None
    while True:
        entrada_L = input(
            "\nLongitud física total de la línea L [m] (deja vacío para omitir): "
        ).strip()
        if not entrada_L:
            break
        try:
            longitud_total_m = float(entrada_L.replace(',', '.'))
            if longitud_total_m <= 0:
                print("La longitud debe ser positiva. Intenta de nuevo.")
                longitud_total_m = None
                continue
            break
        except ValueError:
            print("Entrada no válida. Usa números positivos (ej. 2.5 o 0.75).")

    # --- Distancias físicas específicas donde quieres evaluar Γ y ZL′ ---
    distancias_fisicas: List[float] = []
    while True:
        entrada_dist = input(
            "Distancias físicas d [m] hacia el generador (usa signo para indicar dirección;\n"
            "separa con comas; deja vacío para omitir): "
        ).strip()
        if not entrada_dist:
            break
        if lambda_m is None:
            print("Primero proporciona la frecuencia (y VNP) para convertir distancias físicas en longitudes eléctricas.")
            continue
        try:
            distancias_fisicas = [
                float(token.replace(',', '.'))
                for token in entrada_dist.split(',')
                if token.strip()
            ]
            break
        except ValueError:
            print("Entrada no válida. Usa números separados por comas.")

    # --- Construir perfiles_config usando λ calculada ---
    if lambda_m is not None:
        for entrada in perfiles_config:
            entrada.setdefault("distancia_m", entrada["longitud_norm"] * lambda_m)
        for distancia in distancias_fisicas:
            perfiles_config.append(
                dict(
                    longitud_norm=distancia / lambda_m,
                    distancia_m=distancia,
                    fuente="distancia física",
                )
            )
    elif distancias_fisicas:
        print("Se ignorarán las distancias físicas porque no se especificó la frecuencia/VNP.")
        distancias_fisicas = []

    # --- Empaquetar parámetros de línea ---
    parametros_linea: Dict[str, Any] = {
        "tipo_carga": tipo_carga,
    }
    if frecuencia_hz is not None:
        parametros_linea["frecuencia_hz"] = frecuencia_hz
    if lambda_m is not None:
        parametros_linea["lambda_m"] = lambda_m
    if vp_m_s is not None:
        parametros_linea["vp_m_s"] = vp_m_s
    if vnp is not None:
        parametros_linea["vnp"] = vnp
    if longitud_total_m is not None:
        parametros_linea["longitud_total_m"] = longitud_total_m
    if distancias_fisicas:
        parametros_linea["distancias_fisicas"] = distancias_fisicas
    if roe_resistiva is not None:
        parametros_linea["roe_resistiva"] = roe_resistiva
    if cargas_resistivas_equivalentes is not None:
        parametros_linea["cargas_resistivas_equivalentes"] = cargas_resistivas_equivalentes

    return Z0, ZL, perfiles_config, parametros_linea



# Función para calcular los parámetros asociados 
def calcular_reflexion_y_parametros(
    Z0: complex,
    ZL: complex,
    datos_linea: Optional[Dict[str, Any]] = None
) -> dict[str, Any]:
    """
    Calcula todos los parámetros asociados a la carga, con soporte para:

    - Carga general compleja (tipo_carga = 'general').
    - Circuito abierto (tipo_carga = 'abierto'): Γ_L = +1 ∠0°, z_N = ∞.
    - Cortocircuito (tipo_carga = 'corto'): Γ_L = -1 ∠180°, z_N = 0.
    - ROE conocida con carga resistiva pura (tipo_carga = 'roe_resistiva'):
        Se toma como ZL de referencia la solución en fase ZL1 = Z0·S,
        pero se reportan también ZL1 y ZL2 = Z0/S en el diccionario.

    Fórmulas usadas (ver Blake, 2004; Frenzel, 2003; HP App. Note 95-1):

        z_N = ZL / Z0

        Γ_L = (ZL - Z0)/(ZL + Z0)                 (carga general)
        S   = (1 + |Γ_L|)/(1 - |Γ_L|)            (ROE)

        |Γ|^2   = Γ_P = P_r / P_i
        P_L/P_i = 1 - |Γ|^2

        RL        = -20·log10(|Γ|)               (Return Loss > 0 dB)
        α_RL      =  20·log10(|Γ|) = -RL        (pérdida de retorno, negativa)
        L_mis     = -10·log10(1 - |Γ|^2)        (mismatch loss, > 0 dB)
        α_des     =  10·log10(1 - |Γ|^2) = -L_mis  (pérdida de desacople, negativa)

    Además calcula:
        - Ángulos de Γ y τ = 1 + Γ.
        - Parámetros de transmisión (P_trans, T_E, etc.).
        - Z_in(0) en la carga (para general) y datos de línea (λ, Rem-LE, #camp. VROE).
    """
    tipo_carga = "general"
    if datos_linea is not None:
        tipo_carga = datos_linea.get("tipo_carga", "general")

    # --- Determinar Γ_L y z_norm según el tipo de carga ---
    gamma_L: complex
    z_norm: complex
    gamma_num: Optional[complex] = None
    gamma_den: Optional[complex] = None

    if tipo_carga == "abierto":
        # Circuito abierto ideal: ZL = ∞, Γ_L = +1.
        z_norm = complex(np.inf)
        gamma_L = complex(1.0, 0.0)
        gamma_mag = 1.0

    elif tipo_carga == "corto":
        # Cortocircuito ideal: ZL = 0, Γ_L = -1.
        z_norm = 0.0 + 0.0j
        gamma_L = complex(-1.0, 0.0)
        gamma_mag = 1.0
        gamma_num = ZL - Z0
        gamma_den = ZL + Z0  # sólo por referencia, no se usa para el cálculo principal

    elif tipo_carga == "roe_resistiva":
        # ZL se tomó como ZL1 = Z0.real * S en leer_parametros_usuario().
        # Aquí simplemente calculamos z_norm y Γ_L de manera estándar.
        z_norm = ZL / Z0
        if np.isclose(z_norm, -1):
            gamma_L = complex(-1.0, 0.0)
        else:
            gamma_L = (z_norm - 1) / (z_norm + 1)
        gamma_num = ZL - Z0
        gamma_den = ZL + Z0
        gamma_mag = abs(gamma_L)

    else:
        # Caso general: carga compleja arbitraria.
        z_norm = ZL / Z0
        if np.isclose(z_norm, -1):
            gamma_L = complex(-1.0, 0.0)
        else:
            gamma_L = (z_norm - 1) / (z_norm + 1)
        gamma_num = ZL - Z0
        gamma_den = ZL + Z0
        gamma_mag = abs(gamma_L)

    # Asegurar que la magnitud esté en [0,1] por errores numéricos muy pequeños.
    if gamma_mag > 1.0 and np.isclose(gamma_mag, 1.0):
        gamma_mag = 1.0

    gamma_ang_deg = _envolver_angulo_deg(np.degrees(np.angle(gamma_L)))

    # --- ROE (SWR) ---
    if np.isclose(gamma_mag, 1.0):
        SWR = np.inf
    else:
        SWR = (1 + gamma_mag) / (1 - gamma_mag)

    dBS = np.inf if not np.isfinite(SWR) else (20 * np.log10(SWR) if SWR > 0 else 0.0)

    # --- Parámetros de potencia y pérdidas ---
    with np.errstate(divide='ignore', invalid='ignore'):
        # Return Loss clásico (positivo)
        RL_dB = np.inf if np.isclose(gamma_mag, 0.0) else -20 * np.log10(gamma_mag)
        # Coeficiente de reflexión en tensión (módulo y en dB)
        Gamma_E = gamma_mag
        Gamma_E_dB = -np.inf if np.isclose(gamma_mag, 0.0) else 20 * np.log10(gamma_mag)
        # Coeficiente de reflexión de potencia
        Gamma_P = gamma_mag ** 2
        # Potencia transmitida normalizada
        P_trans = 1.0 - Gamma_P
        # Mismatch loss (positivo)
        if P_trans <= 1e-12:
            RFL_LOSS_dB = np.inf
        else:
            RFL_LOSS_dB = -10.0 * np.log10(P_trans)
        # Pérdida de retorno con signo negativo (como en las diapositivas)
        alpha_RL_dB = -RL_dB if np.isfinite(RL_dB) else -np.inf
        # Pérdida de desacople con signo negativo
        alpha_des_dB = -RFL_LOSS_dB if np.isfinite(RFL_LOSS_dB) else -np.inf
        # Atenuación equivalente (positiva) asociada a |Γ|
        ATTEN_dB = np.inf if np.isclose(gamma_mag, 0.0) else -20 * np.log10(gamma_mag)

    # % de potencia reflejada y % de potencia absorbida (eficiencia de acople)
    porcentaje_Pr = Gamma_P * 100.0
    porcentaje_PL = P_trans * 100.0

    # Coeficiente de pérdida por ROE F clásico
    if np.isfinite(SWR) and SWR > 0:
        SW_LOSS_COEFF = (1 + SWR**2) / (2 * SWR)
    else:
        SW_LOSS_COEFF = np.inf

    # --- Coeficientes de transmisión ---
    tau_L = 1.0 + gamma_L
    tau_ang_deg = _envolver_angulo_deg(np.degrees(np.angle(tau_L)))
    T_E_mag = abs(tau_L)
    fase_coef_deg = gamma_ang_deg  # se mantiene para compatibilidad

    # --- Impedancia vista en la carga (Z_in en ℓ = 0) ---
    denom_gamma = 1.0 - gamma_L
    if np.isclose(abs(denom_gamma), 0.0, atol=1e-12):
        Z_in_0 = complex(np.inf)
    else:
        Z_in_0 = Z0 * (1.0 + gamma_L) / denom_gamma

    resultados: dict[str, Any] = dict(
        tipo_carga=tipo_carga,
        z_norm=z_norm,
        gamma_L=gamma_L,
        gamma_mag=gamma_mag,
        gamma_ang_deg=gamma_ang_deg,
        fase_coef_deg=fase_coef_deg,
        SWR=SWR,
        dBS=dBS,
        RL_dB=RL_dB,
        alpha_RL_dB=alpha_RL_dB,
        Gamma_E=Gamma_E,
        Gamma_E_dB=Gamma_E_dB,
        Gamma_P=Gamma_P,
        porcentaje_Pr=porcentaje_Pr,
        porcentaje_PL=porcentaje_PL,
        RFL_LOSS_dB=RFL_LOSS_dB,
        alpha_des_dB=alpha_des_dB,
        ATTEN_dB=ATTEN_dB,
        SW_LOSS_COEFF=SW_LOSS_COEFF,
        P_trans=P_trans,
        T_P=P_trans,
        tau_L=tau_L,
        tau_ang_deg=tau_ang_deg,
        T_E_mag=T_E_mag,
        Z_in_0=Z_in_0,
        gamma_num=gamma_num,
        gamma_den=gamma_den,
        perfiles_linea=[],
    )

    # --- Copiar datos de línea básicos ---
    if datos_linea:
        for clave in (
            "frecuencia_hz",
            "lambda_m",
            "vp_m_s",
            "vnp",
            "longitud_total_m",
            "distancias_fisicas",
            "roe_resistiva",
            "cargas_resistivas_equivalentes",
        ):
            valor = datos_linea.get(clave)
            if valor is not None:
                resultados[clave] = valor

    # --- Cálculo de Rem-LE y número de campanas VROE cuando hay L total y λ ---
    lambda_m = resultados.get("lambda_m")
    longitud_total_m = resultados.get("longitud_total_m")
    if lambda_m is not None and longitud_total_m is not None and lambda_m > 0:
        L_lambda = longitud_total_m / lambda_m
        n_camp = int(np.floor(L_lambda))
        rem_le = L_lambda - n_camp
        resultados["L_total_lambda"] = L_lambda
        resultados["N_camp_VROE"] = n_camp
        resultados["Rem_LE_lambda"] = rem_le

    return resultados

# ==== FUNCIÓN DE PROCEDIMIENTO ENRIQUECIDA CON FÓRMULAS ====
def imprimir_procedimiento(Z0, ZL, resultados):
    """
    Imprime en consola un resumen paso a paso con unidades y lo devuelve como texto.

    Incluye:
        - % de potencia reflejada y absorbida.
        - α_RL (negativa) y α_des (negativa).
        - Rem-LE y # de campanas VROE cuando hay longitud total.
        - Notación ZL′(ℓ) para las impedancias transformadas a lo largo de la línea.
        - Fórmulas explícitas de cada parámetro.
    """
    lineas: List[str] = []
    lineas.append("")
    lineas.append("=== Procedimiento paso a paso ===")

    tipo_carga = resultados.get("tipo_carga", "general")

    lineas.append("1) Datos de entrada:")

    lineas.append(f"   Z0 = {_formatear_complejo_rectangular(Z0)} Ω")

    if tipo_carga == "abierto":
        lineas.append("   Carga: circuito abierto (ZL = ∞).")
    elif tipo_carga == "corto":
        lineas.append("   Carga: cortocircuito (ZL = 0 Ω).")
    elif tipo_carga == "roe_resistiva":
        roe_val = resultados.get("roe_resistiva")
        cargas_equiv = resultados.get("cargas_resistivas_equivalentes")
        lineas.append("   Carga: resistiva pura definida por ROE.")
        if roe_val is not None:
            lineas.append(f"   ROE (SWR) especificada = {roe_val:.3f}")
        if cargas_equiv is not None:
            ZL1, ZL2 = cargas_equiv
            lineas.append(f"   ZL1 (en fase, ZL > Z0) ≈ {ZL1:.3f} Ω")
            lineas.append(f"   ZL2 (contra fase, ZL < Z0) ≈ {ZL2:.3f} Ω")
        lineas.append(f"   Para la carta se usa ZL1 como ZL de referencia: {ZL.real:.3f} Ω")
    else:
        lineas.append(f"   ZL = {_formatear_complejo_rectangular(ZL)} Ω")

    frecuencia_hz = resultados.get("frecuencia_hz")
    if frecuencia_hz is not None:
        lineas.append(f"   f = {_formatear_frecuencia_hz(frecuencia_hz)}")

    lambda_m = resultados.get("lambda_m")
    vp_m_s = resultados.get("vp_m_s")
    vnp = resultados.get("vnp")

    if vp_m_s is not None and vnp is not None:
        lineas.append(f"   v_p = {vp_m_s:.3e} m/s  (VNP = {vnp*100:.1f} % de c)")
    if lambda_m is not None:
        lineas.append(f"   λ (en la LT) = {_formatear_longitud_m(lambda_m, signo=False)}")

    longitud_total_m = resultados.get("longitud_total_m")
    if longitud_total_m is not None:
        lineas.append(f"   Longitud física total de la línea = {_formatear_longitud_m(longitud_total_m, signo=False)}")

    distancias_fisicas = resultados.get("distancias_fisicas")
    if distancias_fisicas:
        dist_texto = ", ".join(f"{valor:+.4f}" for valor in distancias_fisicas)
        lineas.append(f"   Distancias físicas solicitadas d = {dist_texto} m")

    # Rem-LE y número de campanas VROE
    L_lambda = resultados.get("L_total_lambda")
    N_camp = resultados.get("N_camp_VROE")
    Rem_LE_lambda = resultados.get("Rem_LE_lambda")
    if L_lambda is not None and N_camp is not None and Rem_LE_lambda is not None:
        lineas.append(f"   Longitud total en longitudes de onda: L/λ = {L_lambda:.3f}")
        lineas.append(f"   Número de campanas VROE ≈ {N_camp}")
        lineas.append(f"   Rem-LE (remanente eléctrico) = {Rem_LE_lambda:.3f} λ")

    # 2) Impedancia normalizada
    z_norm = resultados["z_norm"]
    lineas.append("")
    lineas.append("2) Impedancia normalizada z_N = ZL / Z0:")
    lineas.append("   Fórmula: z_N = ZL / Z0")
    if np.isfinite(z_norm.real) or np.isfinite(z_norm.imag):
        lineas.append(f"   z_N = {z_norm.real:.2f} + j{z_norm.imag:.2f} (adimensional)")
    else:
        lineas.append("   z_N = ∞ (circuito abierto ideal)")

    # 3) Coeficiente de reflexión en la carga
    gamma_L = resultados["gamma_L"]
    lineas.append("")
    lineas.append("3) Coeficiente de reflexión en la carga Γ_L:")
    lineas.append("   Fórmula general: Γ_L = (ZL − Z0) / (ZL + Z0)")
    lineas.append("   |Γ_L| = √[(Re{Γ_L})² + (Im{Γ_L})²]")
    lineas.append("   ∠Γ_L = atan2(Im{Γ_L}, Re{Γ_L})")
    gamma_num = resultados.get("gamma_num")
    gamma_den = resultados.get("gamma_den")
    if gamma_num is not None and gamma_den is not None and np.isfinite(gamma_num.real):
        lineas.append(
            f"   Numerador (ZL − Z0) = {gamma_num.real:.2f} + j{gamma_num.imag:.2f} Ω"
        )
        lineas.append(
            f"   Denominador (ZL + Z0) = {gamma_den.real:.2f} + j{gamma_den.imag:.2f} Ω"
        )
    lineas.append(
        f"   Γ_L = {gamma_L.real:.2f} + j{gamma_L.imag:.2f}"
    )
    lineas.append(f"   |Γ_L| = {resultados['gamma_mag']:.3f} (adimensional)")
    lineas.append(f"   ∠Γ_L = {resultados['gamma_ang_deg']:.2f} °")

    # 4) Parámetros derivados basados en |Γ_L|
    lineas.append("")
    lineas.append("4) Parámetros derivados basados en |Γ_L|:")
    lineas.append("   Fórmula ROE: S = (1 + |Γ_L|) / (1 − |Γ_L|)")
    lineas.append("   Fórmula RL: RL = −20·log10(|Γ_L|)")
    lineas.append("   Fórmula |Γ|²: |Γ|² = P_r / P_i")
    lineas.append("   Fórmula L_mis: L_mis = −10·log10(1 − |Γ|²)")
    lineas.append("   Fórmula α_des: α_des = −L_mis")
    lineas.append(f"   ROE (SWR) = {resultados['SWR']:.3f} (adimensional)")
    lineas.append(f"   ROE en dB (dBS) = {resultados['dBS']:.3f} dB")
    lineas.append(f"   Coef. reflexión |Γ| = {resultados['Gamma_E']:.3f} (adimensional)")
    lineas.append(f"   Coef. reflexión en dB = {resultados['Gamma_E_dB']:.3f} dB")
    lineas.append(f"   Coef. reflexión de potencia |Γ|² = {resultados['Gamma_P']:.4f} (adimensional)")
    lineas.append(f"   % Potencia reflejada = {resultados['porcentaje_Pr']:.2f} %")
    lineas.append(f"   % Potencia absorbida (η, eficiencia de acople) = {resultados['porcentaje_PL']:.2f} %")
    lineas.append(f"   Pérdida de retorno RL (positiva) = {resultados['RL_dB']:.3f} dB")
    lineas.append(f"   Pérdida de retorno α_RL (negativa) = {resultados['alpha_RL_dB']:.3f} dB")
    lineas.append(f"   Pérdida por desajuste L_mis (positiva) = {resultados['RFL_LOSS_dB']:.3f} dB")
    lineas.append(f"   Pérdida de desacople α_des (negativa) = {resultados['alpha_des_dB']:.3f} dB")
    lineas.append(f"   Coef. pérdida por ROE F = {resultados['SW_LOSS_COEFF']:.3f} (adimensional)")
    lineas.append(f"   Atenuación equivalente asociada a |Γ| = {resultados['ATTEN_dB']:.3f} dB")

    # 5) Parámetros de transmisión
    lineas.append("")
    lineas.append("5) Parámetros de transmisión:")
    lineas.append("   Fórmula T_P: T_P = P_L / P_i = 1 − |Γ|²")
    lineas.append("   Fórmula τ_L: τ_L = 1 + Γ_L")
    lineas.append(f"   Potencia transmitida normalizada P_L/P_i = {resultados['P_trans']:.4f}")
    lineas.append(f"   Coef. transmisión de potencia T_P = {resultados['T_P']:.4f}")
    lineas.append(f"   |Coef. transmisión de tensión| = {resultados['T_E_mag']:.3f}")
    lineas.append(f"   ∠(1 + Γ_L) (ángulo del coef. de transmisión τ_L) = {resultados['tau_ang_deg']:.2f} °")

    # 6) Impedancia vista en la carga
    Zi0 = resultados.get("Z_in_0")
    lineas.append("")
    lineas.append("6) Impedancia vista en la carga (ℓ = 0):")
    lineas.append("   Fórmula general: Z_in(0) = Z0 · (1 + Γ_L) / (1 − Γ_L)")
    if Zi0 is not None:
        if isinstance(Zi0, complex) and np.isfinite(Zi0):
            lineas.append(
                "   Z_in(0) = "
                f"{_formatear_complejo_rectangular(Zi0)} Ω"
            )
        else:
            lineas.append("   Z_in(0) = ∞ (circuito abierto ideal).")

    # 7) Desplazamientos a lo largo de la línea (ZL′)
    perfiles = resultados.get("perfiles_linea", [])
    if perfiles:
        lineas.append("")
        lineas.append("7) Desplazamientos a lo largo de la línea (ZL′(ℓ)):")
        lineas.append("   Fórmula general: Γ(ℓ) = Γ_L · e^{-j4πℓ}")
        lineas.append("   Fórmula ZL′(ℓ): ZL′(ℓ) = Z0 · (1 + Γ(ℓ)) / (1 − Γ(ℓ))")
        for perfil in perfiles:
            sentido = perfil["direccion"]
            longitud_eq = perfil.get("longitud", 0.0)
            longitud_in = perfil.get("longitud_original", longitud_eq)
            vueltas_medios = perfil.get("vueltas_lambda_media", 0)
            ajuste_total = perfil.get("ajuste_total_lambda", 0.0)

            if vueltas_medios:
                lineas.append(
                    f"   ℓ ingresada = {longitud_in:+.2f} λ ({sentido})"
                )
                lineas.append(
                    f"     Ajuste aplicado: {vueltas_medios} × 0.5 λ = {ajuste_total:+.2f} λ"
                )
                lineas.append(
                    f"     ℓ equivalente = {longitud_eq:+.2f} λ, rotación = {perfil['rotacion_deg']:+.2f} °"
                )
            else:
                lineas.append(
                    f"   ℓ = {longitud_eq:+.2f} λ ({sentido}), rotación = {perfil['rotacion_deg']:+.2f} °"
                )
            gamma_d = perfil["gamma"]
            tau_d = perfil["tau"]
            lineas.append(
                f"     Γ(ℓ) = {gamma_d.real:.2f} + j{gamma_d.imag:.2f}; ∠Γ(ℓ) = {perfil['gamma_ang_deg']:.2f} °"
            )
            lineas.append(
                f"     τ(ℓ) = {tau_d.real:.2f} + j{tau_d.imag:.2f}; ∠τ(ℓ) = {perfil['tau_ang_deg']:.2f} °"
            )
            Zi = perfil.get("Zi")
            if Zi is not None:
                if isinstance(Zi, complex) and np.isfinite(Zi):
                    lineas.append(
                        f"     ZL′(ℓ) = {_formatear_complejo_rectangular(Zi)} Ω (impedancia equivalente vista en ese punto)"
                    )
                    lineas.append(
                        f"     R′(ℓ) = {Zi.real:.2f} Ω, X′(ℓ) = {Zi.imag:.2f} Ω"
                    )
                else:
                    lineas.append("     ZL′(ℓ) = ∞ (equivalente a circuito abierto).")

            distancia_m = perfil.get("distancia_m")
            if distancia_m is not None:
                lineas.append(f"     d ingresada = {distancia_m:+.4f} m")

            long_eq_m = perfil.get("longitud_fisica_equivalente_m")
            if long_eq_m is not None:
                lineas.append(f"     ℓ equivalente ≈ {long_eq_m:+.4f} m")

            long_in_m = perfil.get("longitud_fisica_original_m")
            if long_in_m is not None and (distancia_m is None or not np.isclose(long_in_m, distancia_m)):
                lineas.append(f"     ℓ ingresada ≈ {long_in_m:+.4f} m")

            if perfil.get("fuente"):
                lineas.append(f"     Fuente declarada: {perfil['fuente']}")

    lineas.append("")
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

# === Función texto “curvado” sobre un anillo ===
def texto_en_arco(ax, texto: str, radio: float, angulo_centro_deg: float, ancho_grados: float = 180.0, **kwargs):
    """Dibuja texto a lo largo de un arco."""
    if not texto:
        return

    n = len(texto)
    if n == 1:
        angulos = [angulo_centro_deg]
    else:
        angulos = [
            angulo_centro_deg - ancho_grados / 2.0 + ancho_grados * (i + 0.5) / n
            for i in range(n)
        ]

    for ch, ang_deg in zip(texto, angulos):
        rad = np.radians(ang_deg)
        x = radio * np.cos(rad)
        y = radio * np.sin(rad)
        ha, va, rot = _orientacion_texto_circular(x, y)
        ax.text(x, y, ch, ha=ha, va=va, rotation=rot, rotation_mode='anchor', **kwargs)


def _mostrar_procedimiento_scrolleable(procedimiento_texto: str, texto_leyenda: str) -> None:
    """Muestra el procedimiento en una ventana con scroll; si falla, recurre a Matplotlib."""
    contenido = procedimiento_texto + ("\n\n" + texto_leyenda if texto_leyenda else "")

    try:
        import tkinter as tk
        from tkinter import scrolledtext
    except Exception:
        _mostrar_procedimiento_fallback(contenido)
        return

    try:
        root = getattr(tk, "_default_root", None)
        if root is None:
            root = tk.Tk()
            root.withdraw()

        ventana = tk.Toplevel(root)
        ventana.title("Procedimiento paso a paso")
        ventana.geometry("720x780")
        ventana.minsize(500, 400)

        widget = scrolledtext.ScrolledText(
            ventana,
            wrap=tk.WORD,
            font=("Consolas", 10),
            padx=10,
            pady=10,
        )
        widget.pack(fill="both", expand=True)
        widget.insert("1.0", contenido)
        widget.configure(state="disabled")
        ventana.focus_force()
    except Exception:
        _mostrar_procedimiento_fallback(contenido)


def _mostrar_procedimiento_fallback(contenido: str) -> None:
    """Reproduce el comportamiento original basado en Matplotlib."""
    fig = plt.figure(figsize=(7, 8))
    fig.suptitle("Procedimiento paso a paso", fontsize=11)
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    ax.text(0.01, 0.98, contenido, ha='left', va='top', fontsize=9, wrap=True)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.05)

# Función para calcular perfiles a lo largo de la línea 
def _calcular_perfiles_desplazamiento(
    gamma_L: complex,
    desplazamientos: List[Any],
    Z0,
    lambda_m: Optional[float] = None,
):
    """Genera los perfiles de Γ, τ y Z_in a lo largo de la línea para cada entrada."""
    perfiles = []
    HALF_LAMBDA = 0.5
    TOL = 1e-9
    
    for entrada in desplazamientos:
        if isinstance(entrada, dict):
            desplazamiento = float(entrada.get("longitud_norm", 0.0))
            info_extra = entrada
        else:
            desplazamiento = float(entrada)
            info_extra = {}

        num_medios = int(np.floor((abs(desplazamiento) + TOL) / HALF_LAMBDA))
        ajuste_total = 0.0
        remanente = desplazamiento
        if num_medios > 0:
            ajuste_total = math.copysign(num_medios * HALF_LAMBDA, desplazamiento)
            remanente = desplazamiento - ajuste_total
            if abs(remanente) > HALF_LAMBDA - TOL:
                ajuste_total += math.copysign(HALF_LAMBDA, desplazamiento)
                remanente -= math.copysign(HALF_LAMBDA, desplazamiento)
        
        rot_rad = -4.0 * np.pi * remanente
        gamma_d = gamma_L * np.exp(1j * rot_rad)
        tau_d = 1 + gamma_d
        denom = 1 - gamma_d
        
        if np.isclose(abs(denom), 0.0, atol=1e-12):
            Zi = complex(np.inf)
        else:
            Zi = Z0 * (1 + gamma_d) / denom
            
        direccion_ref = (
            "hacia el generador" if desplazamiento > 0
            else "hacia la carga" if desplazamiento < 0
            else "en la carga"
        )
        perfiles.append(dict(
            longitud=remanente,
            longitud_original=desplazamiento,
            direccion=direccion_ref,
            rotacion_deg=-720.0 * remanente,
            gamma=gamma_d,
            gamma_ang_deg=_envolver_angulo_deg(np.degrees(np.angle(gamma_d))),
            tau=tau_d,
            tau_ang_deg=_envolver_angulo_deg(np.degrees(np.angle(tau_d))),
            Zi=Zi,
            vueltas_lambda_media=num_medios,
            ajuste_total_lambda=ajuste_total,
            longitud_equivalente=remanente,
        ))
        perfil_actual = perfiles[-1]

        if lambda_m is not None:
            perfil_actual["lambda_m"] = lambda_m
            perfil_actual["longitud_fisica_original_m"] = desplazamiento * lambda_m
            perfil_actual["longitud_fisica_equivalente_m"] = remanente * lambda_m
            perfil_actual["remanente_fisico_m"] = remanente * lambda_m
            perfil_actual["ajuste_total_m"] = ajuste_total * lambda_m

        distancia_m = info_extra.get("distancia_m") if isinstance(info_extra, dict) else None
        if distancia_m is not None:
            perfil_actual["distancia_m"] = distancia_m
        elif lambda_m is not None:
            perfil_actual["distancia_m"] = desplazamiento * lambda_m

        if isinstance(info_extra, dict) and info_extra.get("fuente"):
            perfil_actual["fuente"] = info_extra["fuente"]

    return perfiles

# Función auxiliar para detallar longitudes desplazadas
def _detallar_perfil_longitud(idx: int, perfil: dict[str, Any]) -> Tuple[str, str, str]:
    """Genera textos consistentes para anotaciones y tooltips de una longitud eléctrica."""
    sentido = perfil.get("direccion", "")
    longitud_eq = perfil.get("longitud", 0.0)
    longitud_in = perfil.get("longitud_original", longitud_eq)
    vueltas_medios = perfil.get("vueltas_lambda_media", 0)
    ajuste_total = perfil.get("ajuste_total_lambda", 0.0)
    Zi = perfil.get("Zi")
    angulo = perfil.get("gamma_ang_deg")

    titulo = f"Longitud eléctrica #{idx}"

    etiqueta_lineas: List[str] = []
    tooltip_lineas: List[str] = [titulo]

    linea_longitud = f"ℓ{idx} = {longitud_eq:+.2f} λ"
    if not np.isclose(longitud_in, longitud_eq):
        linea_longitud += f" (eq. de {longitud_in:+.2f} λ)"
    etiqueta_lineas.append(linea_longitud)
    tooltip_lineas.append(linea_longitud)

    if sentido:
        tooltip_lineas.append(f"Dirección: {sentido}")

    if vueltas_medios:
        linea_ajuste = f"Δℓ = {ajuste_total:+.2f} λ ({vueltas_medios} × 0.5λ)"
        etiqueta_lineas.append(linea_ajuste)
        tooltip_lineas.append(linea_ajuste)

    long_eq_m = perfil.get("longitud_fisica_equivalente_m")
    if long_eq_m is not None:
        linea_eq_m = f"ℓ eq ≈ {_formatear_longitud_m(long_eq_m)}"
        etiqueta_lineas.append(linea_eq_m)
        tooltip_lineas.append(linea_eq_m)

    dist_m = perfil.get("distancia_m")
    if dist_m is not None:
        tooltip_lineas.append(f"d ingresada ≈ {_formatear_longitud_m(dist_m)}")

    long_in_m = perfil.get("longitud_fisica_original_m")
    if long_in_m is not None and (dist_m is None or not np.isclose(long_in_m, dist_m)):
        tooltip_lineas.append(f"ℓ ingresada ≈ {_formatear_longitud_m(long_in_m)}")

    remanente_m = perfil.get("remanente_fisico_m")
    if remanente_m is not None and vueltas_medios:
        tooltip_lineas.append(f"Remanente ≈ {_formatear_longitud_m(remanente_m)}")

    if isinstance(Zi, complex) and np.isfinite(Zi):
        zi_txt = _formatear_complejo_rectangular(Zi)
    else:
        zi_txt = "∞"
    etiqueta_lineas.append(f"ZL′ = {zi_txt} Ω")
    tooltip_lineas.append(f"ZL′ (impedancia equivalente) = {zi_txt} Ω")


    if angulo is not None:
        etiqueta_lineas.append(f"∠Γ = {angulo:.2f}°")
        tooltip_lineas.append(f"∠Γ = {angulo:.2f}°")

    fuente = perfil.get("fuente")
    if fuente:
        tooltip_lineas.append(f"Fuente: {fuente}")

    tooltip_unico = "\n".join(dict.fromkeys(tooltip_lineas))
    etiqueta_texto = "\n".join(etiqueta_lineas)

    return titulo, etiqueta_texto, tooltip_unico


# Función para dibujar la escala de ángulos 
def _dibujar_escala_angulos(ax):
    """Añade las escalas de ángulos de reflexión y transmisión."""
    minor_step = 10
    major_step = 30
    
    segmentos_ticks = []
    
    for ang_deg in range(0, 360, minor_step):
        rad = np.radians(ang_deg)
        r_inner = 1.0
        r_outer = 1.03 if ang_deg % major_step else ESCALA_ANGULO_INTERNA
        segmentos_ticks.append([
            (r_inner * np.cos(rad), r_inner * np.sin(rad)),
            (r_outer * np.cos(rad), r_outer * np.sin(rad))
        ])
    
    col_ticks = LineCollection(segmentos_ticks, color='silver', lw=0.4, zorder=2)
    ax.add_collection(col_ticks)

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

# Función para marcar los ángulos
def _marcar_angulos_coeficientes(ax, angulo_reflexion_deg: Optional[float], angulo_transmision_deg: Optional[float]) -> List[dict[str, Any]]:
    """Marca los ángulos de reflexión y transmisión con puntos, texto numérico y devuelve su metadata."""
    marcadores: List[dict[str, Any]] = []

    def _dibujar_marcador(angulo_deg: Optional[float], radio: float, color: str, offset: float, nombre: str) -> None:
        if angulo_deg is None or not np.isfinite(angulo_deg):
            return
        ang_rad = np.radians(angulo_deg)
        x = radio * np.cos(ang_rad)
        y = radio * np.sin(ang_rad)
        
        ax.plot(x, y, marker='o', markersize=8, markerfacecolor=color, markeredgecolor='white',
                markeredgewidth=1.2, linestyle='None', zorder=6)
        
        etiqueta = f"{angulo_deg:.1f}°"
        radio_texto = radio + offset
        x_txt = radio_texto * np.cos(ang_rad)
        y_txt = radio_texto * np.sin(ang_rad)
        ha, va, _ = _orientacion_texto_circular(x_txt, y_txt)
        ax.text(x_txt, y_txt, etiqueta, fontsize=7, color=color, ha=ha, va=va, zorder=6)
        
        marcadores.append(dict(
            nombre=nombre,
            angulo_deg=angulo_deg,
            posicion=(x, y),
            color=color,
            radio_det=0.05,
            tooltip=f"{nombre}: {angulo_deg:.2f}°",
        ))

    _dibujar_marcador(angulo_reflexion_deg, ESCALA_ANGULO_INTERNA, 'tab:green', 0.08, "Ángulo de reflexión")
    _dibujar_marcador(angulo_transmision_deg, ESCALA_ANGULO_EXTERNA, 'tab:blue', 0.05, "Ángulo de transmisión")

    return marcadores

# Función original de longitudes eléctricas (para la figura interactiva)
def _dibujar_longitudes_electricas(ax, perfiles_linea: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """Dibuja los puntos asociados a las longitudes eléctricas solicitadas."""
    marcadores: List[dict[str, Any]] = []

    if not perfiles_linea:
        return marcadores

    for idx, perfil in enumerate(perfiles_linea, start=1):
        gamma_d = perfil.get("gamma")
        if gamma_d is None:
            continue

        try:
            gamma_d = complex(gamma_d)
        except Exception:
            continue

        x = gamma_d.real
        y = gamma_d.imag
        ax.plot(x, y, marker='o', markersize=6, markerfacecolor='purple',
                markeredgecolor='white', markeredgewidth=1.0,
                linestyle='None', zorder=6)

        titulo, etiqueta, tooltip = _detallar_perfil_longitud(idx, perfil)

        norma = np.hypot(x, y)
        if np.isclose(norma, 0.0):
            direccion_vec = np.array([1.0, 0.0])
        else:
            direccion_vec = np.array([x, y]) / norma

        offset_text = 0.35
        punto_texto = np.array([x, y]) + direccion_vec * offset_text

        ha = 'left' if punto_texto[0] >= 0 else 'right'
        va = 'bottom' if punto_texto[1] >= 0 else 'top'

        ax.annotate(
            etiqueta,
            xy=(x, y),
            xytext=(punto_texto[0], punto_texto[1]),
            textcoords='data',
            ha=ha, va=va, fontsize=6, color='purple',
            bbox=dict(boxstyle='round,pad=0.2', fc='#f4ecff', ec='purple', lw=0.6, alpha=0.95),
            arrowprops=dict(arrowstyle='->', color='purple', lw=0.6, shrinkA=0.0, shrinkB=1.0),
            zorder=6,
        )

        marcadores.append(dict(
            nombre=titulo,
            etiqueta=etiqueta,
            tooltip=tooltip,
            posicion=(x, y),
            posicion_texto=(punto_texto[0], punto_texto[1]),
            direccion=perfil.get("direccion", ""),
            angulo=perfil.get("gamma_ang_deg"),
            angulo_deg=perfil.get("gamma_ang_deg"),
            longitud=perfil.get("longitud", 0.0),
            longitud_original=perfil.get("longitud_original", 0.0),
            vueltas_lambda_media=perfil.get("vueltas_lambda_media", 0),
            ajuste_total_lambda=perfil.get("ajuste_total_lambda", 0.0),
            Zi=perfil.get("Zi"),
            distancia_m=perfil.get("distancia_m"),
            longitud_fisica_equivalente_m=perfil.get("longitud_fisica_equivalente_m"),
            longitud_fisica_original_m=perfil.get("longitud_fisica_original_m"),
            fuente=perfil.get("fuente"),
            radio_det=0.05,
        ))

    return marcadores

# Función para dibujar la escala de longitudes de onda 
def _dibujar_escala_longitudes(ax):
    """Añade las escala de longitudes de onda hacia generador y carga."""
    segmentos = []
    
    # Generador
    generador_vals = np.arange(0.0, 0.55, 0.05)
    for valor in generador_vals:
        ang_deg = valor * 360.0
        rad = np.radians(ang_deg)
        segmentos.append([
            (ESCALA_LONGITUD_GENERADOR * np.cos(rad), ESCALA_LONGITUD_GENERADOR * np.sin(rad)),
            ((ESCALA_LONGITUD_GENERADOR + 0.04) * np.cos(rad), (ESCALA_LONGITUD_GENERADOR + 0.04) * np.sin(rad))
        ])
        
        if np.isclose(valor % 0.1, 0.0, atol=1e-9) or np.isclose(valor, 0.05):
            cos_v = np.cos(rad)
            sin_v = np.sin(rad)
            ha, va, rot = _orientacion_texto_circular(cos_v, sin_v)
            ax.text((ESCALA_LONGITUD_GENERADOR + 0.08) * cos_v,
                    (ESCALA_LONGITUD_GENERADOR + 0.08) * sin_v,
                    f"{valor:.2f}", fontsize=6, ha=ha, va=va,
                    rotation=rot, rotation_mode='anchor')

    # Carga
    carga_vals = np.arange(0.0, 0.55, 0.05)
    for valor in carga_vals:
        ang_deg = 360.0 - valor * 360.0
        rad = np.radians(ang_deg)
        segmentos.append([
            (ESCALA_LONGITUD_CARGA * np.cos(rad), ESCALA_LONGITUD_CARGA * np.sin(rad)),
            ((ESCALA_LONGITUD_CARGA + 0.04) * np.cos(rad), (ESCALA_LONGITUD_CARGA + 0.04) * np.sin(rad))
        ])

        if np.isclose(valor % 0.1, 0.0, atol=1e-9) or np.isclose(valor, 0.05):
            cos_v = np.cos(rad)
            sin_v = np.sin(rad)
            ha, va, rot = _orientacion_texto_circular(cos_v, sin_v)
            ax.text((ESCALA_LONGITUD_CARGA + 0.08) * cos_v,
                    (ESCALA_LONGITUD_CARGA + 0.08) * sin_v,
                    f"{valor:.2f}", fontsize=6, ha=ha, va=va,
                    color='dimgray', rotation=rot, rotation_mode='anchor')

    ax.add_collection(LineCollection(segmentos, color='silver', lw=0.35, zorder=2))

    ax.text(0.5, 1.015, "Longitudes de onda hacia el generador →",
            fontsize=7, ha='center', va='bottom', transform=ax.transAxes)
    ax.text(0.0, -1.40, "← Longitudes de onda hacia la carga",
            fontsize=7, ha='center', va='top')

# Función para dibujar los anillos exteriores 
def _dibujar_anillos_exteriores(ax):
    """Dibuja los anillos externos asociados a las escalas adicionales."""
    radios = [ESCALA_ANGULO_INTERNA, ESCALA_LONGITUD_GENERADOR,
              ESCALA_LONGITUD_CARGA, 1.32, 1.40]
    for radio in radios:
        circulo = Circle((0.0, 0.0), radio, fill=False, color='lightgray',
                         lw=0.5, linestyle='--', zorder=1)
        ax.add_patch(circulo)

    ax.text(-1.48, 0.0,
            "Reactancia inductiva (+jX/Z0)\nSusceptancia capacitiva (+jB/Y0)",
            fontsize=7, rotation=90, ha='center', va='center')
    ax.text(1.48, 0.0,
            "Reactancia capacitiva (-jX/Z0)\nSusceptancia inductiva (-jB/Y0)",
            fontsize=7, rotation=270, ha='center', va='center')
    ax.text(0.5, 1.03, "Ángulo de reflexión",
            fontsize=7, ha='center', va='bottom', color='black',
            transform=ax.transAxes)
    ax.text(0.0, -1.48, "Longitudes de onda",
            fontsize=7, ha='center', va='top', color='dimgray')

# Función para dibujar la carta de Smith completa
def dibujar_carta_smith(ax):
    """Dibuja la retícula normalizada de la carta de Smith."""
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.axhline(0, color='lightgray', lw=0.5)
    ax.axvline(0, color='lightgray', lw=0.5)

    theta = np.linspace(0, 2*np.pi, 400)
    # Circulo exterior
    ax.plot(np.cos(theta), np.sin(theta), color='black', lw=1)
    ax.plot(0.0, 0.0, marker='o', color='black', markersize=4, zorder=5)
    ax.text(0.04, 0.04, "1 + j0", fontsize=7, ha='left', va='bottom', color='black')

    segmentos_rejilla = []

    # Círculos de Resistencia Constante
    resistencias = [0.1, 0.2, 0.3, 0.5, 1, 2, 5, 10]
    for r in resistencias:
        centro = r / (1 + r)
        radio = 1 / (1 + r)
        x = centro + radio * np.cos(theta)
        y = radio * np.sin(theta)
        segmentos_rejilla.append(np.column_stack((x, y)))

    # Arcos de Reactancia Constante
    reactancias = [0.1, 0.2, 0.3, 0.5, 1, 2, 5, 10]
    phi = np.linspace(-np.pi, np.pi, 400)
    for x_val in reactancias:
        for signo in (+1, -1):
            x_norm = signo * x_val
            centro_x = 1.0
            centro_y = 1.0 / x_norm
            radio = 1.0 / abs(x_norm)
            
            x = centro_x + radio * np.cos(phi)
            y = centro_y + radio * np.sin(phi)
            
            mask = (x**2 + y**2) <= 1.0001
            if np.any(mask):
                segmentos_rejilla.append(np.column_stack((x[mask], y[mask])))

    lc = LineCollection(segmentos_rejilla, colors='lightgray', linewidths=0.7)
    ax.add_collection(lc)

    ax.set_xlabel(r'$\Re\{\Gamma\}$', labelpad=16)
    ax.set_ylabel(r'$\Im\{\Gamma\}$', labelpad=16)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Marcadores en el eje real
    valores_r = [0.0] + resistencias + [np.inf]
    for r in valores_r:
        if np.isinf(r):
            x_pos = 1.0
            etiqueta_r = "∞"
            etiqueta_g = "0"
        else:
            x_pos = (r - 1.0) / (r + 1.0)
            etiqueta_r = f"{r:g}"
            etiqueta_g = "∞" if r == 0.0 else f"{(1.0 / r):g}"
        if -1.05 <= x_pos <= 1.05:
            ax.plot([x_pos, x_pos], [0.0, -0.025], color='gray', lw=0.45, zorder=4)
            ax.text(x_pos, -0.06, etiqueta_r, ha='center', va='top',
                    fontsize=6, color='black')
            ax.text(x_pos, 0.06, etiqueta_g, ha='center', va='bottom',
                    fontsize=6, color='dimgray')

    _dibujar_anillos_exteriores(ax)
    _dibujar_escala_angulos(ax)
    _dibujar_escala_longitudes(ax)

# Funciones auxiliares de cálculo inverso (para regletas)
def _calcular_gamma_desde_S(S):
    return (S - 1.0) / (S + 1.0)

def _calcular_gamma_desde_RL(RL_dB):
    return 10.0 ** (-RL_dB / 20.0)

def _calcular_gamma_desde_GammaP(Gamma_P):
    return np.sqrt(Gamma_P)

def _calcular_gamma_desde_RFL_LOSS(L_dB):
    arg = 1.0 - 10.0 ** (-L_dB / 10.0)
    return np.sqrt(max(0.0, arg))

def _calcular_gamma_desde_ATTEN(A_dB):
    return 10.0 ** (-A_dB / 20.0)

def _calcular_gamma_desde_SW_LOSS(F):
    F = np.asarray(F)
    F_clamped = np.maximum(F, 1.0)
    S = F_clamped + np.sqrt(np.square(F_clamped) - 1.0)
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
    
    num_reglas = len(regletas)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.7, num_reglas + 0.7)
    ax.axis('off')
    
    lineas_base = []
    ticks_regla = []

    for idx, reg in enumerate(regletas):
        y = num_reglas - idx - 0.5
        lineas_base.append([(0, y), (1.0, y)])

        for valor in reg["valores"]:
            x = reg["map_gamma"](valor)
            if np.isfinite(x) and 0 <= x <= 1:
                ticks_regla.append([(x, y - 0.08), (x, y + 0.08)])
                ax.text(x, y + 0.1, f"{valor:g}", ha='center', va='bottom', fontsize=6)

        if np.isfinite(gamma_mag):
            ax.vlines(gamma_mag, y - 0.15, y + 0.15, colors='red', lw=1.2)

        valor_actual = resultados.get(reg["clave_resultado"])
        if valor_actual is not None and np.isfinite(valor_actual):
            x_val = reg["map_gamma"](valor_actual)
            if np.isfinite(x_val) and 0 <= x_val <= 1:
                ax.plot(x_val, y, marker='o', color='red', markersize=4, zorder=5)
                ax.text(x_val, y - 0.12, reg["formato"](valor_actual),
                        ha='center', va='top', fontsize=6, color='red')
        elif valor_actual is not None:
            texto_valor = reg["formato"](valor_actual)
            if idx < num_reglas / 2:
                ax.text(-0.25, y, texto_valor, ha='left', va='center',
                        fontsize=6, color='red')
            else:
                ax.text(1.25, y, texto_valor, ha='right', va='center',
                        fontsize=6, color='red')

        if idx < num_reglas / 2:
            x_text = -0.2
            ax.text(x_text, y, reg["etiqueta"], ha='right', va='center', fontsize=7)
            ax.plot([x_text + 0.02, 0.0], [y, y], color='gray', lw=0.5)
        else:
            x_text = 1.2
            ax.text(x_text, y, reg["etiqueta"], ha='left', va='center', fontsize=7)
            ax.plot([1.0, x_text - 0.02], [y, y], color='gray', lw=0.5)

    ax.add_collection(LineCollection(lineas_base, colors='black', lw=0.6))
    ax.add_collection(LineCollection(ticks_regla, colors='black', lw=0.5))

    ax.text(0.5, 1.10, "PARÁMETROS ESCALADOS RADIALMENTE",
            ha='center', va='bottom', fontsize=9, fontweight='bold',
            transform=ax.transAxes)
    ax.text(0.5, 1, "Hacia la carga →   ← Hacia el generador",
            ha='center', va='top', fontsize=9, transform=ax.transAxes)

# === NUEVAS FUNCIONES: texto estático con “datos de hover” para exportación ===
def _formato_valores_estatico(Z0, ZL, resultados) -> str:
    """Texto equivalente al mostrado en el hover principal de Γ_L."""
    gamma_L = resultados["gamma_L"]
    z = resultados["z_norm"]
    tipo_carga = resultados.get("tipo_carga", "general")

    cabecera = []
    if tipo_carga == "abierto":
        cabecera.append("Carga: CTO. ABIERTO (ZL = ∞)")
    elif tipo_carga == "corto":
        cabecera.append("Carga: CTO. (ZL = 0 Ω)")
    elif tipo_carga == "roe_resistiva":
        cabecera.append("Carga: resistiva pura definida por ROE")
    else:
        cabecera.append(f"ZL = {_formatear_complejo_rectangular(ZL)} Ω")

    lineas = cabecera + [
        f"Z0  = {_formatear_complejo_rectangular(Z0)} Ω",
        f"z_N = {z.real:.2f} + j{z.imag:.2f}",
        f"Γ_L = {gamma_L.real:.2f} + j{gamma_L.imag:.2f}",
        f"|Γ_L| = {resultados['gamma_mag']:.3f}",
        f"∠Γ_L = {resultados['gamma_ang_deg']:.2f}°",
        f"ROE = {resultados['SWR']:.3f}",
        f"RL = {resultados['RL_dB']:.3f} dB",
        f"α_RL = {resultados['alpha_RL_dB']:.3f} dB",
        f"%Pr = {resultados['porcentaje_Pr']:.2f} %",
        f"η = {resultados['porcentaje_PL']:.2f} % (PL/Pi)",
        f"α_des = {resultados['alpha_des_dB']:.3f} dB",
    ]

    frecuencia_hz = resultados.get("frecuencia_hz")
    if frecuencia_hz is not None:
        lineas.append(f"f = {_formatear_frecuencia_hz(frecuencia_hz)}")

    lambda_m = resultados.get("lambda_m")
    if lambda_m is not None:
        lineas.append(f"λ = {_formatear_longitud_m(lambda_m, signo=False)}")

    return "\n".join(lineas)

# Función para anotar Γ_L en la exportación
def _anotar_gamma_principal_export(ax, Z0, ZL, resultados):
    """Dibuja, en la carta exportada, el cuadro con los datos de Γ_L como si fuera hover."""
    gamma_L = resultados["gamma_L"]
    x0, y0 = gamma_L.real, gamma_L.imag
    texto = _formato_valores_estatico(Z0, ZL, resultados)

    vec = np.array([x0, y0])
    if np.isclose(np.linalg.norm(vec), 0.0):
        vec = np.array([1.0, 0.0])
    vec = vec / np.linalg.norm(vec)

    offset = 0.40
    px, py = x0 + vec[0]*offset, y0 + vec[1]*offset
    ha = 'left' if px >= 0 else 'right'
    va = 'bottom' if py >= 0 else 'top'

    ax.annotate(
        texto,
        xy=(x0, y0),
        xytext=(px, py),
        textcoords='data',
        ha=ha, va=va,
        fontsize=7, color='black',
        bbox=dict(boxstyle='round,pad=0.3', fc='#ffffff',
                  ec='dimgray', lw=0.9, alpha=0.95),
        arrowprops=dict(arrowstyle='->', color='dimgray',
                        lw=0.8, shrinkA=0.0, shrinkB=1.0),
        zorder=10,
    )

# Función para dibujar longitudes eléctricas en la exportación
def _dibujar_longitudes_electricas_export(ax, perfiles_linea: List[dict[str, Any]]):
    """
    Dibuja para exportación:
      - El punto morado de cada longitud eléctrica
      - Un cuadro de texto con los datos equivalentes al hover:
        nombre, ℓ, Zi y ∠Γ.
    """
    if not perfiles_linea:
        return

    for idx, perfil in enumerate(perfiles_linea, start=1):
        gamma_d = perfil.get("gamma")
        if gamma_d is None:
            continue

        gamma_d = complex(gamma_d)
        x = gamma_d.real
        y = gamma_d.imag

        ax.plot(x, y, marker='o', markersize=6, markerfacecolor='purple',
                markeredgecolor='white', markeredgewidth=1.0,
                linestyle='None', zorder=6)

        _, _, tooltip = _detallar_perfil_longitud(idx, perfil)

        norma = np.hypot(x, y)
        if np.isclose(norma, 0.0):
            direccion_vec = np.array([1.0, 0.0])
        else:
            direccion_vec = np.array([x, y]) / norma

        offset_text = 0.30
        punto_texto = np.array([x, y]) + direccion_vec * offset_text

        ha = 'left' if punto_texto[0] >= 0 else 'right'
        va = 'bottom' if punto_texto[1] >= 0 else 'top'

        ax.annotate(
            tooltip,
            xy=(x, y),
            xytext=(punto_texto[0], punto_texto[1]),
            textcoords='data',
            ha=ha, va=va, fontsize=6, color='purple',
            bbox=dict(boxstyle='round,pad=0.2', fc='#f4ecff',
                      ec='purple', lw=0.6, alpha=0.95),
            arrowprops=dict(arrowstyle='->', color='purple',
                            lw=0.6, shrinkA=0.0, shrinkB=1.0),
            zorder=9,
        )

# ==== FUNCIÓN DE EXPORTACIÓN ACTUALIZADA (PDF 2 PÁGINAS) ====
def exportar_carta_smith_sola(
    Z0,
    ZL,
    resultados,
    perfiles_linea,
    nombre_base: str = "smithchart_sola",
    formato: str = "svg",
    dpi: int = 600,
    procedimiento_texto: Optional[str] = None,   # NUEVO
    texto_leyenda: Optional[str] = None          # NUEVO
) -> None:
    """
    Genera una figura separada únicamente con la carta de Smith (sin regletas)
    y la guarda en formato vectorial (SVG/PDF) o PNG de alta resolución.

    EN ESTA FIGURA:
      - El punto Γ_L tiene impreso el cuadro con los mismos datos que el hover.
      - Cada longitud eléctrica tiene su cuadro con ℓ, Zi y ∠Γ.

    Si formato='pdf' y se proporciona procedimiento_texto, se genera
    un PDF de dos páginas:
      * Página 1: carta de Smith.
      * Página 2: procedimiento paso a paso + leyenda.
    """
    gamma_L = resultados["gamma_L"]
    gamma_mag = resultados["gamma_mag"]

    fig2 = plt.figure(figsize=(9, 9))
    ax2 = fig2.add_subplot(1, 1, 1)

    dibujar_carta_smith(ax2)

    theta = np.linspace(0, 2 * np.pi, 400)
    ax2.plot(gamma_L.real, gamma_L.imag, 'ro', label=r'$\Gamma_L$')
    ax2.plot(gamma_mag * np.cos(theta), gamma_mag * np.sin(theta),
             'r--', lw=1.0, label=r'|$\Gamma$| constante')

    _marcar_angulos_coeficientes(ax2, resultados['gamma_ang_deg'],
                                 resultados['tau_ang_deg'])
    _dibujar_longitudes_electricas_export(ax2, perfiles_linea)
    _anotar_gamma_principal_export(ax2, Z0, ZL, resultados)

    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1.02),
               fontsize=8, frameon=False)

    nombre_base = (nombre_base or "smithchart_sola").strip() or "smithchart_sola"
    formato = (formato or "svg").lower()
    if formato not in ("svg", "pdf", "png"):
        formato = "svg"

    ruta = f"{nombre_base}.{formato}"

    # Si es PDF y hay procedimiento, hacer PDF multipágina
    if formato == "pdf" and procedimiento_texto:
        with PdfPages(ruta) as pdf:
            # Página 1: carta de Smith
            pdf.savefig(fig2, bbox_inches='tight')
            plt.close(fig2)

            # Página 2: procedimiento paso a paso
            fig_proc = plt.figure(figsize=(7, 8), constrained_layout=True)
            fig_proc.suptitle("Procedimiento paso a paso", fontsize=11)
            ax_proc = fig_proc.add_subplot(1, 1, 1)
            ax_proc.axis('off')

            texto_final = procedimiento_texto
            if texto_leyenda:
                texto_final = texto_final + "\n\n" + texto_leyenda

            ax_proc.text(
                0.01,
                0.98,
                texto_final,
                ha='left',
                va='top',
                fontsize=9,
                wrap=True
            )
            pdf.savefig(fig_proc, bbox_inches='tight')
            plt.close(fig_proc)
        print(f"Carta de Smith + procedimiento guardados en '{ruta}' (PDF multipágina).")
        return

    # Comportamiento original para SVG/PNG (solo imagen)
    kwargs: dict[str, Any] = dict(bbox_inches='tight')
    if formato == "png":
        kwargs["dpi"] = dpi

    fig2.savefig(ruta, **kwargs)
    plt.close(fig2)
    print(f"Carta de Smith sola guardada en '{ruta}' ({formato.upper()}).")

# Función principal para crear la gráfica completa 
def crear_grafica_completa(
    Z0,
    ZL,
    perfiles_config: Optional[List[Any]] = None,
    parametros_linea: Optional[Dict[str, Any]] = None,
):
    """Genera la figura completa con hover y regletas."""
    if perfiles_config is None:
        perfiles_config = []

    resultados = calcular_reflexion_y_parametros(Z0, ZL, parametros_linea)
    lambda_m = resultados.get("lambda_m")
    perfiles_linea = _calcular_perfiles_desplazamiento(
        resultados["gamma_L"],
        perfiles_config,
        Z0,
        lambda_m=lambda_m,
    )
    resultados["perfiles_linea"] = perfiles_linea
    procedimiento_texto = imprimir_procedimiento(Z0, ZL, resultados)
    
    gamma_L = resultados["gamma_L"]
    gamma_mag = resultados["gamma_mag"]
    
    fig = plt.figure(figsize=(9, 11), constrained_layout=True)
    fig.suptitle("Laboratorio Integrador 2025-2 - LTT93", fontsize=10)
    
    if hasattr(fig.canvas, "manager") and fig.canvas.manager:
        try:
            fig.canvas.manager.set_window_title(
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
    ax_smith.plot(gamma_mag * np.cos(theta), gamma_mag * np.sin(theta),
                  'r--', lw=1.0, label=r'|$\Gamma$| constante')
    
    marcadores_angulos = _marcar_angulos_coeficientes(ax_smith,
                                                      resultados['gamma_ang_deg'],
                                                      resultados['tau_ang_deg'])
    marcadores_longitudes = _dibujar_longitudes_electricas(ax_smith, perfiles_linea)
    marcadores_interactivos = marcadores_angulos + marcadores_longitudes
    
    ax_smith.legend(loc='upper left', bbox_to_anchor=(1.15, 1.15),
                    fontsize=8, frameon=False)
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
    
    # Figura de procedimiento para ver en pantalla (independiente del PDF)
    _mostrar_procedimiento_scrolleable(procedimiento_texto, texto_leyenda)

    anot = ax_smith.annotate(
        "", xy=(0, 0), xytext=(18, 18), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="dimgray",
                  lw=0.9, alpha=1.0),
        arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.8),
        zorder=20,
    )
    anot.set_visible(False)

    # Función para obtener el renderer actual de la figura
    def obtener_renderer() -> Optional[RendererBase]:
        candidato = getattr(fig.canvas, "get_renderer", None)
        if callable(candidato):
            candidato = candidato()
        if isinstance(candidato, RendererBase):
            return candidato

        candidato = getattr(fig.canvas, "renderer", None)
        if isinstance(candidato, RendererBase):
            return candidato

        return None

    # Función para ajustar la anotación a los bordes del eje destino
    def ajustar_a_bordes(eje_destino):
        renderer = obtener_renderer()
        if renderer is None:
            return
        try:
            bbox_ax = eje_destino.get_window_extent(renderer)
            bbox_anot = anot.get_window_extent(renderer)
        except Exception:
            return

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
    
    # Función para configurar la posición de la anotación según el eje destino
    def configurar_anotacion(eje_destino, base_point: Optional[Tuple[float, float]] = None):
        if eje_destino is ax_smith:
            if base_point is None:
                base_point = (gamma_L.real, gamma_L.imag)
            x_ref, y_ref = base_point
            dx = 25 if x_ref <= 0 else -120
            dy = 30 if y_ref <= 0 else -90
            ha = 'left' if dx >= 0 else 'right'
            va = 'bottom' if dy >= 0 else 'top'
        else:
            dx, dy, ha, va = 25, -60, 'left', 'top'

        anot.set_position((dx, dy))
        anot.set_horizontalalignment(ha)
        anot.set_verticalalignment(va)

    # Función para formatear los valores mostrados en el hover
    def formato_valores():
        z = resultados["z_norm"]
        lineas = [
            f"Z0 = {_formatear_complejo_rectangular(Z0)} Ω",
            f"ZL = {_formatear_complejo_rectangular(ZL)} Ω",
            f"z_N = {z.real:.2f} + j{z.imag:.2f}",
            f"Γ_L = {gamma_L.real:.2f} + j{gamma_L.imag:.2f}",
            f"|Γ_L| = {resultados['gamma_mag']:.2f}",
            f"∠Γ_L = {resultados['gamma_ang_deg']:.2f}°",
            f"ROE = {resultados['SWR']:.2f}",
            f"RL = {resultados['RL_dB']:.2f} dB",
        ]

        frecuencia_hz = resultados.get("frecuencia_hz")
        if frecuencia_hz is not None:
            lineas.append(f"f = {_formatear_frecuencia_hz(frecuencia_hz)}")

        lambda_m = resultados.get("lambda_m")
        if lambda_m is not None:
            lineas.append(f"λ = {_formatear_longitud_m(lambda_m, signo=False)}")

        perfiles = resultados.get("perfiles_linea", [])
        if perfiles:
            lineas.append("\nDesplazamientos:")
            for p in perfiles[:3]:
                lineas.append(f"ℓ={p['longitud']:+.2f}λ: ∠Γ={p['gamma_ang_deg']:.1f}°")
            if len(perfiles) > 3:
                lineas.append("...")
        return "\n".join(lineas)

    def on_move(event):
        if event.inaxes not in (ax_smith, ax_regletas):
            if anot.get_visible():
                anot.set_visible(False)
                fig.canvas.draw_idle()
            return

        mostrar = False
        if event.inaxes is ax_smith and event.xdata is not None and event.ydata is not None:
            x_evt, y_evt = event.xdata, event.ydata
            if np.hypot(x_evt - gamma_L.real, y_evt - gamma_L.imag) < 0.06:
                anot.xy = (gamma_L.real, gamma_L.imag)
                configurar_anotacion(ax_smith)
                anot.set_text(formato_valores())
                mostrar = True
            else:
                for marcador in marcadores_interactivos:
                    x_m, y_m = marcador['posicion']
                    if np.hypot(x_evt - x_m, y_evt - y_m) < marcador.get('radio_det', 0.05):
                        anot.xy = (x_m, y_m)
                        configurar_anotacion(ax_smith, (x_m, y_m))
                        tooltip = marcador.get('tooltip')
                        if tooltip:
                            anot.set_text(tooltip)
                        else:
                            anot.set_text(f"{marcador['nombre']}: {marcador.get('angulo_deg', 0.0):.2f}°")
                        mostrar = True
                        break
                        
        elif event.inaxes is ax_regletas and event.xdata is not None:
            if abs(event.xdata - gamma_mag) < 0.02:
                anot.xy = (gamma_mag, ax_regletas.get_ylim()[1])
                configurar_anotacion(ax_regletas)
                anot.set_text(f"|Γ| = {gamma_mag:.3f}\nSWR = {resultados['SWR']:.2f}")
                mostrar = True

        if mostrar:
            anot.set_visible(True)
            ajustar_a_bordes(event.inaxes)
        else:
            if anot.get_visible():
                anot.set_visible(False)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)

    # Pregunta por exportar carta sola
    try:
        resp = input("\n¿Deseas guardar la carta de Smith sola (sin regletas) en un archivo? [s/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        resp = 'n'

    if resp == 's':
        try:
            formato = input("Formato de salida (svg/pdf/png) [svg]: ").strip().lower() or "svg"
        except (EOFError, KeyboardInterrupt):
            formato = "svg"
        try:
            nombre_base = input("Nombre base del archivo (sin extensión) [smithchart_sola]: ").strip() or "smithchart_sola"
        except (EOFError, KeyboardInterrupt):
            nombre_base = "smithchart_sola"
        try:
            dpi_str = input("Resolución para PNG en dpi [600]: ").strip()
            dpi_val = int(dpi_str) if dpi_str else 600
        except (EOFError, KeyboardInterrupt, ValueError):
            dpi_val = 600

        exportar_carta_smith_sola(
            Z0,
            ZL,
            resultados,
            perfiles_linea,
            nombre_base=nombre_base,
            formato=formato,
            dpi=dpi_val,
            procedimiento_texto=procedimiento_texto,  # se pasa al PDF
            texto_leyenda=texto_leyenda               # se pasa al PDF
        )

    plt.show()

def main():
    Z0, ZL, perfiles_config, parametros_linea = leer_parametros_usuario()
    crear_grafica_completa(Z0, ZL, perfiles_config, parametros_linea)

if __name__ == "__main__":
    main()
