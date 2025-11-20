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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import RendererBase
from matplotlib.patches import Circle

# =============================================================================
# CONSTANTES DE DISEÑO
# =============================================================================
RADIO_CARTA = 1.0
ESCALA_ANGULO_INTERNA = 1.06
ESCALA_ANGULO_EXTERNA = 1.24
ESCALA_LONGITUD_GENERADOR = 1.18
ESCALA_LONGITUD_CARGA = 1.24
ESCALA_PARAMETROS = 1.34
POTENCIA_INCIDENTE_REF_W = 1.0
TENSION_INCIDENTE_REF_V = 1.0

# =============================================================================
# ESTRUCTURAS DE DATOS
# =============================================================================
@dataclass
class PerfilLinea:
    """Almacena datos de un perfil de línea calculado."""
    longitud: float
    longitud_original: float
    direccion: str
    rotacion_deg: float
    gamma: complex
    gamma_ang_deg: float
    tau: complex
    tau_ang_deg: float
    Zi: complex
    vueltas_lambda_media: int = 0
    ajuste_total_lambda: float = 0.0

@dataclass
class RegletaConfig:
    """Configuración para cada regleta inferior."""
    etiqueta: str
    valores: List[float]
    map_gamma: Callable[[float], float]
    clave_resultado: str
    formato: Callable[[float], str]

# =============================================================================
# FUNCIONES DE UTILIDAD MATEMÁTICA
# =============================================================================
def _envolver_angulo_deg(valor: float) -> float:
    """Normaliza un ángulo a (-180, 180] grados."""
    return ((valor + 180.0) % 360.0) - 180.0

# Compilación de regex para parsing de números complejos
_COMPLEJO_RECT_RE = re.compile(
    r"^\s*(?P<real>[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)?"
    r"(?:(?P<imag_sign>[+-]?)j(?P<imag>\d*(?:\.\d+)?(?:e[+-]?\d+)?))?\s*$",
    re.IGNORECASE,
)

def _parse_complejo_rectangular(texto: str) -> complex:
    """Parsea strings como '50', '50+j25', '-j10' a números complejos."""
    s = texto.strip().replace(',', '.').replace(' ', '').lower()
    if not s:
        raise ValueError("La cadena está vacía.")
    
    match = _COMPLEJO_RECT_RE.match(s)
    if not match:
        raise ValueError("Formato rectangular inválido.")
    
    # Extraer componentes
    real_str = match.group('real')
    imag_sign = match.group('imag_sign')
    imag_str = match.group('imag')
    
    # Verificar si tiene parte imaginaria
    tiene_imaginaria = 'j' in s
    
    # Convertir parte real
    real = float(real_str) if real_str else 0.0
    
    # Convertir parte imaginaria
    imag = 0.0
    if tiene_imaginaria:
        signo = imag_sign if imag_sign else '+'
        imag_val = float(imag_str) if imag_str else 1.0
        imag = imag_val if signo != '-' else -imag_val
    
    return complex(real, imag)

def _formatear_complejo_rectangular(valor: complex, decimales: int = 2) -> str:
    """Formatea un número complejo como 'a ± j b' con decimales especificados."""
    if not np.isfinite(valor):
        return "∞"
    
    tol = 10 ** (-(decimales + 2))
    real = 0.0 if abs(valor.real) < tol else valor.real
    imag = 0.0 if abs(valor.imag) < tol else valor.imag
    
    signo = '+' if imag >= 0 else '-'
    return f"{real:.{decimales}f} {signo} j{abs(imag):.{decimales}f}"


def _formatear_potencia(valor_w: float) -> str:
    """Devuelve potencia en W, mW, dBm y dBW con formato compacto."""
    if not np.isfinite(valor_w) or valor_w < 0:
        return "N/A"

    valor_mw = valor_w * 1e3
    if valor_w <= 0:
        dbm = -np.inf
        dbw = -np.inf
    else:
        dbm = 10.0 * np.log10(valor_w / 1e-3)
        dbw = 10.0 * np.log10(valor_w)

    def _fmt_db(valor_db: float) -> str:
        return "-∞" if not np.isfinite(valor_db) else f"{valor_db:.2f}"

    return (
        f"{valor_w:.3f} W ({valor_mw:.1f} mW, {_fmt_db(dbm)} dBm, {_fmt_db(dbw)} dBW)"
    )


def _formatear_tension(valor_v: float) -> str:
    """Devuelve tensión en V, mV, dBV y dBmV."""
    if not np.isfinite(valor_v) or valor_v < 0:
        return "N/A"

    valor_mv = valor_v * 1e3
    if valor_v <= 0:
        dbv = -np.inf
        dbmv = -np.inf
    else:
        dbv = 20.0 * np.log10(valor_v)
        dbmv = 20.0 * np.log10(valor_v / 1e-3)

    def _fmt_db(valor_db: float) -> str:
        return "-∞" if not np.isfinite(valor_db) else f"{valor_db:.2f}"

    return (
        f"{valor_v:.3f} V ({valor_mv:.1f} mV, {_fmt_db(dbv)} dBV, {_fmt_db(dbmv)} dBmV)"
    )

# =============================================================================
# ENTRADA DE USUARIO
# =============================================================================
def leer_parametros_usuario() -> Tuple[complex, complex, List[float]]:
    """
    Lee desde consola los parámetros básicos del problema.
    
    Returns
    -------
    Z0 : complex
        Impedancia característica de la línea.
    ZL : complex
        Impedancia de carga compleja.
    desplazamientos : List[float]
        Longitudes normalizadas en múltiplos de λ.
    """
    print("=== Generador interactivo de Carta de Smith (completa) ===")
    print("Alumno: David González Herrera (Carné 19221022)")
    print("Curso: LTT93 - Laboratorio Integrador 2025-2")
    print("Ingresa cada valor solicitado en formato rectangular (ej. 50, 50+j25, 75-j10) y presiona ENTER.")
    
    # Leer Z0
    while True:
        try:
            entrada_Z0 = input("Impedancia característica Z0 [Ω] (ej. 50, 50+j25): ")
            Z0 = _parse_complejo_rectangular(entrada_Z0)
            if np.isclose(abs(Z0), 0.0, atol=1e-12):
                print("Z0 no puede ser cero. Intenta de nuevo.")
                continue
            break
        except (ValueError, KeyboardInterrupt):
            print("Entrada no válida para Z0. Usa el formato a+jb, por ejemplo 50, 50+j25 o 75-j10.")
            continue
    
    # Leer ZL
    while True:
        try:
            entrada_ZL = input("Impedancia de carga ZL [Ω] (ej. 200-j50): ")
            ZL = _parse_complejo_rectangular(entrada_ZL)
            break
        except (ValueError, KeyboardInterrupt):
            print("Entrada no válida para ZL. Usa el formato a+jb, por ejemplo 200-j50 o 60+j30.")
            continue
    
    # Leer desplazamientos
    desplazamientos: List[float] = []
    while True:
        entrada_delta = input(
            "Longitudes normalizadas ℓ (en múltiplos de λ) hacia el generador;"
            " usa valores negativos si deseas moverte hacia la carga (separa con comas, deja vacío para omitir) y presiona ENTER: "
        ).strip()
        if not entrada_delta:
            break
        try:
            desplazamientos = [float(token.strip()) for token in entrada_delta.split(',') if token.strip()]
            break
        except ValueError:
            print("Entrada no válida para las longitudes. Usa números separados por comas.")
            continue
    
    return Z0, ZL, desplazamientos

# =============================================================================
# CÁLCULOS DE PARÁMETROS
# =============================================================================
def calcular_reflexion_y_parametros(Z0: complex, ZL: complex) -> Dict[str, Any]:
    """
    Calcula todos los parámetros asociados a la carta de Smith completa.
    
    Returns
    -------
    dict con los parámetros derivados.
    """
    z_norm = ZL / Z0
    gamma_L = (z_norm - 1) / (z_norm + 1)
    gamma_mag = abs(gamma_L)
    gamma_ang_deg = _envolver_angulo_deg(np.degrees(np.angle(gamma_L)))
    
    # Cálculo de SWR
    SWR = np.inf if np.isclose(gamma_mag, 1.0) else (1 + gamma_mag) / (1 - gamma_mag)
    VROE = SWR
    IROE = SWR
    
    # Parámetros en dB
    dBS = np.inf if not np.isfinite(SWR) else 20 * np.log10(SWR)
    RL_dB = np.inf if np.isclose(gamma_mag, 0.0) else -20 * np.log10(gamma_mag)
    Gamma_E_dB = RL_dB  # Equivalente
    ATTEN_dB = RL_dB
    
    # Parámetros de potencia
    Gamma_P = gamma_mag ** 2
    P_trans = 1 - Gamma_P
    
    # Pérdida por desajuste
    RFL_LOSS_dB = -10 * np.log10(P_trans) if P_trans > 0 else np.inf
    
    # Coeficiente de pérdida por ROE
    SW_LOSS_COEFF = (1 + SWR**2) / (2 * SWR) if np.isfinite(SWR) and SWR > 0 else np.inf
    
    # Coeficiente de transmisión
    tau_L = 1 + gamma_L
    tau_ang_deg = _envolver_angulo_deg(np.degrees(np.angle(tau_L)))
    T_E_mag = abs(tau_L)
    
    # Impedancia vista en la carga (redundante pero incluida por compatibilidad)
    Z_in_0 = ZL

    # Potencias tomando 1 W incidente como referencia
    P_incidente_W = POTENCIA_INCIDENTE_REF_W
    P_reflejada_W = Gamma_P * P_incidente_W
    P_trans_W = P_trans * P_incidente_W

    def _calc_dbm(valor_w: float) -> float:
        if not np.isfinite(valor_w) or valor_w <= 0:
            return -np.inf
        return 10.0 * np.log10(valor_w / 1e-3)

    def _calc_dbw(valor_w: float) -> float:
        if not np.isfinite(valor_w) or valor_w <= 0:
            return -np.inf
        return 10.0 * np.log10(valor_w)

    P_incidente_dBm = _calc_dbm(P_incidente_W)
    P_incidente_dBW = _calc_dbw(P_incidente_W)
    P_reflejada_dBm = _calc_dbm(P_reflejada_W)
    P_reflejada_dBW = _calc_dbw(P_reflejada_W)
    P_trans_dBm = _calc_dbm(P_trans_W)
    P_trans_dBW = _calc_dbw(P_trans_W)

    # Tensiones tomando 1 V rms incidente como referencia
    V_incidente_V = TENSION_INCIDENTE_REF_V
    V_reflejada_V = gamma_mag * V_incidente_V
    V_carga_V = T_E_mag * V_incidente_V

    def _calc_dbv(valor_v: float) -> float:
        if not np.isfinite(valor_v) or valor_v <= 0:
            return -np.inf
        return 20.0 * np.log10(valor_v)

    def _calc_dbmv(valor_v: float) -> float:
        if not np.isfinite(valor_v) or valor_v <= 0:
            return -np.inf
        return 20.0 * np.log10(valor_v / 1e-3)

    V_incidente_dBV = _calc_dbv(V_incidente_V)
    V_incidente_dBmV = _calc_dbmv(V_incidente_V)
    V_reflejada_dBV = _calc_dbv(V_reflejada_V)
    V_reflejada_dBmV = _calc_dbmv(V_reflejada_V)
    V_carga_dBV = _calc_dbv(V_carga_V)
    V_carga_dBmV = _calc_dbmv(V_carga_V)
    
    return {
        "z_norm": z_norm,
        "gamma_L": gamma_L,
        "gamma_mag": gamma_mag,
        "gamma_ang_deg": gamma_ang_deg,
        "fase_coef_deg": gamma_ang_deg,
        "SWR": SWR,
        "VROE": VROE,
        "IROE": IROE,
        "dBS": dBS,
        "RL_dB": RL_dB,
        "Gamma_E": gamma_mag,
        "Gamma_E_dB": Gamma_E_dB,
        "Gamma_P": Gamma_P,
        "RFL_LOSS_dB": RFL_LOSS_dB,
        "ATTEN_dB": ATTEN_dB,
        "SW_LOSS_COEFF": SW_LOSS_COEFF,
        "P_trans": P_trans,
        "T_P": P_trans,
        "tau_L": tau_L,
        "tau_ang_deg": tau_ang_deg,
        "T_E_mag": T_E_mag,
        "Z_in_0": Z_in_0,
        "P_incidente_W": P_incidente_W,
        "P_reflejada_W": P_reflejada_W,
        "P_trans_W": P_trans_W,
        "P_incidente_dBm": P_incidente_dBm,
        "P_incidente_dBW": P_incidente_dBW,
        "P_reflejada_dBm": P_reflejada_dBm,
        "P_reflejada_dBW": P_reflejada_dBW,
        "P_trans_dBm": P_trans_dBm,
        "P_trans_dBW": P_trans_dBW,
        "V_incidente_V": V_incidente_V,
        "V_reflejada_V": V_reflejada_V,
        "V_carga_V": V_carga_V,
        "V_incidente_dBV": V_incidente_dBV,
        "V_incidente_dBmV": V_incidente_dBmV,
        "V_reflejada_dBV": V_reflejada_dBV,
        "V_reflejada_dBmV": V_reflejada_dBmV,
        "V_carga_dBV": V_carga_dBV,
        "V_carga_dBmV": V_carga_dBmV,
        "perfiles_linea": [],
    }

# =============================================================================
# IMPRESIÓN DE PROCEDIMIENTO
# =============================================================================
def imprimir_procedimiento(Z0: complex, ZL: complex, resultados: Dict[str, Any]) -> str:
    """Imprime en consola un resumen paso a paso con unidades."""
    lineas = [
        "",
        "=== Procedimiento paso a paso ===",
        "1) Datos de entrada:",
        f"   Z0 = {_formatear_complejo_rectangular(Z0)} Ω",
        f"   ZL = {_formatear_complejo_rectangular(ZL)} Ω",
    ]
    
    # Impedancia normalizada
    z_norm = resultados["z_norm"]
    lineas.extend([
        "2) Impedancia normalizada z_N = ZL / Z0:",
        f"   z_N = {z_norm.real:.2f} + j{z_norm.imag:.2f} (adimensional)",
    ])
    
    # Coeficiente de reflexión
    gamma_L = resultados["gamma_L"]
    gamma_num = ZL - Z0
    gamma_den = ZL + Z0
    lineas.extend([
        "3) Coeficiente de reflexión en la carga:",
        f"   Numerador (ZL − Z0) = {gamma_num.real:.2f} + j{gamma_num.imag:.2f} Ω",
        f"   Denominador (ZL + Z0) = {gamma_den.real:.2f} + j{gamma_den.imag:.2f} Ω",
        f"   Γ_L = (ZL − Z0) / (ZL + Z0) = {gamma_L.real:.2f} + j{gamma_L.imag:.2f}",
        f"   |Γ_L| = {resultados['gamma_mag']:.2f} (adimensional)",
        f"   ∠Γ_L = {resultados['gamma_ang_deg']:.2f} °",
        f"   Ángulo de fase equivalente (coef. de fase) = {resultados['fase_coef_deg']:.2f} °",
    ])
    
    # Parámetros derivados
    lineas.extend([
        "4) Parámetros derivados basados en |Γ_L|:",
        f"   ROE (SWR) = {resultados['SWR']:.2f} (adimensional)",
        f"   VROE (Relación de onda estacionaria de tensión) = {resultados['VROE']:.2f} (adimensional)",
        f"   IROE (Relación de onda estacionaria de corriente) = {resultados['IROE']:.2f} (adimensional)",
        f"   ROE en dB (dBS) = {resultados['dBS']:.2f} dB",
        f"   Atenuación equivalente = {resultados['ATTEN_dB']:.2f} dB",
        f"   Coef. pérdida por ROE = {resultados['SW_LOSS_COEFF']:.2f} (adimensional)",
        f"   Pérdida de retorno = {resultados['RL_dB']:.2f} dB",
        f"   Coef. reflexión potencia |Γ|² = {resultados['Gamma_P']:.2f} (adimensional)",
        f"   Pérdida por desajuste = {resultados['RFL_LOSS_dB']:.2f} dB",
        f"   Coef. reflexión |Γ| = {resultados['Gamma_E']:.2f} (adimensional)",
    ])
    
    # Parámetros de transmisión
    lineas.extend([
        "5) Parámetros de transmisión:",
        f"   Potencia transmitida normalizada = {resultados['P_trans']:.2f} (adimensional)",
        f"   Coef. transmisión de potencia = {resultados['T_P']:.2f} (adimensional)",
        f"   |Coef. transmisión de tensión| = {resultados['T_E_mag']:.2f} (adimensional)",
        f"   ∠(1 + Γ_L) (ángulo del coef. de transmisión) = {resultados['tau_ang_deg']:.2f} °",
    ])
    
    Zi0 = resultados.get("Z_in_0")
    if isinstance(Zi0, complex) and np.isfinite(Zi0):
        lineas.append(f"   Z_in(0) = {_formatear_complejo_rectangular(Zi0)} Ω (impedancia vista en la carga)")
    else:
        lineas.append("   Z_in(0) = ∞ (impedancia vista en la carga)")

    lineas.extend([
        "6) Potencias (referencia 1 W incidente):",
        f"   P_incidente = {_formatear_potencia(resultados['P_incidente_W'])}",
        f"   P_reflejada = {_formatear_potencia(resultados['P_reflejada_W'])}",
        f"   P_transmitida = {_formatear_potencia(resultados['P_trans_W'])}",
    ])

    lineas.extend([
        "7) Tensiones (referencia 1 V rms incidente):",
        f"   V_incidente = {_formatear_tension(resultados['V_incidente_V'])}",
        f"   V_reflejada = {_formatear_tension(resultados['V_reflejada_V'])}",
        f"   V_en carga = {_formatear_tension(resultados['V_carga_V'])}",
    ])
    
    # Perfiles de línea
    perfiles = resultados.get("perfiles_linea", [])
    if perfiles:
        lineas.append("8) Desplazamientos a lo largo de la línea:")
        for perfil in perfiles:
            longitud_eq = perfil.longitud
            longitud_in = perfil.longitud_original
            vueltas_medios = perfil.vueltas_lambda_media
            
            if vueltas_medios:
                lineas.append(f"   ℓ ingresada = {longitud_in:+.2f} λ ({perfil.direccion})")
                lineas.append(f"     Ajuste aplicado: {vueltas_medios} × 0.5 λ = {perfil.ajuste_total_lambda:+.2f} λ")
                lineas.append(f"     ℓ equivalente = {longitud_eq:+.2f} λ, rotación = {perfil.rotacion_deg:+.2f} °")
            else:
                lineas.append(f"   ℓ = {longitud_eq:+.2f} λ ({perfil.direccion}), rotación = {perfil.rotacion_deg:+.2f} °")
            
            lineas.append(f"     Γ(ℓ) = {perfil.gamma.real:.2f} + j{perfil.gamma.imag:.2f}; ∠Γ(ℓ) = {perfil.gamma_ang_deg:.2f} °")
            lineas.append(f"     τ(ℓ) = {perfil.tau.real:.2f} + j{perfil.tau.imag:.2f}; ∠τ(ℓ) = {perfil.tau_ang_deg:.2f} °")
            
            if isinstance(perfil.Zi, complex) and np.isfinite(perfil.Zi):
                lineas.append(f"     Z_in(ℓ) = {_formatear_complejo_rectangular(perfil.Zi)} Ω (impedancia vista)")
                lineas.append(f"     R_in(ℓ) = {perfil.Zi.real:.2f} Ω, X_in(ℓ) = {perfil.Zi.imag:.2f} Ω")
            else:
                lineas.append("     Z_in(ℓ) = ∞ (impedancia vista)")
    
    lineas.append("=== Fin del resumen ===")
    
    texto = "\n".join(lineas)
    print(texto)
    return texto

# =============================================================================
# FUNCIONES DE DIBUJO
# =============================================================================
def _ajustar_rotacion_tangencial(angulo_deg: float) -> float:
    """Devuelve una rotación tangencial envuelta a [-90, 90] grados."""
    rot = ((angulo_deg + 180.0) % 360.0) - 180.0
    if rot > 90.0:
        rot -= 180.0
    elif rot < -90.0:
        rot += 180.0
    return rot

def _orientacion_texto_circular(x: float, y: float) -> Tuple[str, str, float]:
    """Determina alineaciones y rotación tangencial para texto en la circunferencia."""
    ha = 'left' if x > 0.1 else 'right' if x < -0.1 else 'center'
    va = 'bottom' if y > 0.1 else 'top' if y < -0.1 else 'center'
    angulo_deg = np.degrees(np.arctan2(y, x))
    rot = _ajustar_rotacion_tangencial(angulo_deg - 90.0)
    return ha, va, rot

def texto_en_arco(ax, texto: str, radio: float, angulo_centro_deg: float, 
                  ancho_grados: float = 180.0, **kwargs):
    """
    Dibuja texto aproximando un arco de circunferencia.
    Cada carácter se coloca en un ángulo distinto y se rota tangencialmente.
    """
    if not texto:
        return
    
    n = len(texto)
    angulos = [angulo_centro_deg] if n == 1 else [
        angulo_centro_deg - ancho_grados / 2.0 + ancho_grados * (i + 0.5) / n
        for i in range(n)
    ]
    
    for ch, ang_deg in zip(texto, angulos):
        rad = np.radians(ang_deg)
        x, y = radio * np.cos(rad), radio * np.sin(rad)
        ha, va, rot = _orientacion_texto_circular(x, y)
        ax.text(x, y, ch, ha=ha, va=va, rotation=rot, rotation_mode='anchor', **kwargs)

# =============================================================================
# CÁLCULO DE PERFILES DE LÍNEA
# =============================================================================
def _calcular_perfiles_desplazamiento(gamma_L: complex, desplazamientos: List[float], 
                                     Z0: complex) -> List[PerfilLinea]:
    """Genera perfiles de Γ, τ y Z_in a lo largo de la línea."""
    perfiles = []
    HALF_LAMBDA = 0.5
    TOL = 1e-9
    
    for desplazamiento in desplazamientos:
        desplazamiento = float(desplazamiento)
        
        # Ajustar múltiplos de λ/2
        num_medios = int(np.floor((abs(desplazamiento) + TOL) / HALF_LAMBDA))
        ajuste_total = 0.0
        remanente = desplazamiento
        
        if num_medios > 0:
            ajuste_total = math.copysign(num_medios * HALF_LAMBDA, desplazamiento)
            remanente = desplazamiento - ajuste_total
            
            # Ajuste adicional si el remanente es muy cercano a λ/2
            if abs(remanente) > HALF_LAMBDA - TOL:
                ajuste_total += math.copysign(HALF_LAMBDA, desplazamiento)
                remanente -= math.copysign(HALF_LAMBDA, desplazamiento)
        
        # Rotación y nuevos coeficientes
        rot_rad = -4.0 * np.pi * remanente
        gamma_d = gamma_L * np.exp(1j * rot_rad)
        tau_d = 1 + gamma_d
        
        # Impedancia vista
        denom = 1 - gamma_d
        Zi = complex(np.inf) if np.isclose(abs(denom), 0.0, atol=1e-12) else Z0 * (1 + gamma_d) / denom
        
        # Determinar dirección
        direccion = (
            "hacia el generador" if desplazamiento > 0
            else "hacia la carga" if desplazamiento < 0
            else "en la carga"
        )
        
        perfiles.append(PerfilLinea(
            longitud=remanente,
            longitud_original=desplazamiento,
            direccion=direccion,
            rotacion_deg=-720.0 * remanente,
            gamma=gamma_d,
            gamma_ang_deg=_envolver_angulo_deg(np.degrees(np.angle(gamma_d))),
            tau=tau_d,
            tau_ang_deg=_envolver_angulo_deg(np.degrees(np.angle(tau_d))),
            Zi=Zi,
            vueltas_lambda_media=num_medios,
            ajuste_total_lambda=ajuste_total,
        ))
    
    return perfiles

# =============================================================================
# ELEMENTOS GRÁFICOS DE LA CARTA
# =============================================================================
def _dibujar_escala_angulos(ax):
    """Añade las escalas de ángulos de reflexión y transmisión."""
    minor_step, major_step = 10, 30
    
    # Líneas menores
    for ang_deg in range(0, 360, minor_step):
        rad = np.radians(ang_deg)
        r_outer = ESCALA_ANGULO_INTERNA if ang_deg % major_step == 0 else 1.03
        ax.plot(
            [np.cos(rad), r_outer * np.cos(rad)],
            [np.sin(rad), r_outer * np.sin(rad)],
            color='silver', lw=0.4, zorder=2
        )
    
    # Etiquetas principales
    for ang_deg in range(0, 360, major_step):
        if ang_deg % 60 != 0:
            continue
        
        rad = np.radians(ang_deg)
        cos_v, sin_v = np.cos(rad), np.sin(rad)
        ha, va, rot = _orientacion_texto_circular(cos_v, sin_v)
        
        # Ángulo de reflexión
        etiqueta_ref = ang_deg if ang_deg <= 180 else ang_deg - 360
        ax.text(
            1.12 * cos_v, 1.12 * sin_v, f"{etiqueta_ref:d}°",
            fontsize=6, ha=ha, va=va, rotation=rot, rotation_mode='anchor', color='black'
        )
        
        # Ángulo de transmisión
        ang_trans = (ang_deg + 180) % 360
        etiqueta_trans = ang_trans if ang_trans <= 180 else ang_trans - 360
        rot_trans = _ajustar_rotacion_tangencial(rot + 180.0)
        ax.text(
            1.24 * cos_v, 1.24 * sin_v, f"{etiqueta_trans:d}°",
            fontsize=6, ha=ha, va=va, rotation=rot_trans, rotation_mode='anchor', color='dimgray'
        )

def _dibujar_escala_longitudes(ax):
    """Añade las escalas de longitudes de onda."""
    # Hacia el generador
    for valor in np.arange(0.0, 0.55, 0.05):
        ang_deg = valor * 360.0
        rad = np.radians(ang_deg)
        ax.plot(
            [1.18 * np.cos(rad), 1.22 * np.cos(rad)],
            [1.18 * np.sin(rad), 1.22 * np.sin(rad)],
            color='silver', lw=0.35, zorder=2
        )
        
        if np.isclose(valor % 0.1, 0.0, atol=1e-9) or np.isclose(valor, 0.05):
            cos_v, sin_v = np.cos(rad), np.sin(rad)
            ha, va, rot = _orientacion_texto_circular(cos_v, sin_v)
            ax.text(
                1.26 * cos_v, 1.26 * sin_v, f"{valor:.2f}",
                fontsize=6, ha=ha, va=va, rotation=rot, rotation_mode='anchor'
            )
    
    # Hacia la carga
    for valor in np.arange(0.0, 0.55, 0.05):
        ang_deg = 360.0 - valor * 360.0
        rad = np.radians(ang_deg)
        ax.plot(
            [1.24 * np.cos(rad), 1.28 * np.cos(rad)],
            [1.24 * np.sin(rad), 1.28 * np.sin(rad)],
            color='silver', lw=0.35, zorder=2
        )
        
        if np.isclose(valor % 0.1, 0.0, atol=1e-9) or np.isclose(valor, 0.05):
            cos_v, sin_v = np.cos(rad), np.sin(rad)
            ha, va, rot = _orientacion_texto_circular(cos_v, sin_v)
            ax.text(
                1.32 * cos_v, 1.32 * sin_v, f"{valor:.2f}",
                fontsize=6, ha=ha, va=va, rotation=rot, rotation_mode='anchor', color='dimgray'
            )
    
    # Títulos de escalas
    ax.text(0.5, 1.015, "Longitudes de onda hacia el generador →",
            fontsize=7, ha='center', va='bottom', transform=ax.transAxes)
    ax.text(0.0, -1.40, "← Longitudes de onda hacia la carga",
            fontsize=7, ha='center', va='top')

def _dibujar_anillos_exteriores(ax):
    """Dibuja los anillos externos y etiquetas."""
    for radio in [1.06, 1.18, 1.24, 1.32, 1.40]:
        circulo = Circle((0.0, 0.0), radio, fill=False, color='lightgray', lw=0.5, linestyle='--', zorder=1)
        ax.add_patch(circulo)
    
    # Etiquetas laterales
    ax.text(-1.48, 0.0, "Reactancia inductiva (+jX/Z0)\nSusceptancia capacitiva (+jB/Y0)",
            fontsize=7, rotation=90, ha='center', va='center')
    ax.text(1.48, 0.0, "Reactancia capacitiva (-jX/Z0)\nSusceptancia inductiva (-jB/Y0)",
            fontsize=7, rotation=270, ha='center', va='center')
    
    # Títulos
    ax.text(0.5, 1.03, "Ángulo de reflexión", fontsize=7, ha='center', va='bottom',
            color='black', transform=ax.transAxes)
    ax.text(0.0, -1.48, "Longitudes de onda", fontsize=7, ha='center', va='top', color='dimgray')

def dibujar_carta_smith(ax):
    """Dibuja la retícula normalizada de la carta de Smith."""
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.axhline(0, color='lightgray', lw=0.5)
    ax.axvline(0, color='lightgray', lw=0.5)
    
    # Círculo unitario
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color='black', lw=1)
    ax.plot(0.0, 0.0, marker='o', color='black', markersize=4, zorder=5)
    ax.text(0.04, 0.04, "1 + j0", fontsize=7, ha='left', va='bottom', color='black')
    
    # Resistencias constantes
    resistencias = [0.1, 0.2, 0.3, 0.5, 1, 2, 5, 10]
    for r in resistencias:
        centro, radio = r / (1 + r), 1 / (1 + r)
        x = centro + radio * np.cos(theta)
        y = radio * np.sin(theta)
        ax.plot(x, y, color='lightgray', lw=0.7)
    
    # Reactancias constantes
    reactancias = [0.1, 0.2, 0.3, 0.5, 1, 2, 5, 10]
    for x_val in reactancias:
        for signo in (+1, -1):
            centro = (1.0, 1.0 / (signo * x_val))
            radio = 1.0 / abs(x_val)
            phi = np.linspace(-np.pi, np.pi, 400)
            x = centro[0] + radio * np.cos(phi)
            y = centro[1] + radio * np.sin(phi)
            mask = x**2 + y**2 <= 1.0001
            ax.plot(x[mask], y[mask], color='lightgray', lw=0.7)
    
    # Etiquetas de ejes
    ax.set_xlabel(r'$\Re\{\Gamma\}$', labelpad=16)
    ax.set_ylabel(r'$\Im\{\Gamma\}$', labelpad=16)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Marcadores en el eje real
    valores_r = [0.0] + resistencias + [np.inf]
    for r in valores_r:
        if np.isinf(r):
            x_pos = 1.0
            etiqueta_r, etiqueta_g = "∞", "0"
        else:
            x_pos = (r - 1.0) / (r + 1.0) if r != np.inf else 1.0
            etiqueta_r = f"{r:g}"
            etiqueta_g = "∞" if r == 0.0 else f"{(1.0 / r):g}"
        
        if -1.05 <= x_pos <= 1.05:
            ax.plot([x_pos, x_pos], [0.0, -0.025], color='gray', lw=0.45, zorder=4)
            ax.text(x_pos, -0.06, etiqueta_r, ha='center', va='top', fontsize=6, color='black')
            ax.text(x_pos, 0.06, etiqueta_g, ha='center', va='bottom', fontsize=6, color='dimgray')
    
    # Elementos adicionales
    _dibujar_anillos_exteriores(ax)
    _dibujar_escala_angulos(ax)
    _dibujar_escala_longitudes(ax)

# =============================================================================
# CONFIGURACIÓN DE REGLETAS
# =============================================================================
def _crear_configuracion_regletas() -> List[RegletaConfig]:
    """Crea la configuración para todas las regletas."""
    return [
        RegletaConfig(
            "ROE (SWR)",
            [1.1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 4, 5, 10, 20, 40, 100],
            lambda S: (S - 1.0) / (S + 1.0),
            "SWR",
            lambda v: f"{v:.2f}"
        ),
        RegletaConfig(
            "Pérdida de retorno (dB)",
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 20, 30],
            lambda RL: 10.0 ** (-RL / 20.0),
            "RL_dB",
            lambda v: "∞" if not np.isfinite(v) else f"{v:.2f}"
        ),
        RegletaConfig(
            "Coef. reflexión potencia |Γ|²",
            [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            lambda GammaP: np.sqrt(GammaP),
            "Gamma_P",
            lambda v: f"{v:.2f}"
        ),
        RegletaConfig(
            "Coef. reflexión |Γ|",
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            lambda g: g,
            "Gamma_E",
            lambda v: f"{v:.2f}"
        ),
        RegletaConfig(
            "Atenuación (dB)",
            [1, 2, 3, 4, 5, 7, 10, 15],
            lambda A: 10.0 ** (-A / 20.0),
            "ATTEN_dB",
            lambda v: "∞" if not np.isfinite(v) else f"{v:.2f}"
        ),
        RegletaConfig(
            "Coef. pérdida por ROE",
            [1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2, 3, 4, 5, 10, 20],
            lambda F: (F + np.sqrt(F**2 - 1.0) - 1) / (F + np.sqrt(F**2 - 1.0) + 1),
            "SW_LOSS_COEFF",
            lambda v: "∞" if not np.isfinite(v) else f"{v:.2f}"
        ),
        RegletaConfig(
            "Pérdida por desajuste (dB)",
            [0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3, 4, 5, 6, 10, 15],
            lambda L: np.sqrt(1.0 - 10.0 ** (-L / 10.0)),
            "RFL_LOSS_dB",
            lambda v: "∞" if not np.isfinite(v) else f"{v:.2f}"
        ),
        RegletaConfig(
            "ROE en dB (dBS)",
            [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40],
            lambda d: (10.0 ** (d / 20.0) - 1) / (10.0 ** (d / 20.0) + 1),
            "dBS",
            lambda v: "∞" if not np.isfinite(v) else f"{v:.2f}"
        ),
    ]

def dibujar_regletas(ax, resultados: Dict[str, Any]):
    """Dibuja las regletas inferiores con leyendas horizontales."""
    configs = _crear_configuracion_regletas()
    gamma_mag = resultados["gamma_mag"]
    
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.7, len(configs) + 0.7)
    ax.axis('off')
    
    for idx, config in enumerate(configs):
        y = len(configs) - idx - 0.5
        
        # Línea base
        ax.hlines(y, 0, 1.0, colors='black', lw=0.6)
        
        # Marcas de valores
        for valor in config.valores:
            x = config.map_gamma(valor)
            if np.isfinite(x) and 0 <= x <= 1:
                ax.vlines(x, y - 0.08, y + 0.08, colors='black', lw=0.5)
                ax.text(x, y + 0.1, f"{valor:g}", ha='center', va='bottom', fontsize=6)
        
        # Valor actual
        if np.isfinite(gamma_mag):
            ax.vlines(gamma_mag, y - 0.15, y + 0.15, colors='red', lw=1.2)
        
        valor_actual = resultados.get(config.clave_resultado)
        if valor_actual is not None and np.isfinite(valor_actual):
            x_val = config.map_gamma(valor_actual)
            if np.isfinite(x_val) and 0 <= x_val <= 1:
                ax.plot(x_val, y, marker='o', color='red', markersize=4, zorder=5)
                ax.text(x_val, y - 0.12, config.formato(valor_actual),
                        ha='center', va='top', fontsize=6, color='red')
            elif idx < len(configs) / 2:
                ax.text(-0.25, y, config.formato(valor_actual),
                        ha='left', va='center', fontsize=6, color='red')
            else:
                ax.text(1.25, y, config.formato(valor_actual),
                        ha='right', va='center', fontsize=6, color='red')
        
        # Etiqueta de regleta
        if idx < len(configs) / 2:
            ax.text(-0.2, y, config.etiqueta, ha='right', va='center', fontsize=7)
            ax.plot([-0.18, 0.0], [y, y], color='gray', lw=0.5)
        else:
            ax.text(1.2, y, config.etiqueta, ha='left', va='center', fontsize=7)
            ax.plot([1.0, 1.18], [y, y], color='gray', lw=0.5)
    
    # Títulos
    ax.text(0.5, 1.10, "PARÁMETROS ESCALADOS RADIALMENTE",
            ha='center', va='bottom', fontsize=9, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 1.0, "Hacia la carga →   ← Hacia el generador",
            ha='center', va='top', fontsize=9, transform=ax.transAxes)

# =============================================================================
# MARCADORES INTERACTIVOS
# =============================================================================
def _marcar_angulos_coeficientes(ax, angulo_reflexion_deg: Optional[float],
                                angulo_transmision_deg: Optional[float]) -> List[Dict[str, Any]]:
    """Marca los ángulos de reflexión y transmisión."""
    marcadores: List[Dict[str, Any]] = []
    
    def dibujar_marcador(angulo_deg: Optional[float], radio: float, color: str, 
                        offset: float, nombre: str):
        if angulo_deg is None or not np.isfinite(angulo_deg):
            return
        
        ang_rad = np.radians(angulo_deg)
        x, y = radio * np.cos(ang_rad), radio * np.sin(ang_rad)
        
        ax.plot(x, y, marker='o', markersize=8, markerfacecolor=color,
                markeredgecolor='white', markeredgewidth=1.2, linestyle='None', zorder=6)
        
        # Etiqueta numérica
        x_txt, y_txt = (radio + offset) * np.cos(ang_rad), (radio + offset) * np.sin(ang_rad)
        ha, va, _ = _orientacion_texto_circular(x_txt, y_txt)
        ax.text(x_txt, y_txt, f"{angulo_deg:.1f}°", fontsize=7, color=color, ha=ha, va=va, zorder=6)
        
        marcadores.append({
            "nombre": nombre,
            "angulo_deg": angulo_deg,
            "posicion": (x, y),
            "color": color,
            "radio_det": 0.05,
        })
    
    dibujar_marcador(angulo_reflexion_deg, ESCALA_ANGULO_INTERNA, 'tab:green', 0.08, "Ángulo de reflexión")
    dibujar_marcador(angulo_transmision_deg, ESCALA_ANGULO_EXTERNA, 'tab:blue', 0.05, "Ángulo de transmisión")
    
    return marcadores

def _dibujar_longitudes_electricas(ax, perfiles: List[PerfilLinea]) -> List[Dict[str, Any]]:
    """Dibuja los puntos asociados a las longitudes eléctricas."""
    marcadores: List[Dict[str, Any]] = []
    
    for idx, perfil in enumerate(perfiles, start=1):
        x, y = perfil.gamma.real, perfil.gamma.imag
        
        ax.plot(x, y, marker='o', markersize=6, markerfacecolor='purple',
                markeredgecolor='white', markeredgewidth=1.0, linestyle='None', zorder=6)
        
        # Construir etiqueta
        if perfil.vueltas_lambda_media:
            encabezado = (
                f"ℓ{idx} = {perfil.longitud:+.2f} λ (eq. de {perfil.longitud_original:+.2f} λ)\n"
                f"Δℓ = {perfil.ajuste_total_lambda:+.2f} λ ({perfil.vueltas_lambda_media} × 0.5λ)"
            )
        else:
            encabezado = f"ℓ{idx} = {perfil.longitud:+.2f} λ"
        
        if isinstance(perfil.Zi, complex) and np.isfinite(perfil.Zi):
            etiqueta = f"{encabezado}\nZ_in = {_formatear_complejo_rectangular(perfil.Zi)} Ω"
        else:
            etiqueta = f"{encabezado}\nZ_in = ∞"
        
        # Posición del texto
        norma = max(np.hypot(x, y), 1.0)
        direccion_vec = np.array([x, y]) / norma
        punto_texto = np.array([x, y]) + direccion_vec * 0.35
        
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
        
        marcadores.append({
            "nombre": f"Longitud eléctrica #{idx}",
            "etiqueta": etiqueta,
            "posicion": (x, y),
            "posicion_texto": (punto_texto[0], punto_texto[1]),
            "direccion": perfil.direccion,
            "angulo": perfil.gamma_ang_deg,
            "longitud": perfil.longitud,
            "longitud_original": perfil.longitud_original,
            "vueltas_lambda_media": perfil.vueltas_lambda_media,
            "ajuste_total_lambda": perfil.ajuste_total_lambda,
            "Zi": perfil.Zi,
            "radio_det": 0.05,
        })
    
    return marcadores

# =============================================================================
# SISTEMA DE HOVER
# =============================================================================
class HoverManager:
    """Gestiona la anotación flotante y la detección de proximidad."""
    
    def __init__(self, ax_smith, ax_regletas, fig, gamma_L, gamma_mag):
        self.ax_smith = ax_smith
        self.ax_regletas = ax_regletas
        self.fig = fig
        self.gamma_L = gamma_L
        self.gamma_mag = gamma_mag
        
        self.anot = ax_smith.annotate(
            "", xy=(0, 0), xytext=(18, 18), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="dimgray", lw=0.9, alpha=1.0),
            arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.8),
            zorder=20,
        )
        self.anot.set_visible(False)
        
        # Asegurar z-order
        if self.anot.arrow_patch is not None:
            self.anot.arrow_patch.set_zorder(21)
        bbox_patch = self.anot.get_bbox_patch()
        if bbox_patch is not None:
            bbox_patch.set_zorder(20)
            bbox_patch.set_alpha(1.0)
    
    def obtener_renderer(self) -> Optional[RendererBase]:
        """Obtiene el renderer actual del canvas."""
        renderer = getattr(self.fig.canvas, "get_renderer", None)
        if callable(renderer):
            renderer = renderer()
        if renderer is None:
            renderer = getattr(self.fig.canvas, "renderer", None)
        if renderer is None:
            self.fig.canvas.draw()
            renderer = getattr(self.fig.canvas, "renderer", None)
        return renderer if isinstance(renderer, RendererBase) else None
    
    def ajustar_a_bordes(self, eje_destino):
        """Ajusta la anotación para que no se salga de los bordes."""
        renderer = self.obtener_renderer()
        if renderer is None:
            return
        
        bbox_ax = eje_destino.get_window_extent(renderer)
        bbox_anot = self.anot.get_window_extent(renderer)
        
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
            px, py = self.anot.get_position()
            escala = 72.0 / self.fig.dpi
            self.anot.set_position((px + delta_x * escala, py + delta_y * escala))
    
    def configurar_posicion(self, eje_destino, base_point: Optional[Tuple[float, float]] = None):
        """Configura la posición de la anotación según el eje."""
        if eje_destino is self.ax_smith:
            if base_point is None:
                base_point = (self.gamma_L.real, self.gamma_L.imag)
            
            x_ref, y_ref = base_point
            dx_def = 25 if x_ref <= 0 else -120
            dy_def = 30 if y_ref <= 0 else -90
            ha_def = 'left' if dx_def >= 0 else 'right'
            va_def = 'bottom' if dy_def >= 0 else 'top'
            
            self.anot.set_position((dx_def, dy_def))
            self.anot.set_horizontalalignment(ha_def)  # type: ignore[arg-type]
            self.anot.set_verticalalignment(va_def)    # type: ignore[arg-type]
        else:
            self.anot.set_position((25, -60))
            self.anot.set_horizontalalignment('left')  # type: ignore[arg-type]
            self.anot.set_verticalalignment('top')    # type: ignore[arg-type]
    
    def mostrar(self, x: float, y: float, texto: str, eje_destino):
        """Muestra la anotación en la posición especificada."""
        self.anot.xy = (x, y)
        self.anot.set_text(texto)
        self.configurar_posicion(eje_destino)
        self.anot.set_visible(True)
        self.ajustar_a_bordes(eje_destino)
        self.fig.canvas.draw_idle()
    
    def ocultar(self):
        """Oculta la anotación si está visible."""
        if self.anot.get_visible():
            self.anot.set_visible(False)
            self.fig.canvas.draw_idle()

# =============================================================================
# GENERACIÓN DE TEXTO PARA HOVER
# =============================================================================
def _generar_texto_base(Z0: complex, ZL: complex, resultados: Dict[str, Any]) -> str:
    """Genera el texto base para el hover."""
    z_norm = resultados["z_norm"]
    gamma_L = resultados["gamma_L"]
    
    lineas = [
        f"Z0 = {_formatear_complejo_rectangular(Z0)} Ω",
        f"ZL = {_formatear_complejo_rectangular(ZL)} Ω",
        f"z_N = {z_norm.real:.2f} + j{z_norm.imag:.2f} (Ad)",
        f"Γ_L = {gamma_L.real:.2f} + j{gamma_L.imag:.2f}",
        f"|Γ_L| = {resultados['gamma_mag']:.2f} (Ad)",
        f"∠Γ_L = {resultados['gamma_ang_deg']:.2f}°",
    ]
    
    Zi0 = resultados.get("Z_in_0")
    if isinstance(Zi0, complex) and np.isfinite(Zi0):
        lineas.append(f"Z_in(0) = {_formatear_complejo_rectangular(Zi0)} Ω")
    else:
        lineas.append("Z_in(0) = ∞")
    
    lineas.extend([
        f"Coef. de fase = {resultados['fase_coef_deg']:.2f}°",
        f"ROE (SWR) = {resultados['SWR']:.2f} (Ad)",
        f"VROE = {resultados['VROE']:.2f} (Ad)",
        f"IROE = {resultados['IROE']:.2f} (Ad)",
        f"ROE en dB (dBS) = {resultados['dBS']:.2f} dB",
        f"Pérdida de retorno = {resultados['RL_dB']:.2f} dB",
        f"|Γ| (coef. reflexión de tensión) = {resultados['Gamma_E']:.2f} (Ad)",
        f"|Γ| en dB = {resultados['Gamma_E_dB']:.2f} dB",
        f"Coef. reflexión potencia |Γ|² = {resultados['Gamma_P']:.2f} (Ad)",
        f"Potencia transmitida (1−|Γ|²) = {resultados['P_trans']:.2f} (Ad)",
        f"Pérdida por desajuste = {resultados['RFL_LOSS_dB']:.2f} dB",
        f"Atenuación equivalente = {resultados['ATTEN_dB']:.2f} dB",
        f"Coef. pérdida por ROE = {resultados['SW_LOSS_COEFF']:.2f} (Ad)",
        f"Coef. transmisión de potencia = {resultados['T_P']:.2f} (Ad)",
        f"|Coef. transmisión de tensión| = {resultados['T_E_mag']:.2f} (Ad)",
        f"∠(1 + Γ_L) = {resultados['tau_ang_deg']:.2f}°",
    ])

    lineas.extend([
        "",
        "Potencias (ref. 1 W):",
        f"  P_incidente = {_formatear_potencia(resultados['P_incidente_W'])}",
        f"  P_reflejada = {_formatear_potencia(resultados['P_reflejada_W'])}",
        f"  P_transmitida = {_formatear_potencia(resultados['P_trans_W'])}",
        "",
        "Tensiones (ref. 1 V rms):",
        f"  V_incidente = {_formatear_tension(resultados['V_incidente_V'])}",
        f"  V_reflejada = {_formatear_tension(resultados['V_reflejada_V'])}",
        f"  V_en carga = {_formatear_tension(resultados['V_carga_V'])}",
    ])
    
    return "\n".join(lineas)

def _generar_texto_perfil(marcador: Dict[str, Any]) -> str:
    """Construye el texto de hover para un marcador de longitud eléctrica."""
    lineas = [f"{marcador.get('nombre', 'Longitud eléctrica')}:"]

    etiqueta = marcador.get('etiqueta')
    if etiqueta:
        lineas.append(etiqueta)
    else:
        longitud = marcador.get('longitud', 0.0)
        lineas.append(f"ℓ = {longitud:+.2f} λ")

    direccion = marcador.get('direccion')
    if direccion:
        lineas.append(f"Dirección: {direccion}")

    vueltas = marcador.get('vueltas_lambda_media', 0)
    if vueltas:
        longitud_eq = marcador.get('longitud', 0.0)
        longitud_in = marcador.get('longitud_original', longitud_eq)
        ajuste_total = marcador.get('ajuste_total_lambda', 0.0)
        lineas.append(
            f"Equivalente: {longitud_eq:+.3f} λ (entrada {longitud_in:+.3f} λ)"
        )
        lineas.append(
            f"Ajuste aplicado: {vueltas} × 0.5 λ = {ajuste_total:+.3f} λ"
        )

    angulo = marcador.get('angulo')
    if angulo is not None and np.isfinite(angulo):
        lineas.append(f"∠Γ(ℓ) = {angulo:.2f}°")

    Zi = marcador.get('Zi')
    if Zi is not None:
        if isinstance(Zi, complex) and np.isfinite(Zi):
            lineas.append(f"Z_in(ℓ) = {_formatear_complejo_rectangular(Zi)} Ω")
        else:
            lineas.append("Z_in(ℓ) = ∞")

    return "\n".join(lineas)

# =============================================================================
# FUNCIÓN PRINCIPAL DE GRÁFICOS
# =============================================================================
def crear_grafica_completa(Z0: complex, ZL: complex, desplazamientos: Optional[List[float]] = None):
    """Genera la figura completa con hover y regletas."""
    if desplazamientos is None:
        desplazamientos = []
    
    # Cálculos principales
    resultados = calcular_reflexion_y_parametros(Z0, ZL)
    perfiles = _calcular_perfiles_desplazamiento(resultados["gamma_L"], desplazamientos, Z0)
    resultados["perfiles_linea"] = perfiles
    
    # Imprimir procedimiento
    procedimiento_texto = imprimir_procedimiento(Z0, ZL, resultados)
    
    # Crear figura
    fig = plt.figure(figsize=(9, 11), constrained_layout=True)
    fig.suptitle("Laboratorio Integrador 2025-2 - LTT93", fontsize=10)
    
    # Configurar título de ventana
    gestor_ventana = getattr(fig.canvas, "manager", None)
    if gestor_ventana is not None:
        try:
            gestor_ventana.set_window_title("Carta de Smith - David González Herrera (19221022)")
        except Exception:
            pass
    
    # Layout
    gs = fig.add_gridspec(2, 1, height_ratios=[5.0, 1.6])
    ax_smith = fig.add_subplot(gs[0, 0])
    ax_regletas = fig.add_subplot(gs[1, 0])
    
    # Dibujar carta y elementos
    dibujar_carta_smith(ax_smith)
    gamma_L = resultados["gamma_L"]
    gamma_L_x = gamma_L.real
    gamma_L_y = gamma_L.imag
    gamma_mag = resultados["gamma_mag"]
    ax_smith.plot(gamma_L_x, gamma_L_y, 'ro', label=r'$\Gamma_L$')
    
    # Círculo de |Γ| constante
    theta = np.linspace(0, 2*np.pi, 400)
    ax_smith.plot(
        gamma_mag * np.cos(theta),
        gamma_mag * np.sin(theta),
        'r--', lw=1.0, label=r'|$\Gamma$| constante'
    )
    
    # Marcadores
    marcadores_angulos = _marcar_angulos_coeficientes(
        ax_smith, resultados['gamma_ang_deg'], resultados['tau_ang_deg']
    )
    marcadores_longitudes = _dibujar_longitudes_electricas(ax_smith, perfiles)
    marcadores_interactivos = marcadores_angulos + marcadores_longitudes

    # Precomputar texto de hover para cada longitud eléctrica
    for marcador in marcadores_longitudes:
        marcador["texto_hover"] = _generar_texto_perfil(marcador)
    
    # Leyenda
    ax_smith.legend(loc='upper left', bbox_to_anchor=(1.15, 1.15), fontsize=8, frameon=False)
    
    # Regletas
    dibujar_regletas(ax_regletas, resultados)
    
    # Figura de procedimiento
    fig_procedimiento = plt.figure(figsize=(7, 8), constrained_layout=True)
    fig_procedimiento.suptitle("Procedimiento paso a paso", fontsize=11)
    ax_proc = fig_procedimiento.add_subplot(1, 1, 1)
    ax_proc.axis('off')
    
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
    
    ax_proc.text(
        0.01, 0.98, procedimiento_texto + "\n\n" + texto_leyenda,
        ha='left', va='top', fontsize=9, wrap=True
    )
    
    # Sistema de hover
    hover_manager = HoverManager(ax_smith, ax_regletas, fig, gamma_L, gamma_mag)

    # Precomputar texto base de hover
    texto_base_hover = _generar_texto_base(Z0, ZL, resultados)
    
    def on_move(event):
        if event.inaxes not in (ax_smith, ax_regletas):
            hover_manager.ocultar()
            return
        
        mostrar = False
        
        if event.inaxes is ax_smith and event.xdata is not None and event.ydata is not None:
            x_evt, y_evt = event.xdata, event.ydata
            
            # Detectar cercanía a Γ_L
            if np.hypot(x_evt - gamma_L_x, y_evt - gamma_L_y) < 0.06:
                hover_manager.mostrar(gamma_L_x, gamma_L_y, texto_base_hover, ax_smith)
                mostrar = True
            
            # Detectar marcadores
            else:
                for marcador in marcadores_interactivos:
                    x_m, y_m = marcador['posicion']
                    if np.hypot(x_evt - x_m, y_evt - y_m) < marcador.get('radio_det', 0.05):
                        if marcador['nombre'].startswith("Longitud"):
                            texto = marcador.get("texto_hover") or _generar_texto_perfil(marcador)
                        else:
                            texto = f"{marcador['nombre']}: {marcador['angulo_deg']:.2f}°"
                        
                        hover_manager.mostrar(x_m, y_m, texto, ax_smith)
                        mostrar = True
                        break
        
        elif event.inaxes is ax_regletas and event.xdata is not None:
            if abs(event.xdata - gamma_mag) < 0.02:
                hover_manager.mostrar(gamma_mag, ax_regletas.get_ylim()[1],
                                     texto_base_hover, ax_regletas)
                mostrar = True
        
        if not mostrar:
            hover_manager.ocultar()
    
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.show()

# =============================================================================
# FUNCIÓN MAIN
# =============================================================================
def main():
    """Función principal para ejecutar el script."""
    try:
        Z0, ZL, desplazamientos = leer_parametros_usuario()
        crear_grafica_completa(Z0, ZL, desplazamientos)
    except KeyboardInterrupt:
        print("\n\nEjecución interrumpida por el usuario.")
        return
    except Exception as e:
        print(f"\n\nError inesperado: {str(e)}")
        raise

if __name__ == "__main__":
    main()
