# Carta de Smith Interactiva

Herramienta interactiva escrita en Python para graficar la carta de Smith completa empleada en el curso LTT93 (Laboratorio Integrador 2025-2). El script genera la malla normalizada, calcula los parámetros clásicos de una línea de transmisión y muestra tanto los valores normalizados como los desnormalizados en consola y en la interfaz gráfica.

## Características clave

- **Captura guiada de impedancias**: durante la ejecución se ingresan `Z0`, la parte real `R_L` y la parte imaginaria `X_L` de `ZL`. No se aceptan cadenas rectangulares o polares; para `ZL = 201.3 − j50` se introduce `201.3` y `-50` cuando se solicitan las partes real e imaginaria.
- **Resultados normalizados y desnormalizados**: se calculan `z_N`, `Γ_L`, `|Γ_L|`, `∠Γ_L`, el numerador `ZL − Z0`, el denominador `ZL + Z0`, y la impedancia vista `Z_in(ℓ)` junto con `R_in` y `X_in` para cada desplazamiento.
- **Análisis de parámetros derivados**: ROE (SWR) lineal y en dB, pérdidas de retorno, pérdidas por desajuste, coeficiente de pérdida por ROE, potencia transmitida, coeficientes de transmisión de potencia y tensión.
- **Perfiles a lo largo de la línea**: `_calcular_perfiles_desplazamiento` rota `Γ` y `τ` para cada longitud normalizada `ℓ` y obtiene los valores de `Z_in(ℓ)` automáticamente.
- **Visualización enriquecida**: carta de Smith completa, círculos de reflexión constante, escalas suplementarias de ángulo, longitud de onda y susceptancia, y regletas inferiores que resaltan el valor calculado.
- **Experiencia interactiva**: los eventos de desplazamiento del ratón muestran en cuadros emergentes los mismos resultados que el reporte textual, sincronizando datos normalizados y desnormalizados.
- **Procedimiento documentado**: `imprimir_procedimiento` imprime un resumen paso a paso en consola y lo replica en una figura auxiliar.

## Requisitos de software

- Python 3.9 o superior.
- Bibliotecas `matplotlib` y `numpy`.

Instala las dependencias con:

```powershell
py -3 -m pip install matplotlib numpy
```

## Ejecución paso a paso

1. Abre PowerShell en el directorio del proyecto y ejecuta:

   ```powershell
   py -3 smithchart.py
   ```

2. Introduce los valores cuando se soliciten:
   - `Z0`: impedancia característica (ej. `50`). Debe ser positiva.
   - `Parte real de ZL`: resistencia de la carga (ej. `60`).
   - `Parte imaginaria de ZL`: reactancia de la carga (ej. `30` para `+j30`, `-25` para `-j25`).
   - `ℓ`: lista opcional de desplazamientos normalizados en múltiplos de `λ`, separados por comas. Valores positivos representan desplazamientos hacia el generador; negativos, hacia la carga. Deja vacío si no deseas evaluar puntos adicionales.
3. Observa la salida:
   - Consola: reporte paso a paso con fórmulas, resultados normalizados y desnormalizados, y resumen por cada desplazamiento ingresado.
   - Ventana principal: carta de Smith, círculo de magnitud constante, marcadores de ángulos y regletas inferiores resaltando el resultado.
   - Ventana secundaria: transcripción formateada del procedimiento y una guía de las escalas auxiliares.
   - Interacción: al pasar el mouse por el punto `Γ_L`, los marcadores o las regletas se muestran simultáneamente `Γ`, `τ`, `Z_in`, `R_in`, `X_in`, ROE, etc.

## Fundamentos matemáticos

- **Impedancia normalizada**: \( z_N = \frac{Z_L}{Z_0} = r + jx \).
- **Coeficiente de reflexión**: \( \Gamma_L = \frac{Z_L - Z_0}{Z_L + Z_0} \). El script almacena `Γ_L`, `ZL − Z0` y `ZL + Z0` para reportar tanto el valor normalizado como el desnormalizado.
- **Relación de onda estacionaria (ROE/SWR)**: \( \text{ROE} = \frac{1 + |\Gamma_L|}{1 - |\Gamma_L|} \). A partir de \(|\Gamma_L|\) se calculan pérdidas de retorno, pérdidas por desajuste, atenuación equivalente y coeficiente de pérdida por ROE.
- **Coeficiente de transmisión de tensión**: \( \tau = 1 + \Gamma_L \). Se registran magnitud, ángulo y el coeficiente de transmisión de potencia \( T_P = 1 - |\Gamma_L|^2 \).
- **Impedancia vista tras un desplazamiento**: para \( \ell \) en múltiplos de \( \lambda \), \( \Gamma(\ell) = \Gamma_L e^{-j4\pi \ell} \) y \( Z_{in}(\ell) = Z_0 \frac{1 + \Gamma(\ell)}{1 - \Gamma(\ell)} \). El reporte lista `Z_in`, `R_in` y `X_in` cuando el valor es finito.

## Flujo interno del script

1. `leer_parametros_usuario` guía la captura de `Z0`, `R_L`, `X_L` y la lista de desplazamientos, validando los datos numéricos.
2. `calcular_reflexion_y_parametros` aplica las fórmulas anteriores y devuelve un diccionario con todos los parámetros necesarios para la visualización y el reporte.
3. Si se proporcionan desplazamientos, `_calcular_perfiles_desplazamiento` genera las rotaciones de `Γ`, `τ` y las impedancias vistas.
4. `imprimir_procedimiento` construye el resumen textual y lo imprime en consola.
5. `crear_grafica_completa` arma la ventana principal (carta, escalas, regletas) y la figura auxiliar, enlazando los manejadores de eventos para el hover.

## Ejemplos de ejecución

### Carga inductiva suave

Entradas proporcionadas:

```text
Z0 = 50
Parte real de ZL = 60
Parte imaginaria de ZL = 30
ℓ = 0.10, -0.05
```

Extracto del reporte:

```text
=== Procedimiento paso a paso ===
1) Datos de entrada:
   Z0 = 50.00 Ω
   ZL = 60.00 + j30.00 Ω
2) Impedancia normalizada z_N = ZL / Z0:
   z_N = 1.20 + j0.60 (adimensional)
3) Coeficiente de reflexión en la carga:
   Numerador (ZL − Z0) = 10.00 + j30.00 Ω
   Denominador (ZL + Z0) = 110.00 + j30.00 Ω
   Γ_L = (ZL − Z0) / (ZL + Z0) = 0.16 + j0.09
...
   ℓ = +0.10 λ (hacia el generador), rotación = -72.00 °
     Γ(ℓ) = 0.08 + j0.18; ∠Γ(ℓ) = 66.59 °
     Z_in(ℓ) = 59.87 + j44.62 Ω (impedancia vista)
     R_in(ℓ) = 59.87 Ω, X_in(ℓ) = 44.62 Ω
```

La figura principal coloca marcadores púrpura sobre los desplazamientos y resalta los valores correspondientes en las regletas.

### Carga capacitiva desadaptada

Entradas proporcionadas:

```text
Z0 = 75
Parte real de ZL = 20
Parte imaginaria de ZL = -40
ℓ = (vacío)
```

Resultados clave:

- `Γ_L = -0.45 - j0.26`, `|Γ_L| = 0.52`, `∠Γ_L = -150.5 °`.
- `ZL − Z0 = -55.00 - j40.00 Ω`, `ZL + Z0 = 95.00 - j40.00 Ω`.
- `ROE = 3.19`, `RL = 5.70 dB`, `P_trans = 0.73`.

La carta muestra el punto de reflexión, mientras que las regletas indican el valor de ROE y las pérdidas asociadas.

## Limitaciones actuales

- Las entradas de `ZL` deben proporcionarse como partes real e imaginaria separadas. No se admiten expresiones en forma rectangular (`a±jb`) ni polar (`m<θ`).
- El script asume líneas sin pérdidas; los cálculos usan parámetros ideales basados únicamente en `Γ`.
- Se espera que `Z0` sea estrictamente positiva; no hay validación adicional sobre magnitudes máximas.

## Estructura del repositorio

- `smithchart.py`: script principal que contiene cálculos, dibujo e interactividad.
- `README.md`: este documento.
- `docs/`: materiales complementarios (si están presentes).
- `__pycache__/`: artefactos generados por Python.

## Créditos y uso

Proyecto desarrollado por **David González Herrera (Carné 19221022)** para el **Laboratorio Integrador 2025-2** del curso **LTT93**, con asesoría del docente **Willer Ferney Montes Granada**. Uso con fines académicos y demostrativos.
