# Carta de Smith Interactiva – LTT93 2025‑2

Generador interactivo en **Python** que construye una **carta de Smith completa con regletas inferiores**, tal como se utiliza en el curso **LTT93 – Laboratorio Integrador 2025‑2**. El programa calcula los parámetros clásicos de líneas de transmisión, permite definir distintos escenarios de carga y muestra un **procedimiento matemático paso a paso** en una hoja adicional.

---

## 1. Objetivo del proyecto

Proveer una herramienta didáctica que permita a estudiantes de **Líneas de Transmisión**:

* Visualizar la **carta de Smith normalizada** con malla completa (círculos de resistencia, arcos de reactancia, escalas de ángulo y longitud de onda).
* Calcular **coeficientes de reflexión, ROE, pérdidas de retorno, pérdidas por desajuste y parámetros de transmisión** para una carga arbitraria.
* Explorar el efecto de **desplazarse a lo largo de una LT** (en múltiplos de λ o en metros) sobre Γ y la impedancia vista `Z_in(ℓ)`.
* Generar **gráficos exportables** y un **informe textual del procedimiento** para documentar el laboratorio.

---

## 2. Características principales (versión 2025‑2)

### 2.1. Entradas flexibles y robustas

* **Impedancia característica `Z0`** y **carga `ZL`** en **forma rectangular**:

  * Acepta formatos como: `50`, `75+j25`, `200-j50`, `-j10`, `j0.8`.
  * Tolera **espacios** y **coma decimal** (`50,5+j25,2`).
  * Soporta `j` o `i` como unidad imaginaria.

* **Frecuencia** con sufijos opcionales:

  * Ejemplos: `915e6`, `2.45GHz`, `60 MHz`, `433m`, `10k`, `5e9`.
  * Internamente se convierte todo a **Hz**.

* **Factor de velocidad VNP** (opcional):

  * Se ingresa como **porcentaje** (`75`, `80%`, `0.8`).
  * Se usa para calcular la velocidad de propagación `v_p` y la longitud de onda en la línea `λ`.

### 2.2. Modos de carga soportados

El script permite seleccionar el tipo de carga / escenario:

1. **Carga general `ZL` compleja** (modo 1).
2. **Circuito abierto ideal**: `ZL = ∞`, `Γ_L = +1` (modo 2).
3. **Cortocircuito ideal**: `ZL = 0`, `Γ_L = −1` (modo 3).
4. **Carga resistiva pura definida por ROE (SWR)** (modo 4):

   * Se ingresa solo la **ROE** (`S > 1`).
   * El programa calcula las dos soluciones clásicas:

     * `ZL1 = Z0·S` (carga mayor que Z0, en fase).
     * `ZL2 = Z0/S` (carga menor que Z0, en contrafase).
   * Para la carta de Smith se toma por defecto `ZL1` (solución en fase) como impedancia de carga.

### 2.3. Longitudes eléctricas y distancias físicas

El usuario puede especificar:

* **Longitudes normalizadas** `ℓ` en múltiplos de λ:

  * Ejemplo: `0.25, 0.5, -0.1`.
  * Valores **positivos**: desplazamiento **hacia el generador**.
  * Valores **negativos**: desplazamiento **hacia la carga**.

* **Longitud física total de la línea** `L` en metros (opcional).

* **Distancias físicas** `d` (en metros) donde se desea evaluar Γ y `Z_in(ℓ)`.

Si se proporciona la frecuencia (y por tanto se puede calcular `λ`), el script convertirà **todas las distancias físicas** a **longitudes normalizadas** y las mostrará en el informe tanto en λ como en metros.

### 2.4. Gestión automática de remanentes (medias longitudes de onda)

Para cada desplazamiento `ℓ` ingresado, la función `_calcular_perfiles_desplazamiento`:

* Elimina automáticamente **múltiplos de media longitud de onda** (`0.5 λ`), ya que en una línea sin pérdidas:

  * Un desplazamiento de `0.5 λ` rota Γ **180°** y deja la misma ROE.
* Registra cuántas **medias longitudes** se han restado y cuál es el **remanente eléctrico**:

  * `ℓ_original`, `ℓ_equivalente`, número de medias longitudes y `Rem_LE`.
* Reporta también la longitud equivalente **en metros** cuando se conoce `λ`.

Esto facilita el análisis práctico de secciones repetitivas de línea y su efecto sobre el coeficiente de reflexión.

### 2.5. Parámetros calculados

A partir de `Z0` y `ZL`, la función `calcular_reflexion_y_parametros` calcula:

* **Impedancia normalizada**:

  * ( z_N = \dfrac{Z_L}{Z_0} = r + jx ).

* **Coeficiente de reflexión en la carga**:

  * ( \Gamma_L = \dfrac{Z_L - Z_0}{Z_L + Z_0} ) (caso general).
  * Módulo `|Γ_L|` y **ángulo** `∠Γ_L` en grados.
  * Numerador `(ZL − Z0)` y denominador `(ZL + Z0)` se almacenan y se muestran en el informe.

* **ROE (SWR)**:

  * ( \text{ROE} = \dfrac{1 + |\Gamma_L|}{1 - |\Gamma_L|} ).
  * Si `|Γ_L| → 1`, el script marca `ROE → ∞`.

* **Pérdidas y coeficientes asociados**:

  * **Return Loss** positivo: ( RL = -20 \log_{10}(|\Gamma_L|) ).
  * **Coef. de reflexión de tensión**: `|Γ|` y `20·log10(|Γ|)`.
  * **Coef. de reflexión de potencia**: `|Γ|² = Γ_P`.
  * **Potencia transmitida normalizada**: `P_trans = 1 − |Γ|²`.
  * **Pérdida por desajuste** (mismatch loss): ( L_{mis} = -10 \log_{10}(1 - |\Gamma|^2) ).
  * **Pérdida de retorno con signo negativo** `α_RL = −RL`.
  * **Pérdida de desacople** `α_des = −L_mis`.
  * **Atenuación equivalente asociada a |Γ|**: `ATTEN_dB = −20·log10(|Γ|)`.
  * **Coeficiente de pérdida por ROE** clásico `F`, a partir de `S`.

* **Porcentajes útiles**:

  * `%Pr = |Γ|² · 100` → porcentaje de **potencia reflejada**.
  * `%PL = (1 − |Γ|²) · 100` → porcentaje de **potencia absorbida** (eficiencia de acople).

* **Coeficiente de transmisión de tensión**:

  * ( \tau_L = 1 + \Gamma_L ) y su ángulo `∠τ_L`.
  * **Coef. de transmisión de potencia**: ( T_P = 1 - |\Gamma_L|^2 ).

* **Impedancia vista en la carga**:

  * ( Z_{in}(0) = Z_0 \dfrac{1 + \Gamma_L}{1 - \Gamma_L} ).

* **Longitud total de la línea en λ** (si se proporcionan `L` y `λ`):

  * `L_total_lambda = L/λ`.
  * Número aproximado de **campanas VROE** y remanente eléctrico `Rem_LE`.

### 2.6. Perfiles a lo largo de la línea

Para cada longitud normalizada `ℓ` (o distancia `d` transformada a λ), el script calcula:

* ( \Gamma(\ell) = \Gamma_L e^{-j4\pi \ell} ).
* ( Z_{in}(\ell) = Z_0 \dfrac{1 + \Gamma(\ell)}{1 - \Gamma(\ell)} ).

El informe detalla para cada punto:

* `ℓ_original`, `ℓ_equivalente` (tras eliminar medias longitudes), dirección (hacia carga/generador).
* `Γ(ℓ)` en forma rectangular, `∠Γ(ℓ)`.
* `τ(ℓ)`, `∠τ(ℓ)`.
* `ZL′(ℓ) = Z_in(ℓ)` y sus componentes `R′(ℓ)`, `X′(ℓ)`.
* Distancias equivalentes en metros cuando hay frecuencia y VNP.

---

## 3. Visualización gráfica

### 3.1. Carta de Smith completa

La función `dibujar_carta_smith(ax)` genera:

* **Círculo exterior** `|Γ| = 1`.
* **Círculos de resistencia constante** (r = 0.1, 0.2, 0.3, 0.5, 1, 2, 5, 10).
* **Arcos de reactancia constante** (x = ±0.1, ±0.2, ±0.3, ±0.5, ±1, ±2, ±5, ±10).
* Ejes real e imaginario de `Γ`.
* Marcadores de **impedancia** y **admitancia** normalizadas sobre el eje real.

Además se dibujan:

* **Escala de ángulos** de reflexión y transmisión en los anillos externos.
* **Escala de longitudes de onda** hacia el generador y hacia la carga.
* Leyendas laterales para reactancias/susceptancias inductivas y capacitivas.

### 3.2. Regletas inferiores

La función `dibujar_regletas(ax, resultados)` construye un bloque inferior con varias **regletas horizontales** que representan:

1. **ROE (SWR)**.
2. **Pérdida de retorno (dB)**.
3. **Coeficiente de reflexión de potencia |Γ|²**.
4. **Coeficiente de reflexión |Γ|**.
5. **Atenuación (dB)** equivalente.
6. **Coeficiente de pérdida por ROE** `F`.
7. **Pérdida por desajuste (dB)**.
8. **ROE en dB (dBS)**.

En todas las regletas se marca:

* La **posición de |Γ|** en rojo.
* El **valor numérico** asociado al parámetro de esa regleta (por ejemplo ROE calculada, RL, L_mis, etc.).

### 3.3. Interactividad (hover)

En la figura principal:

* El marcador de `Γ_L` y los puntos correspondientes a las longitudes `ℓ` tienen **hover interactivo**.
* Al pasar el ratón cerca de un marcador:

  * Se muestra un cuadro flotante con los datos relevantes (Γ, z_N, ROE, RL, ℓ, ZL′, etc.).
  * El cuadro se **reposiciona** automáticamente para evitar salirse de los bordes de la figura.
* En las regletas, al pasar por la posición de `|Γ|` también aparece un cuadro con `|Γ|` y ROE.

### 3.4. Exportación de la carta sola

La función `exportar_carta_smith_sola(...)` permite guardar una figura **solo con la carta de Smith** (sin regletas):

* Formatos soportados: `SVG`, `PDF` o `PNG` (alta resolución, DPI configurable).
* La carta incluye:

  * El punto `Γ_L` marcado.
  * El círculo de `|Γ|` constante.
  * Cuadros de texto con los datos principales de `Γ_L` y de cada longitud `ℓ`.

El propio programa pregunta si se desea guardar esta figura extra al final de la ejecución.

---

## 4. Procedimiento paso a paso

La función `imprimir_procedimiento(Z0, ZL, resultados)` genera un informe textual que incluye:

1. **Datos de entrada**:

   * `Z0`, `ZL` (o tipo de carga: abierto, corto, ROE resistiva).
   * Frecuencia `f`, velocidad de propagación `v_p`, factor de velocidad `VNP`, longitud de onda `λ`.
   * Longitud total de la línea y distancias físicas (si se suministran).

2. **Impedancia normalizada** `z_N`.

3. **Cálculo de Γ_L** con numerador y denominador explícitos.

4. **Parámetros derivados**: ROE, dBS, `|Γ|`, `|Γ|²`, RL, pérdidas, coeficientes y porcentajes.

5. **Parámetros de transmisión**: `P_trans`, `T_P`, `|τ|` y ángulo de `τ`.

6. **Impedancia vista en la carga** `Z_in(0)`.

7. **Desplazamientos a lo largo de la línea** `ZL′(ℓ)`, con:

   * Longitudes equivalentes en λ.
   * Ángulos de Γ y τ.
   * Impedancias resultantes `ZL′(ℓ)`.
   * Distancias físicas equivalentes en metros.

Este mismo texto se replica en una **figura auxiliar** para facilitar su captura en PDF / imagen.

---

## 5. Flujo interno del script

1. `main()`

   * Llama a `leer_parametros_usuario()` para solicitar:

     * `Z0`, tipo de carga, `ZL` (o ROE), longitudes normalizadas, frecuencia, VNP, longitud física y distancias físicas.
   * Construye el diccionario `parametros_linea` y la lista `perfiles_config`.

2. `calcular_reflexion_y_parametros(Z0, ZL, datos_linea)`

   * Determina `z_N`, `Γ_L`, `|Γ|`, `∠Γ`, ROE, pérdidas, porcentajes, etc.
   * Añade información de línea (`f`, `λ`, `v_p`, `L_total`, campanas VROE, etc.) cuando está disponible.

3. `_calcular_perfiles_desplazamiento(gamma_L, desplazamientos, Z0, lambda_m)`

   * Genera para cada `ℓ` los valores de `Γ(ℓ)`, `τ(ℓ)` y `Z_in(ℓ)`.
   * Documenta remanentes y equivalencias físicas.

4. `imprimir_procedimiento(Z0, ZL, resultados)`

   * Construye el informe textual y lo imprime en consola.

5. `crear_grafica_completa(Z0, ZL, perfiles_config, parametros_linea)`

   * Dibuja la carta de Smith, las regletas, los marcadores y configura los manejadores de eventos para el hover.
   * Crea la figura auxiliar con el procedimiento paso a paso.
   * Ofrece la opción de exportar la carta sola.

---

## 6. Requisitos de instalación

* **Python** 3.10 o superior.
* Paquetes:

  * `numpy`
  * `matplotlib`

Instalación rápida (ejemplo en PowerShell para Windows):

```powershell
python -m pip install numpy matplotlib
```

---

## 7. Ejecución básica

1. Abre una terminal en la carpeta donde está el archivo, por ejemplo `smithchart.py`.
2. Ejecuta:

```bash
python smithchart.py
```

3. Responde a las preguntas en consola:

   * `Z0` (impedancia característica).
   * Tipo de carga (1–4) y `ZL` o ROE según el modo.
   * Lista de longitudes normalizadas `ℓ` (puede quedar vacía).
   * Opcionalmente, frecuencia, VNP, longitud total y distancias físicas.

4. Se abrirán dos ventanas de Matplotlib:

   * La carta de Smith con regletas e interactividad.
   * El informe del procedimiento paso a paso.

5. Cierra las ventanas gráficas para finalizar la ejecución.

---

## 8. Limitaciones actuales

* Se asume **línea sin pérdidas** (solo parámetros reactivos/impedancias normalizadas).
* Las conversiones físicas emplean un modelo simple: `v_p = VNP · c`.
* No se incluyen aún efectos de atenuación distribuida, conductancia o dispersión.

---

## 9. Créditos

Proyecto desarrollado por **David González Herrera (Carné 19221022)** para el curso **LTT93 – Laboratorio Integrador 2025‑2**, como herramienta de apoyo para el análisis de líneas de transmisión y uso avanzado de la carta de Smith completa.
