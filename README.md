# Carta de Smith Interactiva

Herramienta interactiva escrita en Python para generar la carta de Smith completa utilizada en el curso LTT93 (Laboratorio Integrador 2025-2). El script calcula los parámetros clásicos de líneas de transmisión, dibuja la carta normalizada con Matplotlib y muestra las regletas radiales típicas de las cartas comerciales.

## Funcionalidades principales

- **Entrada flexible de impedancias**: acepta `Z0` y `ZL` en forma real, rectangular (`201.3-j50`) o polar (`120<45`, `75∠-30`, `0.5<2rad`).
- **Cálculo automático** en `calcular_reflexion_y_parametros`:
  - Impedancia normalizada `z_N`.
  - Coeficiente de reflexión `Γ`, magnitud, fase y coeficientes asociados (`Gamma_E`, `Gamma_P`).
  - ROE directa y en dB, pérdidas de retorno, pérdidas por desajuste y factor de pérdida por ROE.
  - Coeficientes de transmisión de potencia y tensión (`T_P`, `τ`).
- **Perfiles a lo largo de la línea**: opcionalmente la función `_calcular_perfiles_desplazamiento` rota `Γ` escribiendo las longitudes desplazadas en la salida.
- **Gráfica principal (`dibujar_carta_smith`)**: incluye la carta normalizada, el círculo de reflexión constante y anotaciones de reactancia/conductancia.
- **Escalas suplementarias**: `_dibujar_escala_angulos`, `_dibujar_escala_longitudes` y `_dibujar_anillos_exteriores` agregan anillos de ángulo, longitud de onda y referencias de susceptancia.
- **Regletas inferiores (`dibujar_regletas`)**: representan parámetros como ROE, pérdidas, atenuación y coeficientes radiales; resaltan el valor calculado en rojo.
- **Interactividad**: el hover sobre la carta o sobre la regleta muestra anotaciones dinámicas con todos los valores calculados.
- **Procedimiento documentado**: `imprimir_procedimiento` genera el paso a paso en consola y en una figura adicional, explicando fórmulas y resultados.

## Requisitos de software

- Python 3.9 o superior.
- Bibliotecas: `matplotlib` y `numpy`.

Instala las dependencias con:

```powershell
py -3 -m pip install matplotlib numpy
```

## Cómo ejecutar el script

1. Desde PowerShell ubícate en el directorio del proyecto y ejecuta:

   ```powershell
   py -3 smithchart.py
   ```

2. **Proporciona los valores solicitados** cuando el programa lo indique:
   - `Z0`: impedancia característica (no puede ser cero). Ejemplos válidos: `50`, `75+j0`, `60<-15`.
   - `ZL`: impedancia de carga. Ejemplos: `201.3-j50`, `120<45`, `0.8<1.2rad`.
   - `ℓ`: lista opcional de desplazamientos en múltiplos de `λ` separados por comas. Usa valores positivos hacia el generador y negativos hacia la carga (ej. `0.05, -0.125`). Deja vacío para omitirlos.

3. **Resultados generados**:
   - En consola: reporte paso a paso de todos los parámetros calculados, incluyendo los desplazamientos ingresados.
   - Figura principal: carta de Smith con el punto `ΓL`, círculo de magnitud constante, anillos auxiliares y regletas radiales resaltando el valor obtenido.
   - Figura secundaria: texto con el procedimiento y una leyenda que explica las regletas.
   - Interacción: al pasar el mouse sobre el punto `ΓL` o sobre las regletas se muestran los mismos resultados en un cuadro emergente.

## Descripción del flujo interno

1. `leer_parametros_usuario` gestiona todas las entradas, validando números complejos y permitiendo formatos rectangulares o polares.
2. `calcular_reflexion_y_parametros` reúne las fórmulas del curso para obtener ROE, pérdidas y coeficientes de transmisión basados en `Γ`.
3. Si se proporcionan desplazamientos, `_calcular_perfiles_desplazamiento` rota `Γ` y `τ` para cada distancia.
4. `imprimir_procedimiento` produce la salida textual y el mensaje usado en la figura adicional.
5. `crear_grafica_completa` arma la ventana principal con la carta (`dibujar_carta_smith`) y las regletas (`dibujar_regletas`), además de conectar los eventos de hover.

## Estructura del repositorio

- `smithchart.py`: script principal con cálculos, dibujo e interacción.
- `README.md`: documentación del proyecto.
- Otros archivos auxiliares creados por el sistema operativo (por ejemplo, `desktop.ini`).

## Notas académicas y licencia

Trabajo elaborado por **David González Herrera (Carné 19221022)** para el **Laboratorio Integrador 2025-2** del curso **LTT93**, bajo la guía del docente **Willer Ferney Montes Granada**. Uso académico y demostrativo.
