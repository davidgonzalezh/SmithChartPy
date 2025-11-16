# Carta de Smith Interactiva

Generador interactivo de carta de Smith completo para análisis de líneas de transmisión. El proyecto calcula parámetros clásicos de RF, dibuja la carta con Matplotlib y añade escalas y regletas auxiliares para facilitar la interpretación.

## Características

- Cálculo detallado del coeficiente de reflexión, ROE, pérdidas y coeficientes de transmisión.
- Visualización de la carta de Smith con anillos adicionales, escalas de ángulos y longitudes de onda.
- Regletas inferiores que muestran parámetros radiales comunes en cartas comerciales.
- Hover interactivo con anotaciones dinámicas.
- Personalización para el curso LTT93 (Laboratorio Integrador 2025-2).

## Requisitos

- Python 3.9 o superior.
- Bibliotecas: `matplotlib`, `numpy`.

### Instalación de dependencias

```powershell
py -3 -m pip install matplotlib numpy
```

## Uso

1. Ejecuta el script principal:
   ```powershell
   py -3 smithchart.py
   ```
2. Introduce los valores solicitados para `Z0`, `ZL` y los desplazamientos.
3. Analiza la salida en consola y la gráfica generada.

## Estructura del proyecto

- `smithchart.py`: script principal con toda la lógica de cálculo y visualización.
- `README.md`: este archivo de documentación.
- `.gitignore`: exclusiones sugeridas para el repositorio.

## Licencia

Proyecto académico de David González Herrera para el curso LTT93, Docente Willer Ferney Montes Granada.
