# Trabajo Práctico 2: Matrices Insumo-Producto

Este repositorio contiene el desarrollo del Trabajo Práctico N°2 de la materia "Álgebra Lineal Computacional" de la Universidad de Buenos Aires. 

El objetivo principal es continuar con el analisis de las matrices insumo-producto de diferentes sectores económicos mediante herramientas de álgebra lineal y computacional. Se implementan técnicas como el método de la potencia, cálculo de series infinitas y análisis en componentes principales (ACP).

## Contenidos del Repositorio

- `grupo_36_Tp2.ipynb`: Notebook con el desarrollo completo del trabajo práctico.
- `funciones.py`: Archivo con las funciones auxiliares implementadas.
- `enunciado.pdf`: PDF con las consignas del trabajo práctico y algo de contexto teórico.
- `data_paises.csv`: Dataset utilizado para el análisis.

## Requisitos

Las siguientes bibliotecas son necesarias para la ejecución:

- numpy 1.25.2
- pandas 1.5.3
- matplotlib 3.7.1
- seaborn 0.12.2

Dejamos un comando para poder generar un `conda env` para correr la entrega.
```bash
conda create --name correccion_alc_tp1 python=3.11 seaborn=0.12.2 scikit-learn=1.2.2 matplotlib=3.7.1 numpy=1.25.2 pandas=1.5.3
conda activate correccion_alc_tp2
pip install spyder-kernels==2.5.0
```
Asimismo, también el comando para su remoción.
```bash
conda deactivate correccion_alc_tp2
conda remove --name correccion_alc_tp2 --all
```
