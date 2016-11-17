# Parcial II

1. Construir una aplicación que realice la multiplicación de matrices de gran tamaño
utilizando MPI​ y CUDA.​
2. La aplicación deberá distribuir el trabajo a realizarse en 4 nodos máximo.
3. Para cada uno de los nodos envueltos en el procesamiento se deberá realizar el cálculo
necesario sobre la GPU​.
4. El tamaño de las matrices a multiplicar será definido por cada uno de los estudiantes y
será tenido en cuenta como un 10%​ de ésta nota.
5. Se tendrá que hacer un análisis de tiempo que muestre cómo el algoritmo desarrollado
se comporta al correr usando 1 solo nodo, usando 2, usando 3 y finalmente 4. (10%​)
6. Se deberá comparar la solución obtenida con una donde se usen solamente CPU’s.
Esta comparativa deberá también incluir tiempos de ejecución y gráficas de aceleración.
(10%​). De nuevo hacer el análisis para cada uno de los nodos.
7. Será necesario para el desarrollo del trabajo que los estudiantes investiguen sobre el
proceso de compilación de aplicaciones que usan OPENMPI​ +​ CUDA.​
8. El correcto funcionamiento de la aplicación será evaluado, esto significa incluir una
rutina de comparación de resultados que deberá ser ejecutada sólo cuando el profesor
lo indique y con ánimo de verificar que la multiplicación es correcta. (10%​)
9. La calidad y claridad del reporte que se presentará utilizando el markdown del
repositorio tendrá un peso de 10%
10. La entrega se hará de forma individual y tendrá un peso en la evaluación del 50%. ​Esto
indica la claridad en la explicación, el conocimiento de los comandos básicos para
ejecutar programas a través de slurm y la comprensión en los procesos de compilaciónConstruir una aplicación que realice la multiplicación de matrices de gran tamaño
utilizando MPI​ y CUDA.​

## Comandos
``` bash
/usr/local/cuda/bin/nvcc matmult.cu -c matmult.o
mpic++ -c matmult_MPI.cpp -o matmult_MPI.o
mpic++ matmult.o matmult_MPI.o -o matmult -L/usr/local/cuda/lib64/ -lcudar
```
