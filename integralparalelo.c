#include <stdio.h>
#include <math.h>
#include "mpi.h"

#define f(x) ((x) / (pow((x), 4) + 1)) // Función a integrar

#define a 1.0 // Límite inferior
#define b 3.0 // Límite superior
#define n 8   // Número de subintervalos

#define h ((b - a) / n) // Longitud de cada subintervalo

double calcular_integral(double local_a, double local_b, int local_n, double local_h) {
    double local_sum = 0.0;
    double x;

    // Calcula la suma local de la integral usando la regla del trapecio
    for (int i = 0; i <= local_n; i++) {
        x = local_a + i * local_h;
        if (i == 0 || i == local_n) {
            local_sum += f(x); // Añade f(a) y f(b)
        } else {
            local_sum += 2 * f(x); // Suma f(x_i) para i = 1 a n-1
        }
    }
    local_sum *= local_h / 2.0; // Multiplica por h/2 para obtener la aproximación de la integral local

    return local_sum;
}

void imprimir(double local_a, double local_b, int local_n, double local_h, int rank) {
    double x;
    printf("Proceso %d:\n", rank);
    for (int i = 0; i <= local_n; i++) {
        x = local_a + i * local_h;
        printf("x(%d) = %f\n", i, x);
        printf("f(x(%d)) = %f\n", i, f(x));
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    double total_integral, local_integral;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    start_time = MPI_Wtime(); // Tiempo inicial de ejecución

    double local_a = a + rank * (b - a) / size;       // Límite inferior local para este proceso
    double local_b = a + (rank + 1) * (b - a) / size; // Límite superior local para este proceso
    int local_n = n / size;                          // Número de subintervalos locales por proceso
    double local_h = (local_b - local_a) / local_n;   // Longitud de cada subintervalo local

    imprimir(local_a, local_b, local_n, local_h, rank); // Imprimir resultados locales

    local_integral = calcular_integral(local_a, local_b, local_n, local_h);

    MPI_Reduce(&local_integral, &total_integral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("La integral x/(x^4 + 1) que va desde %f hasta %f tiene una aproximación de %f\n", a, b, total_integral);
    }

    end_time = MPI_Wtime(); // Tiempo final de ejecución

    if (rank == 0) {
        printf("Tiempo de ejecución: %f segundos\n", end_time - start_time);
    }

    MPI_Finalize();

    return 0;
}
