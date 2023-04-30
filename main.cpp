#include <iostream>
#include <omp.h>
#include <fstream>
#include <cmath>
#include <string>
#include <stdio.h>
#include <stdlib.h>
using namespace std;



class Determinant
{
public:
    double GLOBAL_DET = 0;
    int COUNT_THREADS;
    Determinant(int count_threads) {
        COUNT_THREADS = count_threads;
    }

    int ChangeElements(double* matrix, unsigned int n, unsigned int k)
    {
        for (unsigned int i = k + 1; i < n; i++)
        {
            if (abs(matrix[i * n + i - 1]) >= 0.00001f)
            {
                for (unsigned int j = 0; j < n; j++)
                {
                    swap(matrix[k * n + j], matrix[i * n + j]);
                    matrix[i * n + j] *= -1;
                }
                return 1;
            }
        }

        return 0;
    }

    double Single(double* matrix, int n)
    {
        if (n == 1)
        {
            return matrix[0];
        }
        double det = 1.0;
        for (unsigned int i = 0; i < n; i++)
        {
            unsigned int reference = i * n;

            if (abs(matrix[reference + i]) <= 0.00001f)
            {

                if (!ChangeElements(matrix, n, i))
                    return 0;
            }
            det *= matrix[i * n + i];
            for (unsigned int j = i + 1; j < n; j++)
            {
                unsigned int current = j * n;
                double factor = matrix[current + i] / matrix[reference + i];

                for (unsigned int p = i; p < n; p++)
                    matrix[current + p] -= matrix[reference + p] * factor;
            }
        }
        return det;
    }

    double Extension(double* matrix, int n)
    {
        if (n == 1)
        {
            return matrix[0];
        }
        double det = 1.0;

        for (unsigned int i = 0; i < n; i++)
        {
            unsigned int reference = i * n;

            if (abs(matrix[reference + i]) <= 0.00001f)
            {

                if (!ChangeElements(matrix, n, i))
                    return 0;
            }

            if (((n - i) * (n - 1 - i) > 7000) && (n - 1 - i >= COUNT_THREADS))
            {
#pragma omp parallel num_threads(COUNT_THREADS)
                {
#pragma omp for schedule(static,100)
                    for (unsigned int j = i + 1; j < n; j++)
                    {
                        unsigned int current = j * n;
                        double factor = matrix[current + i] / matrix[reference + i];

                        for (unsigned int p = i; p < n; p++)
                            matrix[current + p] -= matrix[reference + p] * factor;
                    }
                }
            }
            else
            {
                for (unsigned int j = i + 1; j < n; j++)
                {
                    unsigned int current = j * n;
                    double factor = matrix[current + i] / matrix[reference + i];

                    for (unsigned int p = i; p < n; p++)
                        matrix[current + p] -= matrix[reference + p] * factor;
                }
            }
        }


#pragma omp parallel num_threads(COUNT_THREADS)
        {
            double TDet = 1.0;

#pragma omp for schedule(static,100)
            for (int i = 0; i < n; i++)
                TDet *= matrix[i * n + i];

#pragma omp atomic
            det *= TDet;
        }
        return det;
    }
};

int main(int argc, char* argv[])
{
    short int count_treads;
    unsigned int n;
    double check_dob;
    if (argc != 4)
    {
        cerr << "error: invalid number of arguments" << endl;
        return 1;
    }
    else
    {
        try
        {
            count_treads = stoi(argv[3]);
            if (count_treads == 0)
                count_treads = omp_get_max_threads();
        }
        catch (const std::exception&)
        {
            cerr << "error: The passed value is not a number" << endl;
            return 1;
        }
    }
    ifstream file(argv[1]);
    if (!file.is_open())
    {
        cerr << "error: Could not open the file" << endl;
        return 1;
    }
    file >> n;
    double* matrix = new(nothrow) double[n * n];
    for (unsigned int i = 0; i < n * n; i++)
    {
        if (file >> check_dob)
        {
            try
            {
                matrix[i] = check_dob;
            }
            catch (const std::exception&)
            {
                cerr << "It is impossible to record due to the lack of allocated memory" << endl;
                return 1;
            }
        }
        else
        {
            cerr << "Enter not numbers" << endl;
            delete[] matrix;
            return 1;
        }
    }
    file.close();
    Determinant det = Determinant(count_treads);
    double ret;
    if (count_treads != (-1))
    {
        auto start = omp_get_wtime();
        ret = det.Extension(matrix, n);
        auto end = omp_get_wtime();
        printf("Time (%i thread(s)): %.7f ms\n", count_treads, (end - start) * 1000);
    }
    else
    {
        auto start = omp_get_wtime();
        ret = det.Single(matrix, n);
        auto end = omp_get_wtime();
        printf("Time (%i thread(s)): %.7f ms\n", 1, (end - start) * 1000);
    }
    ofstream file_out(argv[2]);
    if (!file_out.is_open())
    {
        cerr << "error: Could not open the file out" << endl;
        delete[] matrix;
        return 1;
    }
    file_out << ret << '\n';
    file_out.close();

    delete[] matrix;
    return 0;
}