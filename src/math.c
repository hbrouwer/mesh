/*
 * math.c
 *
 * Copyright 2012, 2013 Harm Brouwer <me@hbrouwer.eu>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "main.h"
#include "math.h"

#include <math.h>

double minimum(double x, double y)
{
        if (x <= y)
                return x;
        else
                return y;
}

double maximum(double x, double y)
{
        if (x >= y)
                return x;
        else
                return y;
}

double sign(double x)
{
        if (x == 0.0)
                return 0.0;
        else if (x == fabs(x))
                return 1.0;
        else
                return -1.0;
}

/*
 * Box-Muller transform for the generation of pairs of normally distributed
 * random numbers. See:
 *
 * Box, G. E. P. and Muller, M. E. (1958). A note on the generation of
 *     random normal deviates. The Annals of Mathematical Statistics, 29 
 *     (2), 610-611.
 */

double normrand(double mu, double sigma)
{
        double rs1;
        static double rs2 = 0.0;

        if (rs2 != 0.0) {
                rs1 = rs2;
                rs2 = 0.0;
        } else {
                double x, y, r;
                do {
                        x = 2.0 * rand() / RAND_MAX - 1;
                        y = 2.0 * rand() / RAND_MAX - 1;
                        r = (x * x) + (y * y);
                } while (r == 0 || r > 1.0);
                rs1 = x * sqrt(-2.0 * log(r) / r);
                rs2 = y * sqrt(-2.0 * log(r) / r);
        }

        return rs1 * sigma + mu;
}

/*
 * Inner product:
 *
 * ip = sum_i (x_i * y_i)
 */

double inner_product(struct vector *v1, struct vector *v2)
{
        double ip = 0.0;

        for (int i = 0; i < v1->size; i++)
                ip += v1->elements[i] * v2->elements[i];

        return ip;
}

/*
 * Cosine similarity:
 *
 *                  sum_i (x_i * y_i)
 * cs = ---------------------------------------------
 *      sqrt(sum_i (x_i ^ 2)) * sqrt(sum_i (y_i ^ 2))
 */

double cosine_similarity(struct vector *v1, struct vector *v2)
{
        double nom = 0.0, xsq = 0.0, ysq = 0.0;

        for (int i = 0; i < v1->size; i++) {
                nom += v1->elements[i] * v2->elements[i];
                xsq += pow(v1->elements[i], 2.0);
                ysq += pow(v2->elements[i], 2.0);
        }

        xsq = sqrt(xsq);
        ysq = sqrt(ysq);

        return nom / (xsq * ysq);
}

/*
 * Pearson's correlation:
 *
 *                   sum_i ((x_i - x) * (y_i - y))
 * pc = -----------------------------------------------------
 *      sqrt(sum_i (x_i - x) ^ 2) * sqrt(sum_i (y_i - y) ^ 2)
 *
 * where x is the average of vector x and y the average of vector y.
 */

double pearson_correlation(struct vector *v1, struct vector *v2)
{
        double xmn = 0.0, ymn = 0.0;

        for (int i = 0; i < v1->size; i++) {
                xmn += v1->elements[i];
                ymn += v2->elements[i];
        }

        xmn /= v1->size;
        ymn /= v2->size;

        double nom = 0.0, xsq = 0.0, ysq = 0.0;

        for (int i = 0; i < v1->size; i++) {
                nom += (v1->elements[i] - xmn) * (v2->elements[i] - ymn);
                xsq += pow(v1->elements[i] - xmn, 2.0);
                ysq += pow(v2->elements[i] - ymn, 2.0);
        }

        xsq = sqrt(xsq);
        ysq = sqrt(ysq);

        return nom / (xsq * ysq);
}
