/*
 * math.c
 *
 * Copyright 2012 Harm Brouwer <me@hbrouwer.eu>
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

double square(double x)
{
        return x * x;
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
 * Inverse squared city-block distance:
 *
 *              1
 * -------------------------
 * (sum_i |a_i - b_i|)^2 + 1
 */

double inv_sq_city_block(struct vector *v1, struct vector *v2)
{
        double cb = 0.0;

        for (int i = 0; i < v1->size; i++)
                cb += fabs(v1->elements[i] - v2->elements[i]);
        cb = pow(cb, 2.0);
        cb++;

        return 1.0 / cb;
}

/*
 * Inverse squared euclidean distance:
 *
 *            1
 * -----------------------
 * sum_i (a_i - b_i)^2 + 1
 */

double inv_sq_euclidean(struct vector *v1, struct vector *v2)
{
        double ed = 0.0;

        for (int i = 0; i < v1->size; i++)
                ed += pow(v1->elements[i] - v2->elements[i], 2.0);
        ed++;

        return 1.0 / ed;
}

/*
 * Cosine similarity:
 *
 *                 1
 * ---------------------------------
 * (sum a_i^2)^0.5 * (sum b_i^2)^0.5
 */

double cosine_similarity(struct vector *v1, struct vector *v2)
{
        double nom = 0.0, asq = 0.0, bsq = 0.0;

        for (int i = 0; i < v1->size; i++) {
                nom += v1->elements[i] * v2->elements[i];
                asq += pow(v1->elements[i], 2.0);
                bsq += pow(v2->elements[i], 2.0);
        }

        return nom / (pow(asq, 0.5) * pow(bsq, 0.5));
}

/*
 * Correlation:
 *
 *        sum (a_i - a) * (b_i - b)
 * ---------------------------------------
 * (sum (a_i - a)^2 * sum (b_i - b)^2)^0.5
 */

double correlation(struct vector *v1, struct vector *v2)
{
        double amn = 0.0, bmn = 0.0;

        for (int i = 0; i < v1->size; i++) {
                amn += v1->elements[i];
                bmn += v2->elements[i];
        }

        amn /= v1->size;
        bmn /= v2->size;

        double nom = 0.0, asq = 0.0, bsq = 0.0;
        for (int i = 0; i < v1->size; i++) {
                nom += (v1->elements[i] - amn) * (v2->elements[i] - bmn);
                asq += pow(v1->elements[i] - amn, 2.0);
                bsq += pow(v2->elements[i] - bmn, 2.0);
        }

        double denom = pow(asq * bsq, 0.5);

        return nom / denom;
}
