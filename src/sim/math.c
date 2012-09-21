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

/*
 * Box-Muller transform for the generation of paris of normally distributed
 * random numbers. See:
 *
 * Box, G. E. P. and Muller, M. E. (1958). A note on the generation of
 *   random normal deviates. The Annals of Mathematical Statistics, 29 (2),
 *   610-611.
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
