/*
 * random.c
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

#include <math.h>
#include <stdint.h>

#include "math.h"
#include "random.h"

/**************************************************************************
 * Randomizes the values of a matrix using samples from a Gaussian normal
 * distribution N(mu,sigma).
 *************************************************************************/
void randomize_gaussian(struct matrix *m, struct network *n)
{
        for (uint32_t i = 0; i < m->rows; i++)
                for (uint32_t j = 0; j < m->cols; j++)
                        m->elements[i][j] = normrand(n->random_mu, n->random_sigma);
}

/************************************************************************** 
 * Randomizes a matrix with uniformly sampled values from a given range.
 *************************************************************************/
void randomize_range(struct matrix *m, struct network *n)
{
        for (uint32_t i = 0; i < m->rows; i++)
                for (uint32_t j = 0; j < m->cols; j++)
                        m->elements[i][j] = (double)rand() / RAND_MAX
                                * (n->random_max - n->random_min)
                                + n->random_min;
}

/**************************************************************************
 * Randomize a matrix using Nguyen-Widrow (NW; Nguyen & Widrow, 1990)
 * randomization. In NW randomization, all weights are first randomized to
 * values within a range [min,max]. Next, the Euclidean norm of the weight
 * matrix is computed:
 *
 *     en = sqrt(sum_i (w_ij ^ 2))
 *
 * as well as a beta value:
 *
 *     beta = 0.7 * h ^ (1 / i)
 *
 * where h is the number of neurons in the group that is being projected to,
 * and i the number of units in the projecting group. Based on this beta
 * value and the Euclidean norm, each weight is then adjusted to:
 *
 *     w_ij = (beta * w_ij) / en
 *
 * References
 *
 * Nguyen, D. & Widrow, B. (1990). Improving the learning speed of 2-layer
 *     neural networks by choosing initial values of adaptive weights.
 *     Proceedings of the International Joint Conference on Neural Networks
 *     (IJCNN), 3:21-26, June 1990.
 *************************************************************************/
void randomize_nguyen_widrow(struct matrix *m, struct network *n)
{
        randomize_range(m, n);
        
        /* 
         * Compute Euclidean norm:
         *
         * en = sqrt(sum_i (w_ij ^ 2))
         */
        double en = 0.0;
        for (uint32_t i = 0; i < m->rows; i++)
                for (uint32_t j = 0; j < m->cols; j++)
                        en += pow(m->elements[i][j], 2.0);
        en = sqrt(en);

        /* 
         * Compute beta value:
         * 
         * beta = 0.7 * h ^ (1 / i)
         */
        double beta = 0.7 * pow(m->cols, 1.0 / m->rows);

        /*
         * Compute weights:
         *
         * w_ij = (beta * w_ij) / en
         */
        for (uint32_t i = 0; i < m->rows; i++)
                for (uint32_t j = 0; j < m->cols; j++)
                        m->elements[i][j] = (beta * m->elements[i][j]) / en;
}

/**************************************************************************
 * Randomize a matrix using Fan-In (FI) randomization. In FI randomization,
 * each weight is defined as:
 *
 *     w_ij = (min / h) + R * ((max - min) / h)
 *
 * where h is the number of units in the group that is projected to and R
 * is a random number in the range [-1,1].
 *************************************************************************/
void randomize_fan_in(struct matrix *m, struct network *n)
{
        /*
         * Randomize weights in the range [-1,1].
         */
        double random_min = n->random_min;
        double random_max = n->random_max;

        n->random_min = -1;
        n->random_max = 1;

        randomize_range(m, n);

        n->random_min = random_min;
        n->random_max = random_max;

        /*
         * Compute weights:
         *
         * w_ij = (min / h) + w_ij * ((max - min) / h)
         *
         * where h is the number of units in the group
         * that is projected to.
         */
        for (uint32_t i = 0; i < m->rows; i++)
                for (uint32_t j = 0; j < m->cols; j++)
                        m->elements[i][j] = n->random_min / m->cols
                                + m->elements[i][j]
                                * ((n->random_max - n->random_min) / m->cols);
}

/**************************************************************************
 * Randomizes a matrix with binary values.
 *************************************************************************/
void randomize_binary(struct matrix *m, struct network *n)
{
        for (uint32_t i = 0; i < m->rows; i++)
                for (uint32_t j = 0; j < m->cols; j++)
                        m->elements[i][j] = round((double)rand() / RAND_MAX);
}
