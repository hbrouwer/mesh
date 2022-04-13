/*
 * Copyright 2012-2022 Harm Brouwer <me@hbrouwer.eu>
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

#include "act.h"
#include "main.h"

                /**********************************
                 **** feed forward propagation ****
                 **********************************/

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
This function propagates activation forward from a group g. Let j be a unit
in one of the network's groups, and i a unit in a group projecting to it.
The net input x_j to unit j is defined as:

        x_j = sum_i (y_i * w_ij)
 
where y_i is the activation level of unit i in the projecting group, and
w_ij the weight of the "synaptic" connection between unit j and unit i.
Provided the net input x_j, the activation level y_j of unit j is then
defined as:

        y_j = f(x_j)

where f is typically a non-linear activation function. 
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

void feed_forward(struct network *n, struct group *g)
{
        /*
         * Under the assumption that activation levels for the units in the
         * current group have already been determined, determine the
         * activation levels of all the groups towards which the current
         * group maintains a projection.
         */
        for (uint32_t i = 0; i < g->out_projs->num_elements; i++) {
                struct projection *op = g->out_projs->elements[i];
                /*
                 * During BPTT, we want activation to propagate only through
                 * the network of the current timestep.
                 */
                if (op->flags->recurrent)
                        continue;
                
                /*
                 * Compute net input for the units in each group that the
                 * current group projects to.
                 */
                struct group *rg = op->to;
#ifdef _OPENMP
#pragma omp parallel for if (n->flags->omp_mthreaded)
#endif /* _OPENMP */
                for (uint32_t j = 0; j < rg->vector->size; j++) {
                        /* 
                         * Reset the activation level of the current unit.
                         */ 
                        rg->vector->elements[j] = 0.0;

                        /*
                         * Determine the net input to the current unit:
                         *
                         * x_j = sum_i (y_i * w_ij)
                         *
                         * Note: A unit can receive activation from units in
                         * different projecting groups.
                         */
                        for (uint32_t x = 0; x < rg->inc_projs->num_elements; x++) {
                                struct projection *ip = rg->inc_projs->elements[x];
                                struct group *pg = ip->to;
                                struct matrix *w = ip->weights;
                                for (uint32_t z = 0; z < pg->vector->size; z++)
                                        rg->vector->elements[j] += pg->vector->elements[z]
                                                * w->elements[z][j];
                        }

                        /*
                         * Apply an activation function to the net input
                         * (unless the softmax function is used, which
                         * requires all net inputs to be computed first).
                         *
                         * y_j = f(x_j)
                         */
                        if (rg->act_fun->fun != act_fun_softmax)
                                rg->vector->elements[j] = rg->act_fun->fun(rg, j);
                }

                /* apply softmax activation function (if required) */
                if (rg->act_fun->fun == act_fun_softmax)
                        for (uint32_t j = 0; j < rg->vector->size; j++)
                                rg->vector->elements[j] = rg->act_fun->fun(rg, j);
        }

        /* 
         * Recursively repeat the above for all of the groups towards which
         * the current group maintains a projection. Again, we skip
         * recurrent projections, as we want activation to only propagate
         * through the network of the current timestep during BPTT.
         */
        for (uint32_t i = 0; i < g->out_projs->num_elements; i++) {
                struct projection *op = g->out_projs->elements[i];
                if (!op->flags->recurrent)
                        feed_forward(n, op->to);
        }
}

                /******************************
                 **** activation functions ****
                 ******************************/

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Logistic function:

        f(x) = 1 / (1 + e ^ (-(g * x)))

and its derivative:

        f'(x) = g * y * (1 - y)

where g is a gain parameter
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Flat spot correction. 

A small value (e.g., 0.1) is added to the derivative f'(x_j) of the logistic
activation function to avoid that it approaches zero when y_j is near 1.0 or
0.0. See:

Fahlman, S. E. (1988). An empirical study of learning speed in back-
        propagation networks. Technical report CMU-CS-88-162. School of
        Computer Science, Carnegie Mellon University, Pittsburgh, PA 15213.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

double act_fun_logistic(struct group *g, uint32_t i)
{
        return 1.0 / (1.0 + EXP(-(g->pars->logistic_gain
                * g->vector->elements[i])));
}

double act_fun_logistic_deriv(struct group *g, uint32_t i)
{
        return g->pars->logistic_gain
                * g->vector->elements[i] * (1.0 - g->vector->elements[i])
                + g->pars->logistic_fsc;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Bipolar sigmoid function:

        f(x) = (-1) + 2 / (1 / e ^ (-x))

and its derivative:

        f'(x) = 0.5 * (1 + y) * (1 - y)
 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

double act_fun_bipolar_sigmoid(struct group *g, uint32_t i)
{
        return (-1.0) + 2.0 / (1.0 + EXP(-g->vector->elements[i]));
}

double act_fun_bipolar_sigmoid_deriv(struct group *g, uint32_t i)
{
        return 0.5 * (1.0 + g->vector->elements[i])
                * (1.0 - g->vector->elements[i]);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Softmax function:

        f(x) = (e ^ x) / sum_j (e ^ x_j)
 
and its derivative:

        f'(x) = J * y

where J is the Jacobian matrix with:

                 | y_i * (1.0 - y_j)  , if i = j
        J[i,j] = |
                 | -1.0 * y_i * y_j   , if i != j
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

double act_fun_softmax(struct group *g, uint32_t i)
{
        static double sum;
        if (i == 0) {
                sum = 0.0;
                for (uint32_t j = 0; j < g->vector->size; j++)
                        sum += EXP(g->vector->elements[j]);
        }
        return EXP(g->vector->elements[i]) / sum;
}

double act_fun_softmax_deriv(struct group *g, uint32_t i)
{
        static struct matrix *jm;
        double ip = 0.0;
        /* compute Jacobian matrix */
        if (i == 0) {
                struct vector *v = g->vector;
                jm = create_matrix(v->size, v->size);
                for (uint32_t r = 0; r < v->size; r++)
                        for (uint32_t c = 0; c < v->size; c++)
                                if (r == c)
                                        jm->elements[r][c] = v->elements[r]
                                                * (1.0 - v->elements[c]);
                                else
                                        jm->elements[r][c] = -1.0
                                                * v->elements[r]
                                                * v->elements[c];
        }
        /* compute derivative for current unit */
        for (uint32_t j = 0; j < g->vector->size; j++)
                ip += jm->elements[i][j] * g->vector->elements[j];
        /* free matrix */
        if (i == g->vector->size - 1)
                free_matrix(jm);
        return ip;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Hyperbolic tangent function:

        f(x) = (e ^ (2 * x) - 1) / (e ^ (2 * x) + 1)
 
and its derivative:

        f'(x) = 1 - y ^ 2
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

double act_fun_tanh(struct group *g, uint32_t i)
{
        return tanh(g->vector->elements[i]);
}

double act_fun_tanh_deriv(struct group *g, uint32_t i)
{
        return 1.0 - pow(g->vector->elements[i], 2.0);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Linear function:

        f(x) = x
 
and its derivative:
 
        f'(x) = 1
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

double act_fun_linear(struct group *g, uint32_t i)
{
        return g->vector->elements[i];
}

double act_fun_linear_deriv(struct group *g, uint32_t i)
{
        return 1.0;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Rectified Linear Unit (ReLU) function:

        f(x) = max(0,x)

where x may be clipped by a maximum value, and its derivative:

                | 1     iff x > 0
        f'(x) = |
                | 0     otherwise
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

double act_fun_relu(struct group *g, uint32_t i)
{
        return maximum(0.0,
                minimum(g->vector->elements[i], g->pars->relu_max));
}

double act_fun_relu_deriv(struct group *g, uint32_t i)
{
        if (g->vector->elements[i] > 0.0)
                return 1.0;
        else
                return 0.0;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Leaky Rectified Linear Unit (ReLU) function:

                | x             iff x > 0
        f(x) =  |
                | alpha * x     otherwise

where x may be clipped by a maximum value, and its derivative:

                | 1             iff x > 0
        f'(x) = |
                | alpha         otherwise
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

double act_fun_leaky_relu(struct group *g, uint32_t i)
{
        if (g->vector->elements[i] > 0.0)
                return minimum(
                        g->vector->elements[i],
                        g->pars->relu_max);
        else
                return g->pars->relu_alpha * g->vector->elements[i];
}

double act_fun_leaky_relu_deriv(struct group *g, uint32_t i)
{
        if (g->vector->elements[i] > 0.0)
                return 1.0;
        else
                return g->pars->relu_alpha;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Exponential Linear Unit (ELU) function:

                | x                     iff x > 0
        f(x) =  |
                | alpha(e ^ x - 1)      otherwise

where x may be clipped by a maximum value, and its derivative:

                | 1                     iff x > 0
        f'(x) = |
                | y + alpha             otherwise
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

double act_fun_elu(struct group *g, uint32_t i)
{
        if (g->vector->elements[i] > 0.0)
                return minimum(
                        g->vector->elements[i],
                        g->pars->relu_max);
        else
                return g->pars->relu_alpha
                        * (EXP(g->vector->elements[i]) - 1.0);
}

double act_fun_elu_deriv(struct group *g, uint32_t i)
{
        if (g->vector->elements[i] > 0.0)
                return 1.0;
        else
                return g->vector->elements[i] + g->pars->relu_alpha;
}
