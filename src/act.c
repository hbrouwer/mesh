/*
 * act.c
 *
 * Copyright 2012-2014 Harm Brouwer <me@hbrouwer.eu>
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

/**************************************************************************
 * Feed forward
 *
 * This function propagates activation forward from a group g. Let j be
 * a unit in one of the network's groups, and i a unit in a group projecting
 * to it. The net input x_j to unit j is defined as:
 *
 *     x_j = sum_i (y_i * w_ij)
 *
 * where y_i is the activation level of unit i in the projecting group, and
 * w_ij the weight of the "synaptic" connection between unit j and unit i.
 * Provided the net input x_j, the activation level y_j of unit j is then
 * defined as:
 *
 *    y_j = f(x_j)
 *
 * where f is typically a non-linear activation function.
 *************************************************************************/
void feed_forward(struct network *n, struct group *g)
{
        /*
         * Under the assumption that activation levels for the units
         * in the current group have already been determined, determine
         * the activation levels of all the groups towards which the
         * current group maintains a projection.
         */
        for (uint32_t i = 0; i < g->out_projs->num_elements; i++) {
                struct projection *op = g->out_projs->elements[i];

                /*
                 * During BPTT, we want activation to propagate only 
                 * through the network of the current timestep.
                 */
                if (op->recurrent)
                        continue;

                /*
                 * Compute net input for the units in each group that the
                 * current group projects to.
                 */
                struct group *rg = op->to;
#ifdef _OPENMP
#pragma omp parallel for
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
                         * Note: A unit can receive activation from units
                         * in different projecting groups.
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
                        if (rg->act_fun->fun != act_fun_softmax) {
                                if (!n->act_lookup) {
                                        rg->vector->elements[j] = rg->act_fun->fun(rg->vector, j);
                                } else {
                                        rg->vector->elements[j] = act_lookup(rg->vector->elements[j],
                                                        rg->act_fun->lookup);
                                }
                        }
                }

                /* apply softmax activation function (if required) */
                if (rg->act_fun->fun == act_fun_softmax) {
#ifdef _OPENMP
#pragma omp parallel for
#endif /* _OPENMP */
                        for (uint32_t j = 0; j < rg->vector->size; j++) {
                                rg->vector->elements[j] = rg->act_fun->fun(rg->vector, j);
                        }
                }
        }

        /* 
         * Recursively repeat the above for all of the groups towards
         * which the current group maintains a projection. Again, we
         * skip recurrent projections, as we want activation to only
         * propagate through the network of the current timestep during
         * BPTT.
         */
        for (uint32_t i = 0; i < g->out_projs->num_elements; i++) {
                struct projection *op = g->out_projs->elements[i];
                if (!op->recurrent)
                        feed_forward(n, op->to);
        }
}

/**************************************************************************
 * Activation lookup
 *************************************************************************/
#define ACT_LOOKUP_MINIMUM -16
#define ACT_LOOKUP_MAXIMUM 16
#define ACT_LOOKUP_GRANULARITY 1024

double ACT_LOOKUP_STEP_SIZE = ((double)ACT_LOOKUP_MAXIMUM 
                - ACT_LOOKUP_MINIMUM) / ACT_LOOKUP_GRANULARITY;

/**************************************************************************
 * Creates a lookup vector for the specified activation function.
 *************************************************************************/
struct vector *create_act_lookup_vector(double (*fun)(struct vector *,
                        uint32_t))
{
        /* skip softmax */
        if (fun == act_fun_softmax)
                return NULL;

        struct vector *lv = create_vector(ACT_LOOKUP_GRANULARITY);

        for (uint32_t i = 0; i < ACT_LOOKUP_GRANULARITY; i++) {
                lv->elements[i] = ACT_LOOKUP_MINIMUM + i * ACT_LOOKUP_STEP_SIZE;
                lv->elements[i] = fun(lv, i);
        }

        return lv;
}

/**************************************************************************
 * Lookup the activation value for net input x in lookup vector lv.
 *************************************************************************/
double act_lookup(double x, struct vector *lv)
{
        uint32_t i;

        if (x <= ACT_LOOKUP_MINIMUM) {
                i = 0;
        } else if (x >= ACT_LOOKUP_MAXIMUM) {
                i = ACT_LOOKUP_GRANULARITY - 1;
        } else {
                i = ((ACT_LOOKUP_MAXIMUM + x) / ACT_LOOKUP_STEP_SIZE) - 1;
        }

        return lv->elements[i];
}

/**************************************************************************
 * Binary sigmoid function:
 *
 * f(x) = 1 / (1 + e ^ (-x))
 *************************************************************************/
double act_fun_binary_sigmoid(struct vector *v, uint32_t i)
{
        double x = v->elements[i];

        return 1.0 / (1.0 + exp(-x));
}

/**************************************************************************
 * Derivative of the binary sigmoid function:
 *
 * f'(x) = y * (1 - y)
 *************************************************************************/
double act_fun_binary_sigmoid_deriv(struct vector *v, uint32_t i)
{
        double y = v->elements[i];

        return y * (1.0 - y);
}

/**************************************************************************
 * Bipolar sigmoid function:
 *
 * f(x) = (-1) + 2 / (1 / e ^ (-x))
 *************************************************************************/
double act_fun_bipolar_sigmoid(struct vector *v, uint32_t i)
{
        double x = v->elements[i];

        return (-1.0) + 2.0 / (1.0 + exp(-x));
}

/**************************************************************************
 * Derivative of the bipolar sigmoid function:
 *
 * f'(x) = 0.5 * (1 + y) * (1 - y)
 *************************************************************************/
double act_fun_bipolar_sigmoid_deriv(struct vector *v, uint32_t i)
{
        double y = v->elements[i];

        return 0.5 * (1.0 + y) * (1.0 - y);
}

/**************************************************************************
 * Softmax function:
 *
 * f(x) = (e ^ x) / sum_j (e ^ x_j)
 *************************************************************************/
double act_fun_softmax(struct vector *v, uint32_t i)
{
        static double sum;
        if (i == 0) {
                sum = 0.0;
                for (uint32_t j = 0; j < v->size; j++)
                        sum += exp(v->elements[j]);
        }

        double x = exp(v->elements[i]);

        return x / sum;
}

/**************************************************************************
 * Derivative of the softmax function:
 *
 * f'(x) = 1
 *************************************************************************/
double act_fun_softmax_deriv(struct vector *v, uint32_t i)
{
        return 1.0;
}

/**************************************************************************
 * Hyperbolic tangent function:
 *
 * f(x) = (e ^ (2 * x) - 1) / (e ^ (2 * x) + 1)
 *************************************************************************/
double act_fun_tanh(struct vector *v, uint32_t i)
{
        double x = v->elements[i];

        return tanh(x);
}

/**************************************************************************
 * Derivative of the hyperbolic tangent function:
 *
 * f'(x) = 1 - y ^ 2;
 *************************************************************************/
double act_fun_tanh_deriv(struct vector *v, uint32_t i)
{
        double y = v->elements[i];

        return 1.0 - pow(y, 2.0);
}

/**************************************************************************
 * Linear function:
 *
 * f(x) = x
 *************************************************************************/
double act_fun_linear(struct vector *v, uint32_t i)
{
        double x = v->elements[i];

        return x;
}

/**************************************************************************
 * Derivative of the linear function:
 *
 * f'(x) = 1
 *************************************************************************/
double act_fun_linear_deriv(struct vector *v, uint32_t i)
{
        return 1.0;
}

/**************************************************************************
 * Step function:
 *        
 *        | 1 , if x >= 0
 * f(x) = |
 *        | 0 , otherwise
 *************************************************************************/
double act_fun_step(struct vector *v, uint32_t i)
{
        double x = v->elements[i];

        if (x >= 0.0)
                return 1.0;
        else
                return 0.0;
}

/**************************************************************************
 * Derivative of the step function:
 *
 * f'(x) = 1
 *************************************************************************/
double act_fun_step_deriv(struct vector *v, uint32_t i)
{
        return 1.0;
}
