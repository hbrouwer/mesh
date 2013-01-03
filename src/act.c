/*
 * act.c
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
#include <stdio.h>

#include "act.h"
#include "vector.h"

/*
 * ########################################################################
 * ## Feed forward                                                       ##
 * ########################################################################
 */

/*
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
 */

void feed_forward(struct network *n, struct group *g)
{
        /*
         * If the current group has a context group, copy this group's
         * previous activation vector to the context group's vector.
         * Recursively repeat this, if the context group has a context 
         * group too.
         */
        if (g->context_group)
                shift_context_group_chain(n, g, g->vector);

        /*
         * Under the assumption that activation levels for the unit's
         * in the current group have already been determined, determine
         * the activation levels of all the groups towards which the
         * current group maintains a projection.
         */
        for (int i = 0; i < g->out_projs->num_elements; i++) {
                /*
                 * During BPTT, we want activation to propagate only 
                 * through the network of the current timestep.
                 */
                if (g->out_projs->elements[i]->recurrent)
                        continue;

                /*
                 * Compute net input and activation level for the units in 
                 * each group that the current group projects to.
                 */
                struct group *rg = g->out_projs->elements[i]->to;
                for (int j = 0; j < rg->vector->size; j++) {
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
                        for (int x = 0; x < rg->inc_projs->num_elements; x++) {
                                struct group *pg = rg->inc_projs->elements[x]->to;
                                struct matrix *w = rg->inc_projs->elements[x]->weights;
                                for (int z = 0; z < pg->vector->size; z++)
                                        rg->vector->elements[j] += pg->vector->elements[z]
                                                * w->elements[z][j];
                        }

                        /*
                         * Apply an activation function to the net input.
                         *
                         * y_j = f(x_j)
                         */
                        if (!n->use_act_lookup) {
                                rg->vector->elements[j] = rg->act_fun->fun(rg->vector, j);
                        } else {
                                rg->vector->elements[j] = act_lookup(rg->vector->elements[j],
                                                rg->act_fun->lookup);
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
        for (int i = 0; i < g->out_projs->num_elements; i++)
                if (!g->out_projs->elements[i]->recurrent)
                        feed_forward(n, g->out_projs->elements[i]->to);
}

/*
 * ########################################################################
 * ## Activation lookup                                                  ##
 * ########################################################################
 */

#define ACT_LOOKUP_MINIMUM -16
#define ACT_LOOKUP_MAXIMUM 16
#define ACT_LOOKUP_GRANULARITY 1024

double ACT_LOOKUP_STEP_SIZE = ((double)ACT_LOOKUP_MAXIMUM 
                - ACT_LOOKUP_MINIMUM) / ACT_LOOKUP_GRANULARITY;

/*
 * Creates a lookup vector for the specified activation function.
 */

struct vector *create_act_lookup_vector(double (*fun)(struct vector *, int))
{
        struct vector *lv = create_vector(ACT_LOOKUP_GRANULARITY);

        for (int i = 0; i < ACT_LOOKUP_GRANULARITY; i++) {
                lv->elements[i] = ACT_LOOKUP_MINIMUM + i * ACT_LOOKUP_STEP_SIZE;
                lv->elements[i] = fun(lv, i);
        }

        return lv;
}

/*
 * Lookup the activation value for net input x in lookup vector lv.
 */

double act_lookup(double x, struct vector *lv)
{
        if (x < ACT_LOOKUP_MINIMUM)
                x = ACT_LOOKUP_MINIMUM;
        if (x > ACT_LOOKUP_MAXIMUM)
                x = ACT_LOOKUP_MAXIMUM;

        int i = ((ACT_LOOKUP_MAXIMUM + x) / ACT_LOOKUP_STEP_SIZE) - 1;

        return lv->elements[i];
}

/*
 * ########################################################################
 * ## Binary sigmoid function                                            ##
 * ########################################################################
 */

double act_fun_binary_sigmoid(struct vector *v, int i)
{
        double x = v->elements[i];

        return 1.0 / (1.0 + exp(-x));
}

double act_fun_binary_sigmoid_deriv(struct vector *v, int i)
{
        double y = v->elements[i];

        return y * (1.0 - y);
}

/*
 * ########################################################################
 * ## Bipolar sigmoid function                                           ##
 * ########################################################################
 */

double act_fun_bipolar_sigmoid(struct vector *v, int i)
{
        double x = v->elements[i];

        return (-1.0) + 2.0 / (1.0 + exp(-x));
}

double act_fun_bipolar_sigmoid_deriv(struct vector *v, int i)
{
        double y = v->elements[i];

        return 0.5 * (1.0 + y) * (1.0 - y);
}

/*
 * ########################################################################
 * ## Softmax function                                                   ##
 * ########################################################################
 */

double act_fun_softmax(struct vector *v, int i)
{
        double x = exp(v->elements[i]);

        double sum = 0.0;
        for (int j = 0; j < v->size; j++)
                sum += exp(v->elements[j]);

        return x / sum;
}

double act_fun_softmax_deriv(struct vector *v, int i)
{
        return 1.0;
}

/*
 * ########################################################################
 * ## Hyperbolic tangent function                                        ##
 * ########################################################################
 */

double act_fun_tanh(struct vector *v, int i)
{
        double x = v->elements[i];

        return tanh(x);
}

double act_fun_tanh_deriv(struct vector *v, int i)
{
        double y = v->elements[i];

        return 1.0 - y * y;
}

/*
 * ########################################################################
 * ## Linear function                                                    ##
 * ########################################################################
 */

double act_fun_linear(struct vector *v, int i)
{
        double x = v->elements[i];

        return x;
}

double act_fun_linear_deriv(struct vector *v, int i)
{
        return 1.0;
}

/*
 * ########################################################################
 * ## Step function                                                      ##
 * ########################################################################
 */

double act_fun_step(struct vector *v, int i)
{
        double x = v->elements[i];

        if (x >= 0.0)
                return 1.0;
        else
                return 0.0;
}

double act_fun_step_deriv(struct vector *v, int i)
{
        return 1.0;
}
