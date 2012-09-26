/*
 * act.c
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
 * This function propagates activation forward from a group g.  Let j be
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
                 * Compute net input and activation level for each group
                 * that the current group projects to.
                 */
                struct group *rg = g->out_projs->elements[i]->to;
                for (int j = 0; j < rg->vector->size; j++) {
                        rg->vector->elements[j] = unit_net_input(n, rg, j);
                        rg->vector->elements[j] = rg->act->fun(rg->vector, j);
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
 * Compute the summed, weighted net input to a unit.
 */

double unit_net_input(struct network *n, struct group *g, int u)
{
        double act = 0.0;

        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct group *pg = g->inc_projs->elements[i]->to;
                struct matrix *w = g->inc_projs->elements[i]->weights;
                for (int j = 0; j < pg->vector->size; j++)
                        act += w->elements[j][u] * pg->vector->elements[j];
        }

        return act;
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
