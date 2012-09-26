/*
 * bp.c
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

#include "bp.h"

/*
 * This provides an implementation of the backpropagation (BP) algorithm
 * (Rumelhart, Hinton, & Williams, 1986). BP minimizes the network's error
 * E, given some error function. A commonly used error function is sum 
 * squared error, which is defined as:
 *
 *     E = 0.5 * sum_j (y_j - d_j)^2
 *
 * where y_j is the observed activation level for output unit j, and d_j
 * its target activation level. To minimize this error function, we first
 * determine the error derivative EA_j, which defines how fast the error at
 * unit j changes as a function of its activation level:
 *
 *     EA_j = @E / @y_j = y_j - d_j
 *
 * Provided EA_j, we can compute how the error changes as function of the
 * the net input to unit j. We term this quantity EI_j, and define it as:
 *
 *     EI_j = @E / @x_j = EA_j * f'(y_j)
 *
 * where f' is the derivative of the activation function used.  We can use
 * the EI quantities of all units of the group towards which unit j belongs
 * to compute the error derivative EA_i for a unit i that is connected to
 * all units in that group. The error derivative EA_i for unit i is simply
 * the sum of all EI_j quantities multiplied by the weight W_ij of the
 * connection between each unit j and unit i:
 *
 *     EA_i = @E / @y_i
 *          = sum_j ((@E / @x_j) * (@x_j / @y_i)
 *          = sum_j (EI_j * w_ij)
 *
 * We can repeat this procedure to compute the EA quantities for as many
 * preceding groups as required. Provided the error derivative EA_j for
 * a unit j, we can also obtain EI_j for that unit, which we can in turn use
 * to compute how fast the error changes with respect to a weight w_ij on
 * the connection between unit j in the output layer, and unit i in
 * a preceding layer:
 *
 *     EW_ij = @E / @w_ij = (@E / @x_j) * (@x_j / @w_ij) = EI_j * Y_i
 *
 * This can then be used to update the respective weight W_ij by means of:
 *
 *     W_ij = e * EW_ij
 *
 * References
 *
 * Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning
 *     representations by back-propagating errors. Nature, 323, 553-536.
 */

void bp_backpropagate_error(struct network *n, struct group *g,
                struct vector *e)
{
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *p = g->inc_projs->elements[i];
                zero_out_vector(p->error);
                bp_projection_deltas_and_error(n, p, e);
        }

        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct group *ng = g->inc_projs->elements[i]->to;
                struct vector *ge = bp_sum_group_error(n, ng);
                bp_backpropagate_error(n, ng, ge);
                dispose_vector(ge);
        }
}

void bp_projection_deltas_and_error(struct network *n, struct projection *p,
                struct vector *e)
{
        for (int i = 0; i < p->to->vector->size; i++) {
                for (int j = 0; j < e->size; j++) {
                        p->error->elements[i] += p->weights->elements[i][j]
                                * e->elements[j];
                        p->deltas->elements[i][j] += p->to->vector->elements[i]
                                * e->elements[j];
                }
        }        
}

struct vector *bp_sum_group_error(struct network *n, struct group *g)
{
        struct vector *e = create_vector(g->vector->size);

        for (int i = 0; i < g->vector->size; i++) {
                for (int j = 0; j < g->out_projs->num_elements; j++) {
                        struct projection *p = g->out_projs->elements[j];
                        e->elements[i] += p->error->elements[i];
                }

                double act_deriv;
                if (g != n->input)
                        act_deriv = g->act->deriv(g->vector, i);
                else 
                        act_deriv = g->vector->elements[i];

                e->elements[i] *= act_deriv;
        }

        return e;
}

/*
 * ########################################################################
 * ## Weight adjustment                                                  ##
 * ########################################################################
 */

void bp_adjust_weights(struct network *n, struct group *g)
{
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                bp_adjust_projection_weights(n, g, g->inc_projs->elements[i]);
                if (!g->inc_projs->elements[i]->recurrent)
                        bp_adjust_weights(n, g->inc_projs->elements[i]->to);
        }        
}

void bp_adjust_projection_weights(struct network *n, struct group *g,
                struct projection *p)
{
        for (int i = 0; i < p->to->vector->size; i++)
                for (int j = 0; j < g->vector->size; j++)
                        p->weights->elements[i][j] += 
                                n->learning_rate
                                * p->deltas->elements[i][j]
                                - n->weight_decay
                                * p->prev_deltas->elements[i][j]
                                + n->momentum
                                * p->prev_deltas->elements[i][j];
        
        copy_matrix(p->prev_deltas, p->deltas);
        zero_out_matrix(p->deltas);        
}
