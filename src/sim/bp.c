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
#include "error.h"

#include <math.h>

/*
 * This implements the backpropagation (BP) algorithm (Rumelhart, Hinton,
 * & Williams, 1986). BP minimizes the network's error E, given some error
 * function. A commonly used error function is sum squared error, which is
 * defined as:
 *
 *     E = 0.5 * sum_j (y_j - d_j)^2
 *
 * where y_j is the observed activation level for output unit j, and d_j its
 * target activation level. To minimize this error function, we first
 * determine the error derivative EA_j, which defines how fast the error at
 * unit j changes as a function of its activation level:
 *
 *     EA_j = @E / @y_j
 *          = y_j - d_j
 *
 * Provided EA_j, we can compute how the error changes as function of the
 * the net input to unit j. We term this quantity EI_j, and define it as:
 *
 *     EI_j = @E / @x_j 
 *          = EA_j * f'(y_j)
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
 *          = sum_j (EI_j * W_ij)
 *
 * We can repeat this procedure to compute the EA quantities for as many
 * preceding groups as required. Provided the error derivative EA_j for
 * a unit j, we can also obtain EI_j for that unit, which we can in turn use
 * to compute how fast the error changes with respect to a weight W_ij on
 * the connection between unit j in the output layer, and unit i in
 * a preceding layer:
 *
 *     EW_ij = @E / @W_ij
 *           = (@E / @x_j) * (@x_j / @w_ij)
 *           = EI_j * Y_i
 *
 * This can then be used to update the respective weight W_ij by means of:
 *
 *     W_ij = W_ij + DW_ij
 *
 * where DW_ij is defined as:
 *
 *     DW_ij(t) = -e * EW_ij + a * DW_ij(t-1) - d * W_ij
 *
 * and where, in turn, e is a learning rate coefficient, a is a momentum
 * coeffecieint, d is weight decay coefficient, and  DW_ij(t-1) is the
 * previous weight change on the connection between unit i and unit j.
 *
 * References
 *
 * Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning
 *     representations by back-propagating errors. Nature, 323, 553-536.
 */

/*
 * Flat spot correction constant. See:
 *
 * Fahlman, S. E. (1988). An empirical study of learning speed in back-
 *     propagation networks. Technical report CMU-CS-88-162. School of
 *     Computer Science, Caernie Mellon University, Pittsburgh, PA 15213.
 */
#define BP_FLAT_SPOT_CORRECTION 0.1

/*
 * ########################################################################
 * ## Error backpropagation                                              ##
 * ########################################################################
 */

/*
 * This computes EI_j quantities for all units j in the output layer.
 */

struct vector *bp_output_error(struct network *n)
{
        struct vector *e = n->error->deriv(n);

        /*
         * If the error function E that is being minimized is sum of 
         * squares, we multiply EA_j with f'(y_i). For cross-entropy
         * error, the f'(y_i) is cancelled out.
         */
        if (n->error->fun == error_sum_of_squares) {
                for (int i = 0; i < e->size; i++) {
                        struct group *g = n->output;
                        e->elements[i] *= g->act->deriv(g->vector, i)
                                + BP_FLAT_SPOT_CORRECTION;
                }
        }

        return e;
}

/*
 * This is the main BP function. Provided a group g, and a vector e with
 * errors EI for that group's units, it first computes the error derivatives
 * EA and weight deltas EW for each projection to g. In case of unfolded
 * networks, which are used for BP through time, a group may project to
 * multiple later groups, which means that an error derivative EA_i for
 * a unit i in that group, may depend on multiple projections. Therefore, we
 * need to sum the EA_i values for all outgoing projections of the group to
 * which unit i belongs, before we can determine EI_i. Once we have obtained
 * all EI values for a projecting group, we recursively backpropagate that
 * error to earlier groups.
 */

void bp_backpropagate_error(struct network *n, struct group *g,
                struct vector *e)
{
        /*
         * For each group that projects to g, compute the error derivatives
         * EA and weight deltas EW with respect to g.
         */
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *p = g->inc_projs->elements[i];
                
                /* 
                 * Clean previous error for this projection. Do not touch
                 * weight deltas, as these can cumulate over multiple
                 * backpropagation sweeps.
                 */
                zero_out_vector(p->error);

                bp_projection_error_and_weight_deltas(n, p, e);
        }

        /*
         * Sum the error derivatives for each group that projects towards
         * g, compute EI quantities for each units in that group, and
         * recursively backpropagate that error to earlier groups.
         */
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct group *ng = g->inc_projs->elements[i]->to;
                
                struct vector *ge = bp_group_error(n, ng);
                bp_backpropagate_error(n, ng, ge);

                dispose_vector(ge);
        }
}

/* 
 * This function computes the error derivates EA and weight deltas EW for
 * a given projection p between g' and g. 
 */

void bp_projection_error_and_weight_deltas(struct network *n, struct 
                projection *p, struct vector *e)
{
        for (int i = 0; i < p->to->vector->size; i++) {
                for (int j = 0; j < e->size; j++) {
                        /*
                         * Compute how the error changes as a function of
                         * the output of unit i:
                         *
                         * EA_i = sum_j (EI_j * W_ij)
                         */
                        p->error->elements[i] += e->elements[j]
                                * p->weights->elements[i][j];

                        /*
                         * Compute how the error changes as a function of
                         * the weight on the connection between unit i and
                         * unit j:
                         *
                         * E_Wij += EI_j * Y_i
                         */
                        p->deltas->elements[i][j] += e->elements[j]
                                * p->to->vector->elements[i];
                }
        }
}

/*
 * This function compute the EI quantities for a group g. We first sum for
 * each of its units i, the error derivates EA_i for all of its outgoing
 * projections.  Next, we obtain EI_i by multiplying the summed EA_i
 * quantities with f'(Y_i). However, if g is the network's input group, EI_i
 * is simply the summed EA_i.
 */

struct vector *bp_group_error(struct network *n, struct group *g)
{
        struct vector *e = create_vector(g->vector->size);

        for (int i = 0; i < g->vector->size; i++) {
                /* 
                 * Sum error derivates EA_i for all outgoing projections
                 * of the current group.
                 */
                for (int j = 0; j < g->out_projs->num_elements; j++) {
                        struct projection *p = g->out_projs->elements[j];
                        e->elements[i] += p->error->elements[i];
                }

                /*
                 * Compute how the error changes as function of the net
                 * input to unit i: 
                 *
                 * EI_i = EA_i * f'(y_i)
                 */
                double act_deriv;
                if (g != n->input) {
                        act_deriv = g->act->deriv(g->vector, i)
                                + BP_FLAT_SPOT_CORRECTION;
                } else {
                        act_deriv = g->vector->elements[i];
                }

                e->elements[i] *= act_deriv;
        }

        return e;
}

/*
 * ########################################################################
 * ## Weight adjustment                                                  ##
 * ########################################################################
 */

/*
 * This recursively adjusts the weights of all incoming projections of a
 * group g.
 */

void bp_adjust_weights(struct network *n, struct group *g)
{
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *p = g->inc_projs->elements[i];
                /*
                 * Adjust weights if projection is not frozen.
                 */
                if (!p->frozen)
                        bp_adjust_projection_weights(n, g, p);
        
                /*
                 * Make a copy of the weight deltas for the application of
                 * momentum and weight decay upon next update, and reset the
                 * the current weight deltas.
                 */
                copy_matrix(p->prev_deltas, p->deltas);
                zero_out_matrix(p->deltas);

                /*
                 * During BPTT, we want to only adjust weights
                 * in the network of the current timestep.
                 */
                if (p->recurrent)
                        continue;

                bp_adjust_weights(n, p->to);
        }
}

/*
 * This adjusts the weights of a projection p between a group g' and g.
 */

void bp_adjust_projection_weights(struct network *n, struct group *g,
                struct projection *p)
{
        /*
         * Adjust the weight between unit i in group g'
         * and unit j in group g.
         */
        for (int i = 0; i < p->to->vector->size; i++) {
                for (int j = 0; j < g->vector->size; j++) {
                        /*
                         * First, we apply learning:
                         *
                         * DW_ij = -e * EW_ij
                         */
                        double weight_change = -n->learning_rate
                                * p->deltas->elements[i][j];

                        /*
                         * Next, we apply momentum:
                         *
                         * DW_ij = DW_ij + a * DW_ij(t-1)
                         */
                        weight_change += n->momentum
                                * p->prev_weight_changes->elements[i][j];
                        
                        /*
                         * Finally, we apply weight decay:
                         *
                         * DW_ij = DW_ij - d * W_ij
                         */
                        weight_change -= n->weight_decay 
                                * p->weights->elements[i][j];

                        /*
                         * Adjust the weight:
                         *
                         * W_ij = W_ij + DW_ij
                         */
                        p->weights->elements[i][j] += weight_change;
                        
                        /* 
                         * Store a copy of the weight change.
                         */
                        p->prev_weight_changes->elements[i][j] = weight_change;
                }
        }
}
