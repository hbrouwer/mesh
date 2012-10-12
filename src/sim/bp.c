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
#include "math.h"

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
 * unit j changes as a function of that unit's activation level:
 *
 *     EA_j = @E / @y_j 
 *          = y_j - d_j
 *
 * Provided EA_j, we can compute how the error changes as function of the
 * the net input to unit j. We term this quantity EI_j, and define it as:
 *
 *     EI_j = @E / @x_j 
 *          = @E / @y_j * @y_j / @x_j 
 *          = EA_j * f'(y_j)
 *
 * where f' is the derivative of the activation function used.  We can use
 * the EI quantities of all units of the group towards which unit j belongs
 * to compute the error derivative EA_i for a unit i that is connected to
 * all units in that group. The error derivative EA_i for unit i is simply
 * the sum of all EI_j quantities multiplied by the weight w_ij of the
 * connection between each unit j and unit i:
 *
 *     EA_i = @E / @y_i
 *          = sum_j ((@E / @x_j) * (@x_j / @y_i) 
 *          = sum_j (EI_j * w_ij)
 *
 * We can repeat this procedure to compute the EA quantities for as many
 * preceding groups as required. Provided the error derivative EA_j for
 * a unit j, we can also obtain EI_j for that unit, which we can in turn use
 * to compute how fast the error changes with respect to a weight W_ij on
 * the connection between unit j in the output layer, and unit i in
 * a preceding layer:
 *
 *     EW_ij = @E / @w_ij 
 *           = (@E / @x_j) * (@x_j / @w_ij) 
 *           = EI_j * Y_i
 *
 * This can then be used to update the respective weight W_ij by means of:
 *
 *     w_ij = w_ij + DW_ij
 *
 * When using steepest descent weight updating, DW_ij is defined as:
 *
 *     DW_ij(t) = -e * EW_ij + a * DW_ij(t-1) - d * w_ij
 *
 * where e is a learning rate coefficient, a is a momentum coefficient,
 * d is weight decay coefficient, and  DW_ij(t-1) is the previous weight
 * change on the connection between unit i and unit j.
 *
 * References
 *
 * Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning
 *     representations by back-propagating errors. Nature, 323, 553-536.
 */

/*
 * Flat spot correction constant. This constant is added to the activation
 * function derivative f'(y_j) to avoid that it approaches zero when y_j
 * is near 1.0 or 0.0. See:
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

void bp_output_error(struct group *g, struct vector *t)
{
        /*
         * First, compute error derivates EA_j for all units in the output
         * layer.
         */
        g->err_fun->deriv(g, t);

        /*
         * Multiply all EA_j quantities with f'(y_i) to obtain EI_j for
         * each unit.
         */
        for (int i = 0; i < g->error->size; i++) {
                g->error->elements[i] *= g->act_fun->deriv(g->vector, i)
                        + BP_FLAT_SPOT_CORRECTION;
        }
}

/*
 * This is the main BP function. Provided a group g, and a vector e with
 * errors EI for that group's units, it first computes the error derivatives
 * EA and weight gradients EW for each projection to g. In case of unfolded
 * networks, which are used for BP through time, a group may project to
 * multiple later groups, which means that an error derivative EA_i for
 * a unit i in that group, may depend on multiple projections. Therefore, we
 * need to sum the EA_i values for all outgoing projections of the group to
 * which unit i belongs, before we can determine EI_i. Once we have obtained
 * all EI values for a projecting group, we recursively backpropagate that
 * error to earlier groups.
 */

void bp_backpropagate_error(struct network *n, struct group *g)
{
        /*
         * For each group that projects to g, compute the error derivatives
         * EA and weight gradients EW with respect to g.
         */
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *p = g->inc_projs->elements[i];
                
                /* 
                 * Clean previous error for this projection. Do not touch
                 * weight gradients, as these can cumulate over multiple
                 * backpropagation sweeps.
                 */
                zero_out_vector(p->error);

                /*
                 * Compute error derivatives EA and weight gradients EW for
                 * the current projection between g' and g.
                 */
                for (int x = 0; x < p->to->vector->size; x++) {
                        for (int z = 0; z < g->error->size; z++) {
                                /*
                                 * Compute how the error changes as a function of
                                 * the output of unit i:
                                 *
                                 * EA_i = sum_j (EI_j * W_ij)
                                 */
                                p->error->elements[x] += g->error->elements[z]
                                        * p->weights->elements[x][z];

                                /*
                                 * Compute how the error changes as a function of
                                 * the weight on the connection between unit i and
                                 * unit j:
                                 *
                                 * EW_ij += EI_j * Y_i
                                 *
                                 * Note: gadients sum over an epoch.
                                 */
                                p->gradients->elements[x][z] += g->error->elements[z]
                                        * p->to->vector->elements[x];
                        }
                }
        }

        /*
         * Sum the error derivatives for each group that projects towards
         * g, compute EI quantities for each unit in that group, and
         * recursively backpropagate that error to earlier groups.
         */
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct group *ng = g->inc_projs->elements[i]->to;
                zero_out_vector(ng->error);

                for (int x = 0; x < ng->vector->size; x++) {
                        /* 
                         * Sum error derivates EA_i for all outgoing
                         * projections of the current group.
                         */
                        for (int z = 0; z < ng->out_projs->num_elements; z++) {
                                struct projection *p = ng->out_projs->elements[z];
                                ng->error->elements[x] += p->error->elements[x];
                        }

                        /*
                         * Compute how the error changes as function of the
                         * net input to unit i: 
                         *
                         * EI_i = EA_i * f'(y_i)
                         */
                        double act_deriv;
                        if (ng != n->input) {
                                act_deriv = ng->act_fun->deriv(ng->vector, x)
                                        + BP_FLAT_SPOT_CORRECTION;
                        } else {
                                act_deriv = ng->vector->elements[x];
                        }

                        ng->error->elements[x] *= act_deriv;
                }

                /*
                 * Backpropagate error.
                 */
                bp_backpropagate_error(n, ng);
        }
}

/*
 * ########################################################################
 * ## Steepest descent weight updating                                   ##
 * ########################################################################
 */

/*
 * This implements steepest (or gradient) descent backpropagation. Steepest
 * descent is a first-order optimization algorithm for finding the nearest
 * local minimum of a function. On each weight update, a step is taken that
 * is proportional to the negative of the gradient of the function that is
 * being minimized.
 */

void bp_update_sd(struct network *n)
{
        n->status->weight_cost = 0.0;
        n->status->gradient_linearity = 0.0;
        n->status->last_weight_deltas_length = 0.0;
        n->status->gradients_length = 0.0;

        bp_recursively_update_sd(n, n->output);

        n->status->gradient_linearity /=
                sqrt(n->status->last_weight_deltas_length
                                * n->status->gradients_length);
}

/*
 * This recursively adjusts the weights of all incoming projections of a
 * group g.
 */

void bp_recursively_update_sd(struct network *n, struct group *g)
{
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *p = g->inc_projs->elements[i];
                /*
                 * Adjust weights if projection is not frozen.
                 */
                if (!p->frozen)
                        bp_update_projection_sd(n, g, p);
                
                /*
                 * Make a copy of the weight gradients, and reset the
                 * current weight gradients.
                 */
                copy_matrix(p->prev_gradients, p->gradients);
                zero_out_matrix(p->gradients);

                /*
                 * During BPTT, we want to only adjust weights
                 * in the network of the current timestep.
                 */
                if (p->recurrent)
                        continue;

                bp_recursively_update_sd(n, p->to);
        }
}

/*
 * This adjusts the weights of a projection p between a group g' and g.
 */

void bp_update_projection_sd(struct network *n, struct group *g,
                struct projection *p)
{
        /*
         * Adjust the weight between unit i in group g'
         * and unit j in group g.
         */
        for (int i = 0; i < p->to->vector->size; i++) {
                for (int j = 0; j < g->vector->size; j++) {
                        double weight_delta = 0.0;

                        /*
                         * First, we apply learning:
                         *
                         * DW_ij = -e * EW_ij
                         */
                        weight_delta += -n->learning_rate
                                * p->gradients->elements[i][j];

                        /*
                         * Next, we apply momentum:
                         *
                         * DW_ij = DW_ij + a * DW_ij(t-1)
                         */
                        weight_delta += n->momentum
                                * p->prev_weight_deltas->elements[i][j];
                        
                        /*
                         * Finally, we apply weight decay:
                         *
                         * DW_ij = DW_ij - d * W_ij
                         */
                        weight_delta -= n->weight_decay 
                                * p->weights->elements[i][j];

                        /*
                         * Adjust the weight:
                         *
                         * w_ij = w_ij + DW_ij
                         */
                        p->weights->elements[i][j] += weight_delta;
                        
                        /********************************/

                        n->status->weight_cost +=
                                square(p->weights->elements[i][j]);
                        n->status->gradient_linearity -= 
                                p->prev_weight_deltas->elements[i][j]
                                * p->gradients->elements[i][j];
                        n->status->last_weight_deltas_length +=
                                square(p->prev_weight_deltas->elements[i][j]);
                        n->status->gradients_length +=
                                square(p->gradients->elements[i][j]);

                        /********************************/

                        /* 
                         * Store a copy of the weight change.
                         */
                        p->prev_weight_deltas->elements[i][j] = weight_delta;
                }
        }
}

/*
 * ########################################################################
 * ## Resilient backpropagation weight updating                          ##
 * ########################################################################
 */

/*
 * This implements resilient backpropagation (Rprop) (see Igel & Husken,
 * 2000). In Rprop, weight adjustments are made on the basis of the sign of
 * the partial derivative @E/@w_ij. An Rprop update iteration can be divided
 * into two stages. In the first stage, the "update value" u_ij for each 
 * weight w_ij is computed:
 *
 *           | eta_plus * u_ij(t-1)   , if @E/@w_ij(t-1) * @E/@w_ij(t) > 0
 *           |
 * u_ij(t) = | eta_minus * u_ij(t-1)  , if @E/@w_ij(t-1) * @E/@w_ij(t) < 0
 *           |
 *           | u_ij(t-1)              , otherwise
 *
 * where eta_plus and eta_minus are defined as:
 *
 *    0 < eta_minus < 1 < eta_plus.
 *
 * and u_ij(t) is bounded by u_max and u_min. The second stage of an Rprop
 * iteration depends on the particular Rprop flavour. Four Rprop flavours
 * are implemented (see Igel & Husken, 2000):
 *
 * (1) RPROP+ (rprop with weight-backtracking)
 *
 *     After computing the "update value" u_ij for each weight w_ij, the
 *     second stage depends on whether the sign of @E/@w_ij has changed
 *     from timestep t-1 to t. If it has not changed, we perform a regular
 *     weight update:
 *
 *         if @E/@w_ij(t-1) > @E/@w_ij(t) > 0 then
 *
 *            DW_ij(t) = -sign(@E/@w_ij(t)) * u_ij(t)
 *
 *     where sign(x) returns +1 if x is positive and -1 if x is negative.
 *     If, on the other hand, the sign has changed, we revert the previous
 *     weight update (weight backtracking), and reset the partial derivative
 *     @E/@w_ij(t) to 0, so that the u_ij will not be adjusted on the next 
 *     iteration:
 *
 *         if @E/@w_ij(t-1) > @E/@w_ij(t) > 0 then
 *
 *            DW_ij(t) = -DW_ij(t-1)
 *
 *            @E/@W_ij(t) = 0
 *
 *     Finally, weights are updated by means of: 
 *
 *          w_ij = w_ij + u_ij(t).
 *
 * (2) RPROP- (rprop without weight-backtracking)
 *
 *     A variation on RPROP+ in which weight backtracking is omitted, and
 *     in which E@/@W_ij(t) is not reset to 0 when its sign has changed.
 *
 * (3) iRPROP+ ("modified" rprop with weight-backtracking)
 *
 *     A variation on RPROP+ in which weight backtracking is only performed
 *     if the overall error goes up from timestep t-1 to t.
 *
 * (4) iRPROP- ("modified" Rprop without weight-backtracking)
 *
 *     A variation on RPROP- in which E@/@W_ij(t) is reset to 0 when its
 *     sign has changed.
 *
 * References
 *
 * Igel, C., & Husken, M. (2000). Improving the Rprop Algorithm. Proceedings
 *     of the Second International Symposium on Neural Computation, NC'2000,
 *     pp. 115-121, ICSC, Academic Press, 2000.
 */

#define RP_MAX_STEP_SIZE 50.0
#define RP_MIN_STEP_SIZE 1e-6

void bp_update_rprop(struct network *n)
{
        n->status->weight_cost = 0.0;
        n->status->gradient_linearity = 0.0;
        n->status->last_weight_deltas_length = 0.0;
        n->status->gradients_length = 0.0;

        n->rp_eta_plus = 1.2;
        n->rp_eta_minus = 0.5;

        bp_recursively_update_rprop(n, n->output);

        n->status->gradient_linearity /=
                sqrt(n->status->last_weight_deltas_length
                                * n->status->gradients_length);
}

void bp_recursively_update_rprop(struct network *n, struct group *g)
{
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *p = g->inc_projs->elements[i];
                /*
                 * Adjust weights if projection is not frozen.
                 */
                if (!p->frozen)
                        bp_update_projection_rprop(n, g, p);
                
                /*
                 * Make a copy of the weight gradients, and reset the
                 * current weight gradients.
                 */
                copy_matrix(p->prev_gradients, p->gradients);
                zero_out_matrix(p->gradients);

                /*
                 * During BPTT, we want to only adjust weights
                 * in the network of the current timestep.
                 */
                if (p->recurrent)
                        continue;

                bp_recursively_update_rprop(n, p->to);
        }
}

void bp_update_projection_rprop(struct network *n, struct group *g,
                struct projection *p)
{
        /*
         * Adjust the weight between unit i in group g'
         * and unit j in group g.
         */
        for (int i = 0; i < p->to->vector->size; i++) {
                for (int j = 0; j < g->vector->size; j++) {
                        double weight_delta = 0.0;

                        /*
                         * First, apply weight decay:
                         *
                         * DW_ij = DW_ij - d * W_ij
                         */
                        weight_delta -= n->weight_decay 
                                * p->weights->elements[i][j];

                        /*
                         * Sign of @E/@w_ij has not changed:
                         * 
                         * @E/@w_ij(t-1) * @E/@w_ij(t) > 0
                         */
                        if (p->prev_gradients->elements[i][j]
                                        * p->gradients->elements[i][j] > 0.0) {

                                /*
                                 * Bind update value u_ij to u_max.
                                 */
                                p->dyn_learning_pars->elements[i][j] = minimum(
                                                p->dyn_learning_pars->elements[i][j] * n->rp_eta_plus,
                                                RP_MAX_STEP_SIZE);

                                /*
                                 * Perform weight update:
                                 *
                                 * DW_ij = -sign(@E/@w_ij(t)) * u_ij(t)
                                 *
                                 * w_ij = w_ij + DW_ij
                                 */
                                weight_delta += -sign(p->gradients->elements[i][j]) 
                                        * p->dyn_learning_pars->elements[i][j];
                                p->weights->elements[i][j] += weight_delta;

                        /*
                         * Sign of @E/@w_ij has changed:
                         * 
                         * @E/@w_ij(t-1) * @E/@w_ij(t) < 0
                         */
                        } else if (p->prev_gradients->elements[i][j]
                                        * p->gradients->elements[i][j] < 0.0) {

                                /*
                                 * Bind update value u_ij to u_min.
                                 */
                                p->dyn_learning_pars->elements[i][j] = maximum(
                                                p->dyn_learning_pars->elements[i][j] * n->rp_eta_minus,
                                                RP_MIN_STEP_SIZE);

                                /*
                                 * Perform weight backtracking for RPROP+.
                                 */
                                if (n->rp_type == RPROP_PLUS)
                                        p->weights->elements[i][j] -=
                                                p->prev_weight_deltas->elements[i][j];

                                /*
                                 * Perform weight backtracking for iRPROP+.
                                 */
                                if (n->rp_type == IRPROP_PLUS)
                                        if (n->status->error > n->status->prev_error)
                                                p->weights->elements[i][j] -=
                                                        p->prev_weight_deltas->elements[i][j];

                                /*
                                 * Set @E/@w_ij(t) to 0 for all Rprop
                                 * flavours except RPROP_MINUS.
                                 */
                                if (n->rp_type != RPROP_MINUS)
                                        p->gradients->elements[i][j] = 0.0;

                                /* 
                                 * Perform weight change for RPROP- and
                                 * iRPROP-:
                                 * 
                                 * DW_ij = -sign(@E/@w_ij(t)) * u_ij(t)
                                 *
                                 * w_ij = w_ij + DW_ij
                                 */
                                if (n->rp_type == RPROP_MINUS || n->rp_type == IRPROP_MINUS) {
                                        weight_delta += -sign(p->gradients->elements[i][j]) *
                                                p->dyn_learning_pars->elements[i][j];
                                        p->weights->elements[i][j] += weight_delta;
                                }

                        /*
                         * Otherwise:
                         *
                         * @E/@w_ij(t-1) * @E/@w_ij(t) = 0
                         */
                        } else if (p->prev_gradients->elements[i][j]
                                        * p->gradients->elements[i][j] == 0.0) {
                                /*
                                 * Perform weight update:
                                 *
                                 * DW_ij = -sign(@E/@w_ij(t)) * u_ij(t)
                                 *
                                 * w_ij = w_ij + DW_ij
                                 */
                                weight_delta += -sign(p->gradients->elements[i][j])
                                        * p->dyn_learning_pars->elements[i][j];
                                p->weights->elements[i][j] += weight_delta;
                        }

                        /********************************/

                        n->status->weight_cost +=
                                square(p->weights->elements[i][j]);
                        n->status->gradient_linearity -= 
                                p->prev_weight_deltas->elements[i][j]
                                * p->gradients->elements[i][j];
                        n->status->last_weight_deltas_length +=
                                square(p->prev_weight_deltas->elements[i][j]);
                        n->status->gradients_length +=
                                square(p->gradients->elements[i][j]);

                        /********************************/

                        /* 
                         * Store a copy of the weight change.
                         */
                        p->prev_weight_deltas->elements[i][j] = weight_delta;
                }
        }
}

/*
 * ########################################################################
 * ## Quickprop weight updating                                          ##
 * ########################################################################
 */

/*
 * This implements Quickprop backpropagation (Fahlman, 1988). Quickprop is
 * a second order learning method that draws upon two assumptions:
 *
 *     (1) The error versus weight curve for each weight can be approximated
 *         by a parabola whose arms open upwards.
 *
 *     (2) The change in the error gradient, as seen by each weight is not
 *         affected by all the other weights that are changing at the same
 *         time.
 *
 * For each weight, previous and current error gradients, as well as the
 * weight deltas at the timesteps at which these gradients were measured are
 * used to determine a parabola. On each update, weights are adjusted to
 * jump to the minimum of this parabola:
 *
 *     DW_ij(t) = EW_ij(t) / (EW_ij(t-1) - EW(t)) * DW_ij(t-1)
 *
 * At t=0, this process is bootstrapped by using steepest descent, which is
 * is also used in case a previous weight delta equals 0. Weight updates are
 * bounded by a max step size u. If a weight step is larger than u times the
 * previous step for that weight, u times the previous weight delta is used
 * instead. Moreover, the negative of the learning rate times the current
 * error gradient is included in the current weight delta if the gradient
 * has the same sign as the previous gradient, and weight decay is applied
 * to limit the sizes of the weights.
 *
 * References
 *
 * Fahlman, S. E. (1988). An empirical study of learning speed in back-
 *     propagation networks. Technical report CMU-CS-88-162. School of
 *     Computer Science, Caernie Mellon University, Pittsburgh, PA 15213.
 */

#define QP_MAX_STEP_SIZE 1.75

void bp_update_qprop(struct network *n)
{
        n->status->weight_cost = 0.0;
        n->status->gradient_linearity = 0.0;
        n->status->last_weight_deltas_length = 0.0;
        n->status->gradients_length = 0.0;

        bp_recursively_update_qprop(n, n->output);

        n->status->gradient_linearity /=
                sqrt(n->status->last_weight_deltas_length
                                * n->status->gradients_length);
}

void bp_recursively_update_qprop(struct network *n, struct group *g)
{
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *p = g->inc_projs->elements[i];
                /*
                 * Adjust weights if projection is not frozen.
                 */
                if (!p->frozen)
                        bp_update_projection_qprop(n, g, p);
                
                /*
                 * Make a copy of the weight gradients, and reset the
                 * current weight gradients.
                 */
                copy_matrix(p->prev_gradients, p->gradients);
                zero_out_matrix(p->gradients);

                /*
                 * During BPTT, we want to only adjust weights
                 * in the network of the current timestep.
                 */
                if (p->recurrent)
                        continue;

                bp_recursively_update_qprop(n, p->to);
        }
}

void bp_update_projection_qprop(struct network *n, struct group *g,
                struct projection *p)
{
        double shrink_factor = QP_MAX_STEP_SIZE / (1.0 + QP_MAX_STEP_SIZE);

        /*
         * Adjust the weight between unit i in group g'
         * and unit j in group g.
         */
        for (int i = 0; i < p->to->vector->size; i++) {
                for (int j = 0; j < g->vector->size; j++) {
                        double weight_delta = 0.0;
                        
                        /*
                         * Previous weight delta was positive:
                         *
                         * DW_ij(t-1) > 0
                         */
                        if (p->prev_weight_deltas->elements[i][j] > 0.0) {
                                /*
                                 * If current gradient is negative, include
                                 * the negative of the learning rate times
                                 * the gradient in the weight delta.
                                 */ 
                                if (p->gradients->elements[i][j] < 0.0)
                                        weight_delta += -n->learning_rate
                                                * p->gradients->elements[i][j];
                                
                                /*
                                 * If current gradient is smaller than the
                                 * max step size times the previous
                                 * gradient, take a step of the maximum size
                                 * times the previous weight delta.
                                 */
                                if (p->gradients->elements[i][j] <
                                                shrink_factor * p->prev_gradients->elements[i][j]) {
                                        weight_delta += QP_MAX_STEP_SIZE
                                                * p->prev_weight_deltas->elements[i][j];
                                /*
                                 * Otherwise, use the quadratic estimate:
                                 *
                                 * @DW_ij(t) = EW_ij(t) 
                                 *     / (EW_ij(t-1) - EW_ij(t)
                                 *     * @DW_ij(t-1)
                                 */
                                } else {
                                        weight_delta += p->gradients->elements[i][j]
                                                / (p->prev_gradients->elements[i][j] 
                                                                - p->gradients->elements[i][j])
                                                * p->prev_weight_deltas->elements[i][j];
                                }
                        
                        /*
                         * Previous weight delta was negative:
                         *
                         * DW_ij(t-1) < 0
                         */
                        } else if (p->prev_weight_deltas->elements[i][j] < 0.0) {
                                /*
                                 * If current gradient is positive, include
                                 * the negative of the learning rate times
                                 * the gradient in the weight delta.
                                 */
                                if (p->gradients->elements[i][j] > 0.0)
                                        weight_delta += -n->learning_rate
                                                * p->gradients->elements[i][j];

                                /*
                                 * If current gradient is larger than the
                                 * max step size times the previous
                                 * gradient, take a step of the maximum size
                                 * times the previous weight delta.
                                 */
                                if (p->gradients->elements[i][j] >
                                                shrink_factor * p->prev_gradients->elements[i][j]) {
                                        weight_delta += QP_MAX_STEP_SIZE
                                                * p->prev_weight_deltas->elements[i][j];
                                /*
                                 * Otherwise, use the quadratic estimate:
                                 *
                                 * @DW_ij(t) = EW_ij(t) 
                                 *     / (EW_ij(t-1) - EW_ij(t)
                                 *     * @DW_ij(t-1)
                                 */
                                } else {
                                        weight_delta += p->gradients->elements[i][j]
                                                / (p->prev_gradients->elements[i][j] 
                                                                - p->gradients->elements[i][j])
                                                * p->prev_weight_deltas->elements[i][j];
                                }

                        /*
                         * Previous weight delta was zero:
                         *
                         * DW_ij(t-1) = 0
                         */
                        } else {
                                /*
                                 * Use steepest descent.
                                 *
                                 * First, we apply learning:
                                 *
                                 * DW_ij = -e * EW_ij
                                 */
                                weight_delta += -n->learning_rate
                                        * p->gradients->elements[i][j];

                                /*
                                 * Next, we apply momentum:
                                 *
                                 * DW_ij = DW_ij + a * DW_ij(t-1)
                                 */
                                weight_delta += n->momentum
                                        * p->prev_weight_deltas->elements[i][j];
                        }

                        /*
                         * Apply weight decay:
                         *
                         * DW_ij = DW_ij - d * W_ij
                         */
                        weight_delta -= n->weight_decay 
                                * p->weights->elements[i][j];

                        /*
                         * Adjust the weight:
                         *
                         * w_ij = w_ij + DW_ij
                         */
                        p->weights->elements[i][j] += weight_delta;

                        /********************************/

                        n->status->weight_cost +=
                                square(p->weights->elements[i][j]);
                        n->status->gradient_linearity -= 
                                p->prev_weight_deltas->elements[i][j]
                                * p->gradients->elements[i][j];
                        n->status->last_weight_deltas_length +=
                                square(p->prev_weight_deltas->elements[i][j]);
                        n->status->gradients_length +=
                                square(p->gradients->elements[i][j]);

                        /********************************/

                        /* 
                         * Store a copy of the weight change.
                         */
                        p->prev_weight_deltas->elements[i][j] = weight_delta;
                }
        }
}

/*
 * ########################################################################
 * ## Delta-Bar-Delta weight updating                                    ##
 * ########################################################################
 */

/*
 * This implements Delta-Bar-Delta (DBD) backpropagation (Jacobs, 1988). In
 * DBD backpropagation, each weight has its own learning rate that is
 * updated together with its corresponding weight. Hence, in essence, DBD
 * adds a learning rate update rule to steepest descent. Upon each update,
 * the change in the update rule for weight is defined as:
 *
 *                | kappa            , if EW_ij_bar(t-1) * EW_ij(t) > 0
 *                |
 *     De_ij(t) = | -phi * e_ij(t)   , if EW_ij_bar(t-1) * EW_ij(t) < 0
 *                |
 *                | 0                , otherwise
 *
 * where EW_ij_bar(t) is the exponential average of the current and past
 * gradients, which is defined as:
 *
 *     EW_ij_bar(t) = (1 - theta) * EW_ij + theta * EW_ij_bar(t-1)
 *
 * and has theta as its base, and time as its exponent. Hence, if the
 * current gradient and the average of the past gradients have the same sign,
 * the learning rate is incremented by kappa. If they have opposite signs,
 * by contrast, the learning rate is decremented by phi times its current
 * value.
 
 * References
 *
 * Jacobs, R. A. (1988). Increased Rates of Convergence Through Learning
 *     Rate Adapation. Neural Networks, 1, pp. 295-307.
 */

#define DBD_BASE 0.7

void bp_update_dbd(struct network *n)
{
        n->status->weight_cost = 0.0;
        n->status->gradient_linearity = 0.0;
        n->status->last_weight_deltas_length = 0.0;
        n->status->gradients_length = 0.0;

        n->dbd_rate_increment = 0.1;
        n->dbd_rate_decrement = 0.9;

        bp_recursively_update_dbd(n, n->output);

        n->status->gradient_linearity /=
                sqrt(n->status->last_weight_deltas_length
                                * n->status->gradients_length);
}

/*
 * This recursively adjusts the weights and their learning rates of all 
 * incoming projections of a group g.
 */

void bp_recursively_update_dbd(struct network *n, struct group *g)
{
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *p = g->inc_projs->elements[i];
                /*
                 * Adjust weights if projection is not frozen.
                 */
                if (!p->frozen)
                        bp_update_projection_dbd(n, g, p);
                
                /*
                 * Reset the current weight gradients.
                 */
                zero_out_matrix(p->gradients);

                /*
                 * During BPTT, we want to only adjust weights
                 * in the network of the current timestep.
                 */
                if (p->recurrent)
                        continue;

                bp_recursively_update_dbd(n, p->to);
        }
}

/*
 * This adjusts the weights and their learning rates of a projection p 
 * between a group g' and g.
 */

void bp_update_projection_dbd(struct network *n, struct group *g,
                struct projection *p)
{
        /*
         * Adjust the weight and its learning rate between unit i in group g'
         * and unit j in group g.
         */
        for (int i = 0; i < p->to->vector->size; i++) {
                for (int j = 0; j < g->vector->size; j++) {

                        /*
                         * ################################################
                         * ## Update weight                              ##
                         * ################################################
                         */

                        double weight_delta = 0.0;

                        /*
                         * First, we apply learning:
                         *
                         * DW_ij = -e * EW_ij
                         */
                        weight_delta += -p->dyn_learning_pars->elements[i][j]
                                * p->gradients->elements[i][j];

                        /*
                         * Next, we apply momentum:
                         *
                         * DW_ij = DW_ij + a * DW_ij(t-1)
                         */
                        weight_delta += n->momentum
                                * p->prev_weight_deltas->elements[i][j];
                        
                        /*
                         * Finally, we apply weight decay:
                         *
                         * DW_ij = DW_ij - d * W_ij
                         */
                        weight_delta -= n->weight_decay 
                                * p->weights->elements[i][j];

                        /*
                         * Adjust the weight:
                         *
                         * w_ij = w_ij + DW_ij
                         */
                        p->weights->elements[i][j] += weight_delta;
                        
                        /********************************/

                        n->status->weight_cost +=
                                pow(p->weights->elements[i][j], 2.0);
                        n->status->gradient_linearity -= 
                                p->prev_weight_deltas->elements[i][j]
                                * p->gradients->elements[i][j];
                        n->status->last_weight_deltas_length +=
                                pow(p->prev_weight_deltas->elements[i][j], 2.0);
                        n->status->gradients_length +=
                                pow(p->gradients->elements[i][j], 2.0);

                        /********************************/

                        /* 
                         * Store a copy of the weight change.
                         */
                        p->prev_weight_deltas->elements[i][j] = weight_delta;

                        /*
                         * ################################################
                         * ## Update learning rate                       ##
                         * ################################################
                         */

                        double lr_delta = 0.0;

                        /*
                         * Current gradient and average of past gradients
                         * have same sign:
                         *
                         * EW_ij_bar(t-1) * EW_ij(t) > 0
                         *
                         * Note: EW_ij_bar(t-1) is stored in the
                         * prev_gradients matrix of the projection.
                         */
                        if (p->prev_gradients->elements[i][j]
                                        * p->gradients->elements[i][j] > 0.0) {
                                /*
                                 * De_ij = kappa
                                 */
                                lr_delta = n->dbd_rate_increment;

                        /*
                         * Current gradient and average of past gradients
                         * have opposite sign:
                         *
                         * EW_ij_bar(t-1) * EW_ij(t) < 0
                         */
                        } else if (p->prev_gradients->elements[i][j]
                                        * p->gradients->elements[i][j] < 0.0) {
                                /*
                                 * De_ij = -phi * e_ij(t)
                                 */
                                lr_delta = -n->dbd_rate_decrement
                                        * p->dyn_learning_pars->elements[i][j];
                        }

                        /*
                         * Adjust the learning rate:
                         *
                         * e_ij = e_ij + De_ij
                         */
                        p->dyn_learning_pars->elements[i][j] += lr_delta;

                        /*
                         * Determine the exponential average of the current
                         * and past gradients:
                         * 
                         * EW_ij_bar(t) = (1 - theta) * EW_ij
                         *     + theta * EW_ij_bar(t-1)
                         */
                        double exp_average = (1.0 - DBD_BASE)
                                * p->gradients->elements[i][j]
                                + DBD_BASE * p->prev_gradients->elements[i][i];

                        /*
                         * Store a copy of the current exponentional 
                         * average.
                         */
                        p->prev_gradients->elements[i][j] = exp_average;
                }
        }
}
