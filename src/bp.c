/*
 * Copyright 2012-2018 Harm Brouwer <me@hbrouwer.eu>
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

#include "act.h"
#include "bp.h"
#include "error.h"
#include "main.h"
#include "math.h"

                /*************************
                 **** backpropagation ****
                 *************************/

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
This implements the backpropagation (BP) algorithm (Rumelhart, Hinton, &
Williams, 1986). BP minimizes an error function over a finite set of
input-output pairs by means of gradient descent. Typically, the error
function that is minimized is the sum squared error:

        E = 1/2 sum_c sum_j (y_j,c - d_j,c)^2

where y_j,c is the observed activity of output unit j for input-output pair
c, and d_j,c the desired activity for this unit. For each input- output pair
c, BP operates in two passes. In the forward pass, the network's response to
the input pattern is computed. In the subsequent backward pass, BP employs
the generalized delta rule to reduce the network's error E by adjusting each
of the network's weights proportional to its gradient:

        Dw_ij = -epsilon dE/dw_ij

where epsilon is the network's learning rate. The gradient of a weight w_ij,
in turn, is defined as:

        dE/dw_ij = delta_j * y_i

where delta_j is the error signal of unit j, and y_i is the activation value
of unit i that signals to unit j. The error signal of a unit is the product
of that unit's error derivative and its activation derivative:

        delta_j = dE/dy_j * dy_j/dx_j = dE/d_yj * f'(x_j)

where f'(x) is the derivative of the activation function f(x) for the net
input x_j to unit j. If unit j is a unit in the output layer, this means
that its error signal delta_j is defined as:

        delta_j = (y_j - d_j)f'(x_j)

If unit j is a hidden unit, by contrast, its error signal delta_j is
recursively defined as:

        delta_j = f'(x_j) sum_k delta_k w_jk

where all units k are units that receive signals from unit j.

References

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning
        representations by back-propagating errors. Nature, 323, 553-536.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

/*
 * This computes the error signal delta_j for each output unit j.
 */
void bp_output_error(struct group *g, struct vector *t, double tr,
        double zr)
{
        /*
         * First, compute error derivates dE/dy for all units in the output
         * layer.
         */
        g->err_fun->deriv(g, t, tr, zr);

        /*
         * Multiply all error derivatives dE/dy with the activation function
         * derivative f'(x_j) to obtain the error signal for unit j.
         */
        for (uint32_t i = 0; i < g->error->size; i++) {
                double act_deriv = g->act_fun->deriv(g->vector, i);
                if (g->act_fun->fun == act_fun_binary_sigmoid)
                        act_deriv += BP_FLAT_SPOT_CORRECTION;
                g->error->elements[i] *= act_deriv;
        }
}

/*
 * This is the main BP function. Provided a group g, it computes the error
 * signals for each group g' that projects to g, as well as the gradients
 * for the weights on the projection between these groups. Next, it
 * recursively propagates these error signals to earlier groups.
 */
void bp_backpropagate_error(struct network *n, struct group *g)
{
        /*
         * Note: Each group g' that projects to g can receive error signals
         * from more than one later groups (for instance in backpropagation
         * through time). If this is the case the error derivative of
         * dE/dy_j of a unit j, is then defined as:
         *
         *     dE/dy_j = sum_g'' sum_k delta_k w_jk
         *
         * where all groups g'' are groups to which g' projects.
         */
        for (uint32_t  i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *ip = g->inc_projs->elements[i];
                struct group *ng = ip->to;

                for (uint32_t  j = 0; j < ng->out_projs->num_elements; j++) {
                        struct projection *p = ng->out_projs->elements[j];

#ifdef _OPENMP
#pragma omp parallel for
#endif /* _OPENMP */
                        for (uint32_t  x = 0; x < ng->error->size; x++) {
                                for (uint32_t  z = 0; z < p->to->vector->size; z++) {
                                        /*
                                         * Compute the error derivative:
                                         *
                                         * dE/dy_j += sum_k delta_k w_jk
                                         */
                                        ng->error->elements[x] += p->to->error->elements[z]
                                                * p->weights->elements[x][z];

                                        /*
                                         * We only compute gradients for
                                         * projections to g:
                                         *
                                         * 0
                                         * |
                                         * 1   3
                                         * | \ |
                                         * 2   4   .
                                         *     | \ |
                                         *     5   7
                                         *         |
                                         *         . 
                                         *
                                         * If the current group is 1, we
                                         * compute the gradients for the
                                         * projection between 1 and 2, and
                                         * the one between 1 and 4. If the
                                         * current group is 4, we compute
                                         * the gradients of the projection
                                         * between 4 and 5, and the one
                                         * between 4 and 7, and so forth.
                                         */
                                        if (p->to != g)
                                                continue;

                                        /*
                                         * Compute the weight gradient:
                                         *
                                         * dE/dw_ij += delta_j * y_i
                                         *
                                         * Note: gradients may sum over an
                                         * epoch.
                                         */
                                        p->gradients->elements[x][z] += p->to->error->elements[z]
                                                * ng->vector->elements[x];
                                }
                        }
                }

                /*
                 * Multiply each error derivative with its relevant
                 * activation derivative to get the error signal:
                 *
                 * delta_j = f'(x_j) dE/dy_j
                 */
#ifdef _OPENMP
#pragma omp parallel for
#endif /* _OPENMP */
                for (uint32_t  x = 0; x < ng->error->size; x++) {
                        double act_deriv = ng->act_fun->deriv(ng->vector, x);
                        if (g->act_fun->fun == act_fun_binary_sigmoid)
                                act_deriv += BP_FLAT_SPOT_CORRECTION;
                        ng->error->elements[x] *= act_deriv;
                }
        }

        /*
         * Recursively backpropagate error.
         */
        for (uint32_t  i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *ip = g->inc_projs->elements[i];
                bp_backpropagate_error(n, ip->to);
        }
}

                /**************************
                 **** steepest descent ****
                 **************************/

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
This implements steepest (or gradient) descent backpropagation. Steepest
descent is a first-order optimization algorithm for finding the nearest
local minimum of a function. On each weight update, a step is taken that is
proportional to the negative of the gradient of the function that is being
minimized.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

void bp_update_sd(struct network *n)
{
        /* reset status statistics */
        n->status->weight_cost        = 0.0;
        n->status->gradient_linearity = 0.0;
        n->status->last_deltas_length = 0.0;
        n->status->gradients_length   = 0.0;

        /* determine the scaling factor for steepest descent */
        if (n->sd_type == SD_DEFAULT)
                n->sd_scale_factor = 1.0;
        if (n->sd_type == SD_BOUNDED)
                determine_sd_scale_factor(n);

        bp_update_inc_projs_sd(n, n->output);

        /*
         * Compute gradient linearity:
         *
         *         sum_i sum_j (Dw_ij(t-1) * dE/dw_ij)
         * gl = -( ----------------------------------- )
         *         sqrt(sum_i sum_j (Dw_ij(t-1) ^ 2))
         *         * sqrt(sum_i sum_j (dE/dw_ij ^ 2))
         */
        n->status->gradient_linearity = -(n->status->gradient_linearity
                / sqrt(n->status->last_deltas_length
                        * n->status->gradients_length));
}

/*
 * Recursively adjusts the weights of all incoming projections of a group g.
 */
void bp_update_inc_projs_sd(struct network *n, struct group *g)
{
        for (uint32_t  i = 0; i < g->inc_projs->num_elements; i++) {
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
                 * During BPTT, we want to only adjust weights in the
                 * network of the current timestep.
                 */
                if (p->recurrent)
                        continue;

                bp_update_inc_projs_sd(n, p->to);
        }
}

/*
 * This adjusts the weights of a projection p between a group g' and g.
 */
void bp_update_projection_sd(struct network *n, struct group *g,
        struct projection *p)
{
        /* local status statistics */
        double weight_cost        = 0.0;
        double gradient_linearity = 0.0;
        double last_deltas_length = 0.0;
        double gradients_length   = 0.0;

        /*
         * Adjust the weight between unit i in group g' and unit j in group
         * g.
         */
#ifdef _OPENMP
#pragma omp parallel for reduction(+:weight_cost, gradient_linearity, last_deltas_length, gradients_length)
#endif /* _OPENMP */
        for (uint32_t  i = 0; i < p->to->vector->size; i++) {
                for (uint32_t  j = 0; j < g->vector->size; j++) {
                        double weight_delta = 0.0;

                        /*
                         * First, we apply learning:
                         *
                         * Dw_ij = -epsilon * dE/dw_ij
                         *
                         * Note: If bounded steepest descent is used, the
                         * gradient term is scaled by the length of the
                         * gradient.
                         */
                        weight_delta += -n->learning_rate
                                * n->sd_scale_factor
                                * p->gradients->elements[i][j];

                        /*
                         * Next, we apply momentum:
                         *
                         * Dw_ij = Dw_ij + a * Dw_ij(t-1)
                         */
                        weight_delta += n->momentum
                                * p->prev_deltas->elements[i][j];
                        
                        /*
                         * Finally, we apply weight decay:
                         *
                         * Dw_ij = Dw_ij - d * w_ij
                         */
                        weight_delta -= n->weight_decay 
                                * p->weights->elements[i][j];

                        /*
                         * Adjust the weight:
                         *
                         * w_ij = w_ij + Dw_ij
                         */
                        p->weights->elements[i][j] += weight_delta;
                        
                        /*
                         * Compute weight cost:
                         *
                         * wc = sum_i sum_j (w_ij ^ 2)
                         */
                        weight_cost += pow(p->weights->elements[i][j], 2.0);
                        
                        /*
                         * Compute the numerator of the gradient linearity:
                         *
                         * sum_i sum_j (Dw_ij(t-1) * dE/dw_ij)
                         */
                        gradient_linearity +=
                                p->prev_deltas->elements[i][j]
                                * p->gradients->elements[i][j];
                        
                        /*
                         * Compute the sum of squares of the previous weight
                         * delta vector:
                         *
                         * sum_i sum_j (Dw_ij(t-1) ^ 2
                         */
                        last_deltas_length +=
                                 pow(p->prev_deltas->elements[i][j], 2.0);

                        /*
                         * Compute the sum of squares of the gradients:
                         *
                         * sum_i sum_j (dE/dw_ij ^ 2)
                         */
                        gradients_length +=
                                pow(p->gradients->elements[i][j], 2.0);

                        /* 
                         * Store a copy of the weight change.
                         */
                        p->prev_deltas->elements[i][j] = weight_delta;
                }
        }

        /*
         * Add the local status statistics to the global status statistics.
         */
        n->status->weight_cost        += weight_cost;
        n->status->gradient_linearity += gradient_linearity;
        n->status->last_deltas_length += last_deltas_length;
        n->status->gradients_length   += gradients_length;
}

                /**********************************
                 **** bounded steepest descent ****
                 **********************************/

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Determine the scaling factor for steepest descent. If "bounded" steepest
descent (Rohde, 2002) is used instead of "default" steepest descent, we
scale the gradient term of the weight delta by the length of the gradient if
this length is greater than 1.0. Otherwise, we simply take the product of
the negative learning rate and the gradient by setting the scaling factor to
1.0:

             | 1.0 / ||dE/dw|| , if ||dE/dw|| > 1.0
        sf = |
             | 1.0             , otherwise
           
        where sf is the scaling factor.

Rohde, D. L. T. (2002). A connectionist model of sentence comprehension and
        production. PhD thesis, Carnegie Mellon University.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

void determine_sd_scale_factor(struct network *n)
{
        /* reset scaling factor */
        n->sd_scale_factor = 0.0;

        /* 
         * Resursively compute the sum of squares of the individual weight
         * gradients.
         */
        determine_gradient_ssq(n, n->output);

        /* determine the scaling factor */
        if (n->sd_scale_factor > 1.0)
                n->sd_scale_factor = 1.0 / sqrt(n->sd_scale_factor);
        else
                n->sd_scale_factor = 1.0;
}

/*
 * Resursively compute the sum of squares of the individual weight
 * gradients.
 */
void determine_gradient_ssq(struct network *n, struct group *g)
{
        /* local scale factor */
        double sd_scale_factor = 0.0;

        for (uint32_t  i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *p = g->inc_projs->elements[i];

                /* sum gradients */
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sd_scale_factor)
#endif /* _OPENMP */
                for (uint32_t  j = 0; j < p->to->vector->size; j++)
                        for (uint32_t  x = 0; x < g->vector->size; x++)
                                sd_scale_factor +=
                                        pow(p->gradients->elements[j][x], 2.0);
                
                determine_gradient_ssq(n, p->to);
        }

        /* add local scale factor to global scale factor */
        n->sd_scale_factor += sd_scale_factor;
}

                /***********************************
                 **** resilient backpropagation ****
                 ***********************************/

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
This implements resilient backpropagation (Rprop) (see Igel & Husken, 2000).
In Rprop, weight adjustments are made on the basis of the sign of the
gradient of a weight. An Rprop update iteration can be divided into two
stages. In the first stage, the "update value" u_ij for each weight w_ij is
computed:

                  | eta_plus * u_ij(t-1),  if dE/dw_ij(t-1) * dE/dw_ij(t) > 0
                  |
        u_ij(t) = | eta_minus * u_ij(t-1), if dE/dw_ij(t-1) * dE/dw_ij(t) < 0
                  |
                  | u_ij(t-1)            , otherwise

where eta_plus and eta_minus are defined as:

        0 < eta_minus < 1 < eta_plus.

and u_ij(t) is bounded by u_max and u_min. The second stage of an Rprop
iteration depends on the particular Rprop flavour. Four Rprop flavours are
implemented (see Igel & Husken, 2000):

(1) RPROP+ (Rprop with weight-backtracking)

After computing the "update value" u_ij for each weight w_ij, the second
stage depends on whether the sign of gradient of that weight has changed
from timestep t-1 to t. If it has not changed, we perform a regular weight
update:

        if dE/dw_ij(t-1) > dE/dw_ij(t) > 0 then

                Dw_ij(t) = -sign(dE/dw_ij(t)) * u_ij(t)

where sign(x) returns +1 if x is positive and -1 if x is negative. If, on
the other hand, the sign has changed, we revert the previous weight update
(weight backtracking), and reset the gradient dE/dw_ij(t) to 0, so that the
u_ij will not be adjusted on the next iteration:

        if dE/dw_ij(t-1) > dE/dw_ij(t) > 0 then
 
                Dw_ij(t) = -Dw_ij(t-1)
 
                dE/dw_ij(t) = 0
 
Finally, weights are updated by means of: 

        w_ij = w_ij + u_ij(t)
 
(2) RPROP- (Rprop without weight-backtracking)
 
A variation on RPROP+ in which weight backtracking is omitted, and in which
dE/dw_ij(t) is not reset to 0 when its sign has changed.
 
(3) iRPROP+ ("modified" Rprop with weight-backtracking)
 
A variation on RPROP+ in which weight backtracking is only performed if the
overall error goes up from timestep t-1 to t.
 
(4) iRPROP- ("modified" Rprop without weight-backtracking)

A variation on RPROP- in which dE/dw_ij(t) is reset to 0 when its sign has
changed.

References

Igel, C., & Husken, M. (2000). Improving the Rprop Algorithm. Proceedings of
        the Second International Symposium on Neural Computation, NC'2000,
        pp. 115-121, ICSC, Academic Press, 2000.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#define RP_MAX_STEP_SIZE 50.0
#define RP_MIN_STEP_SIZE 1e-6

void bp_update_rprop(struct network *n)
{
        /* reset status statistics */
        n->status->weight_cost        = 0.0;
        n->status->gradient_linearity = 0.0;
        n->status->last_deltas_length = 0.0;
        n->status->gradients_length   = 0.0;

        bp_update_inc_projs_rprop(n, n->output);

        /*
         * Compute gradient linearity:
         *
         *         sum_i sum_j (Dw_ij(t-1) * dE/dw_ij)
         * gl = -( ----------------------------------- )
         *         sqrt(sum_i sum_j (Dw_ij(t-1) ^ 2))
         *         * sqrt(sum_i sum_j (dE/dw_ij ^ 2))
         */
        n->status->gradient_linearity = -(n->status->gradient_linearity
                / sqrt(n->status->last_deltas_length
                        * n->status->gradients_length));
}

/*
 * Recursively adjusts the weights of all incoming projections of a group g.
 */
void bp_update_inc_projs_rprop(struct network *n, struct group *g)
{
        for (uint32_t i = 0; i < g->inc_projs->num_elements; i++) {
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
                 * During BPTT, we want to only adjust weights in the
                 * network of the current timestep.
                 */
                if (p->recurrent)
                        continue;

                bp_update_inc_projs_rprop(n, p->to);
        }
}

/*
 * This adjusts the weights of a projection p between a group g' and g.
 */
void bp_update_projection_rprop(struct network *n, struct group *g,
        struct projection *p)
{
        /* local status statistics */
        double weight_cost        = 0.0;
        double gradient_linearity = 0.0;
        double last_deltas_length = 0.0;
        double gradients_length   = 0.0;

        /*
         * Adjust the weight between unit i in group g' and unit j in group
         * g.
         */
#ifdef _OPENMP
#pragma omp parallel for reduction(+:weight_cost, gradient_linearity, last_deltas_length, gradients_length)
#endif /* _OPENMP */
        for (uint32_t  i = 0; i < p->to->vector->size; i++) {
                for (uint32_t  j = 0; j < g->vector->size; j++) {
                        double weight_delta = 0.0;

                        /*
                         * First, apply weight decay:
                         *
                         * Dw_ij = Dw_ij - d * w_ij
                         */
                        weight_delta -= n->weight_decay 
                                * p->weights->elements[i][j];

                        /*
                         * Sign of dE/dw_ij has not changed:
                         * 
                         * dE/dw_ij(t-1) * dE/dw_ij(t) > 0
                         */
                        if (p->prev_gradients->elements[i][j]
                                * p->gradients->elements[i][j] > 0.0) {

                                /*
                                 * Bind update value u_ij to u_max.
                                 */
                                p->dynamic_params->elements[i][j] = minimum(
                                        p->dynamic_params->elements[i][j] * n->rp_eta_plus,
                                        RP_MAX_STEP_SIZE);

                                /*
                                 * Perform weight update:
                                 *
                                 * Dw_ij = -sign(dE/dw_ij(t)) * u_ij(t)
                                 *
                                 * w_ij = w_ij + Dw_ij
                                 */
                                weight_delta += -sign(p->gradients->elements[i][j]) 
                                        * p->dynamic_params->elements[i][j];
                                p->weights->elements[i][j] += weight_delta;

                        /*
                         * Sign of dE/dw_ij has changed:
                         * 
                         * dE/dw_ij(t-1) * dE/dw_ij(t) < 0
                         */
                        } else if (p->prev_gradients->elements[i][j]
                                * p->gradients->elements[i][j] < 0.0) {

                                /*
                                 * Bind update value u_ij to u_min.
                                 */
                                p->dynamic_params->elements[i][j] = maximum(
                                        p->dynamic_params->elements[i][j] * n->rp_eta_minus,
                                        RP_MIN_STEP_SIZE);

                                /*
                                 * Perform weight backtracking for RPROP+.
                                 */
                                if (n->rp_type == RPROP_PLUS)
                                        p->weights->elements[i][j] -=
                                                p->prev_deltas->elements[i][j];

                                /*
                                 * Perform weight backtracking for iRPROP+.
                                 */
                                if (n->rp_type == IRPROP_PLUS)
                                        if (n->status->error > n->status->prev_error)
                                                p->weights->elements[i][j] -=
                                                        p->prev_deltas->elements[i][j];

                                /*
                                 * Set dE/dw_ij(t) to 0 for all Rprop
                                 * flavours except RPROP-.
                                 */
                                if (n->rp_type != RPROP_MINUS)
                                        p->gradients->elements[i][j] = 0.0;

                                /* 
                                 * Perform weight change for RPROP- and
                                 * iRPROP-:
                                 * 
                                 * Dw_ij = -sign(dE/dw_ij(t)) * u_ij(t)
                                 *
                                 * w_ij = w_ij + Dw_ij
                                 */
                                if (n->rp_type == RPROP_MINUS || n->rp_type == IRPROP_MINUS) {
                                        weight_delta += -sign(p->gradients->elements[i][j]) *
                                                p->dynamic_params->elements[i][j];
                                        p->weights->elements[i][j] += weight_delta;
                                }

                        /*
                         * Otherwise:
                         *
                         * dE/dw_ij(t-1) * dE/dw_ij(t) = 0
                         */
                        } else if (p->prev_gradients->elements[i][j]
                                        * p->gradients->elements[i][j] == 0.0) {
                                /*
                                 * Perform weight update:
                                 *
                                 * Dw_ij = -sign(dE/dw_ij(t)) * u_ij(t)
                                 *
                                 * w_ij = w_ij + Dw_ij
                                 */
                                weight_delta += -sign(p->gradients->elements[i][j])
                                        * p->dynamic_params->elements[i][j];
                                p->weights->elements[i][j] += weight_delta;
                        }

                        /*
                         * Compute weight cost:
                         *
                         * wc = sum_i sum_j (w_ij ^ 2)
                         */
                        weight_cost += pow(p->weights->elements[i][j], 2.0);
                        
                        /*
                         * Compute the numerator of the
                         * gradient linearity:
                         *
                         * sum_i sum_j (Dw_ij(t-1) * dE/dw_ij)
                         */
                        gradient_linearity +=
                                p->prev_deltas->elements[i][j]
                                * p->gradients->elements[i][j];
                        
                        /*
                         * Compute the sum of squares of the
                         * previous weight delta vector:
                         *
                         * sum_i sum_j (Dw_ij(t-1) ^ 2
                         */
                        last_deltas_length +=
                                 pow(p->prev_deltas->elements[i][j], 2.0);

                        /*
                         * Compute the sum of squares of the
                         * gradients:
                         *
                         * sum_i sum_j (dE/dw_ij ^ 2)
                         */
                        gradients_length +=
                                pow(p->gradients->elements[i][j], 2.0);

                        /* 
                         * Store a copy of the weight change.
                         */
                        p->prev_deltas->elements[i][j] = weight_delta;
                }
        }
        
        /*
         * Add the local status statistics to the global status statistics.
         */
        n->status->weight_cost        += weight_cost;
        n->status->gradient_linearity += gradient_linearity;
        n->status->last_deltas_length += last_deltas_length;
        n->status->gradients_length   += gradients_length;
}

                /***********************************
                 **** quickprop backpropagation ****
                 ***********************************/

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
This implements Quickprop backpropagation (Fahlman, 1988). Quickprop is a
second order learning method that draws upon two assumptions:

        (1) The error versus weight curve for each weight can be approximated
        by a parabola whose arms open upwards.

        (2) The change in the error gradient, as seen by each weight is not
        affected by all the other weights that are changing at the same time.

For each weight, previous and current gradients, as well as the weight
deltas at the timesteps at which these gradients were measured are used to
determine a parabola. On each update, weights are adjusted to jump to the
minimum of this parabola:

        Dw_ij(t) = dE/dw_ij(t) / (dE/dw_ij(t-1) - dE/dw_ij(t)) * Dw_ij(t-1)
 
At t=0, this process is bootstrapped by using steepest descent, which is is
also used in case a previous weight delta equals 0. Weight updates are
bounded by a max step size u. If a weight step is larger than u times the
previous step for that weight, u times the previous weight delta is used
instead. Moreover, the negative of the learning rate times the current
gradient is included in the current weight delta if the gradient has the
same sign as the previous gradient, and weight decay is applied to limit the
sizes of the weights.

References

Fahlman, S. E. (1988). An empirical study of learning speed in back-
        propagation networks. Technical report CMU-CS-88-162. School of
        Computer Science, Carnegie Mellon University, Pittsburgh, PA 15213.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#define QP_MAX_STEP_SIZE 1.75

void bp_update_qprop(struct network *n)
{
        /* reset status statistics */
        n->status->weight_cost        = 0.0;
        n->status->gradient_linearity = 0.0;
        n->status->last_deltas_length = 0.0;
        n->status->gradients_length   = 0.0;

        bp_update_inc_projs_qprop(n, n->output);

        /*
         * Compute gradient linearity:
         *
         *         sum_i sum_j (Dw_ij(t-1) * dE/dw_ij)
         * gl = -( ----------------------------------- )
         *         sqrt(sum_i sum_j (Dw_ij(t-1) ^ 2))
         *         * sqrt(sum_i sum_j (dE/dw_ij ^ 2))
         */
        n->status->gradient_linearity = -(n->status->gradient_linearity
                / sqrt(n->status->last_deltas_length
                        * n->status->gradients_length));
}

/*
 * Recursively adjusts the weights of all incoming projections of a group g.
 */
void bp_update_inc_projs_qprop(struct network *n, struct group *g)
{
        for (uint32_t  i = 0; i < g->inc_projs->num_elements; i++) {
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
                 * During BPTT, we want to only adjust weights in the
                 * network of the current timestep.
                 */
                if (p->recurrent)
                        continue;

                bp_update_inc_projs_qprop(n, p->to);
        }
}

/*
 * This adjusts the weights of a projection p between a group g' and g.
 */
void bp_update_projection_qprop(struct network *n, struct group *g,
        struct projection *p)
{
        double shrink_factor = QP_MAX_STEP_SIZE / (1.0 + QP_MAX_STEP_SIZE);

        /* local status statistics */
        double weight_cost        = 0.0;
        double gradient_linearity = 0.0;
        double last_deltas_length = 0.0;
        double gradients_length   = 0.0;

        /*
         * Adjust the weight between unit i in group g' and unit j in group
         * g.
         */
#ifdef _OPENMP
#pragma omp parallel for reduction(+:weight_cost, gradient_linearity, last_deltas_length, gradients_length)
#endif /* _OPENMP */
        for (uint32_t  i = 0; i < p->to->vector->size; i++) {
                for (uint32_t  j = 0; j < g->vector->size; j++) {
                        double weight_delta = 0.0;
                        
                        /*
                         * Previous weight delta was positive:
                         *
                         * Dw_ij(t-1) > 0
                         */
                        if (p->prev_deltas->elements[i][j] > 0.0) {
                                /*
                                 * If current gradient is negative, include
                                 * a steepest descent term in the weight
                                 * delta:
                                 *
                                 * Dw_ij(t) = -epislon * dE/dw_ij
                                 */ 
                                if (p->gradients->elements[i][j] < 0.0)
                                        weight_delta += -n->learning_rate
                                                * p->gradients->elements[i][j];
                                
                                /*
                                 * If current gradient is smaller than the
                                 * max step size times the previous
                                 * gradient, take a step of the maximum size
                                 * times the previous weight delta:
                                 *
                                 * Dw_ij(t) = Dw_ij(t) + u * Dw(t-1)
                                 */
                                if (p->gradients->elements[i][j] <
                                        shrink_factor * p->prev_gradients->elements[i][j]) {
                                        weight_delta += QP_MAX_STEP_SIZE
                                                * p->prev_deltas->elements[i][j];
                                /*
                                 * Otherwise, use the quadratic estimate:
                                 *
                                 * Dw_ij(t) = Dw_ij(t) + dE/dw_ij(t)
                                 *     / (dE/dw_ij(t-1) - dE/dw_ij(t)
                                 *     * Dw_ij(t-1)
                                 */
                                } else {
                                        weight_delta += p->gradients->elements[i][j]
                                                / (p->prev_gradients->elements[i][j] 
                                                        - p->gradients->elements[i][j])
                                                * p->prev_deltas->elements[i][j];
                                }
                        
                        /*
                         * Previous weight delta was negative:
                         *
                         * Dw_ij(t-1) < 0
                         */
                        } else if (p->prev_deltas->elements[i][j] < 0.0) {
                                /*
                                 * If current gradient is positive, include
                                 * a steepest descent term in the weight
                                 * delta:
                                 *
                                 * Dw_ij(t) = -epislon * dE/dw_ij
                                 */
                                if (p->gradients->elements[i][j] > 0.0)
                                        weight_delta += -n->learning_rate
                                                * p->gradients->elements[i][j];

                                /*
                                 * If current gradient is larger than the
                                 * max step size times the previous
                                 * gradient, take a step of the maximum size
                                 * times the previous weight delta:
                                 *
                                 * Dw_ij(t) = Dw_ij(t) + u * Dw(t-1)
                                 */
                                if (p->gradients->elements[i][j] >
                                        shrink_factor * p->prev_gradients->elements[i][j]) {
                                        weight_delta += QP_MAX_STEP_SIZE
                                                * p->prev_deltas->elements[i][j];
                                /*
                                 * Otherwise, use the quadratic estimate:
                                 *
                                 * Dw_ij(t) = Dw_ij(t) + dE/dw_ij(t)
                                 *     / (dE/dw_ij(t-1) - dE/dw_ij(t)
                                 *     * @Dw_ij(t-1)
                                 */
                                } else {
                                        weight_delta += p->gradients->elements[i][j]
                                                / (p->prev_gradients->elements[i][j] 
                                                        - p->gradients->elements[i][j])
                                                * p->prev_deltas->elements[i][j];
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
                                 * Dw_ij = -epislon * dE/dw_ij
                                 */
                                weight_delta += -n->learning_rate
                                        * p->gradients->elements[i][j];

                                /*
                                 * Next, we apply momentum:
                                 *
                                 * Dw_ij = Dw_ij + a * Dw_ij(t-1)
                                 */
                                weight_delta += n->momentum
                                        * p->prev_deltas->elements[i][j];
                        }

                        /*
                         * Apply weight decay:
                         *
                         * Dw_ij = Dw_ij - d * w_ij
                         */
                        weight_delta -= n->weight_decay 
                                * p->weights->elements[i][j];

                        /*
                         * Adjust the weight:
                         *
                         * w_ij = w_ij + Dw_ij
                         */
                        p->weights->elements[i][j] += weight_delta;

                        /*
                         * Compute weight cost:
                         *
                         * wc = sum_i sum_j (w_ij ^ 2)
                         */
                        weight_cost += pow(p->weights->elements[i][j], 2.0);
                        
                        /*
                         * Compute the numerator of the gradient linearity:
                         *
                         * sum_i sum_j (Dw_ij(t-1) * dE/dw_ij)
                         */
                        gradient_linearity +=
                                p->prev_deltas->elements[i][j]
                                * p->gradients->elements[i][j];
                        
                        /*
                         * Compute the sum of squares of the previous weight
                         * delta vector:
                         *
                         * sum_i sum_j (Dw_ij(t-1) ^ 2
                         */
                        last_deltas_length +=
                                 pow(p->prev_deltas->elements[i][j], 2.0);

                        /*
                         * Compute the sum of squares of the gradients:
                         *
                         * sum_i sum_j (dE/dw_ij ^ 2)
                         */
                        gradients_length +=
                                pow(p->gradients->elements[i][j], 2.0);

                        /* 
                         * Store a copy of the weight change.
                         */
                        p->prev_deltas->elements[i][j] = weight_delta;
                }
        }

        /*
         * Add the local status statistics to the global status statistics.
         */
        n->status->weight_cost        += weight_cost;
        n->status->gradient_linearity += gradient_linearity;
        n->status->last_deltas_length += last_deltas_length;
        n->status->gradients_length   += gradients_length;
}

                /*****************************************
                 **** delta-bar-delta backpropagation ****
                 *****************************************/

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
This implements Delta-Bar-Delta (DBD) backpropagation (Jacobs, 1988). In DBD
backpropagation, each weight has its own learning rate that is updated
together with its corresponding weight. Hence, in essence, DBD adds a
learning rate update rule to steepest descent. Upon each update, the change
in the learning rate for a weight is defined as:

                   | kappa          , if dE/dw_ij_bar(t-1) * dE/dw_ij(t) > 0
                   |
        De_ij(t) = | -phi * e_ij(t) , if dE/dw_ij_bar(t-1) * dE/dw_ij(t) < 0
                   |
                   | 0              , otherwise

where dE/dw_ij_bar(t) is the exponential average of the current and past
gradients, which is defined as:

dE/dw_ij_bar(t) = (1 - theta) * dE/dw_ij + theta * dE/dw_ij_bar(t-1)

and has theta as its base, and time as its exponent. Hence, if the current
gradient and the average of the past gradients have the same sign, the
learning rate is incremented by kappa. If they have opposite signs, by
contrast, the learning rate is decremented by phi times its current value.
 
References

Jacobs, R. A. (1988). Increased Rates of Convergence Through Learning Rate
        Adapation. Neural Networks, 1, 295-307.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#define DBD_BASE 0.7

void bp_update_dbd(struct network *n)
{
        /* reset status statistics */
        n->status->weight_cost        = 0.0;
        n->status->gradient_linearity = 0.0;
        n->status->last_deltas_length = 0.0;
        n->status->gradients_length   = 0.0;

        /*
        n->dbd_rate_increment = 0.1;
        n->dbd_rate_decrement = 0.9;
        */

        bp_update_inc_projs_dbd(n, n->output);

        /*
         * Compute gradient linearity:
         *
         *         sum_i sum_j (Dw_ij(t-1) * dE/dw_ij)
         * gl = -( ----------------------------------- )
         *         sqrt(sum_i sum_j (Dw_ij(t-1) ^ 2))
         *         * sqrt(sum_i sum_j (dE/dw_ij ^ 2))
         */
        n->status->gradient_linearity = -(n->status->gradient_linearity
                / sqrt(n->status->last_deltas_length
                        * n->status->gradients_length));
}

/*
 * Recursively adjusts the weights and their learning rates of all incoming
 * projections of a group g.
 */
void bp_update_inc_projs_dbd(struct network *n, struct group *g)
{
        for (uint32_t  i = 0; i < g->inc_projs->num_elements; i++) {
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
                 * During BPTT, we want to only adjust weights in the
                 * network of the current timestep.
                 */
                if (p->recurrent)
                        continue;

                bp_update_inc_projs_dbd(n, p->to);
        }
}

/*
 * This adjusts the weights and their learning rates of a projection p
 * between a group g' and g.
 */
void bp_update_projection_dbd(struct network *n, struct group *g,
        struct projection *p)
{
        /* local status statistics */
        double weight_cost        = 0.0;
        double gradient_linearity = 0.0;
        double last_deltas_length = 0.0;
        double gradients_length   = 0.0;

        /*
         * Adjust the weight and its learning rate between unit i in group
         * g' and unit j in group g.
         */
#ifdef _OPENMP
#pragma omp parallel for reduction(+:weight_cost, gradient_linearity, last_deltas_length, gradients_length)
#endif /* _OPENMP */
        for (uint32_t  i = 0; i < p->to->vector->size; i++) {
                for (uint32_t  j = 0; j < g->vector->size; j++) {

                        /***********************
                         **** update weight ****
                         ***********************/

                        double weight_delta = 0.0;

                        /*
                         * First, we apply learning:
                         *
                         * Dw_ij = -esilon * dE/dw_ij
                         */
                        weight_delta += -p->dynamic_params->elements[i][j]
                                * p->gradients->elements[i][j];

                        /*
                         * Next, we apply momentum:
                         *
                         * Dw_ij = Dw_ij + a * Dw_ij(t-1)
                         */
                        weight_delta += n->momentum
                                * p->prev_deltas->elements[i][j];
                        
                        /*
                         * Finally, we apply weight decay:
                         *
                         * Dw_ij = Dw_ij - d * w_ij
                         */
                        weight_delta -= n->weight_decay 
                                * p->weights->elements[i][j];

                        /*
                         * Adjust the weight:
                         *
                         * w_ij = w_ij + Dw_ij
                         */
                        p->weights->elements[i][j] += weight_delta;
                        
                        /*
                         * Compute weight cost:
                         *
                         * wc = sum_i sum_j (w_ij ^ 2)
                         */
                        weight_cost += pow(p->weights->elements[i][j], 2.0);
                        
                        /*
                         * Compute the numerator of the gradient linearity:
                         *
                         * sum_i sum_j (Dw_ij(t-1) * dE/dw_ij)
                         */
                        gradient_linearity +=
                                p->prev_deltas->elements[i][j]
                                * p->gradients->elements[i][j];
                        
                        /*
                         * Compute the sum of squares of the previous weight
                         * delta vector:
                         *
                         * sum_i sum_j (Dw_ij(t-1) ^ 2
                         */
                        last_deltas_length +=
                                 pow(p->prev_deltas->elements[i][j], 2.0);

                        /*
                         * Compute the sum of squares of the gradients:
                         *
                         * sum_i sum_j (dE/dw_ij ^ 2)
                         */
                        gradients_length +=
                                pow(p->gradients->elements[i][j], 2.0);

                        /* 
                         * Store a copy of the weight change.
                         */
                        p->prev_deltas->elements[i][j] = weight_delta;

                        /******************************
                         **** update learning rate ****
                         ******************************/

                        double lr_delta = 0.0;

                        /*
                         * Current gradient and average of past gradients
                         * have same sign:
                         *
                         * dE/dw_ij_bar(t-1) * dE/dw_ij(t) > 0
                         *
                         * Note: dE/dw_ij_bar(t-1) is stored in the
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
                         * dE/dw_ij_bar(t-1) * dE/dw_ij(t) < 0
                         */
                        } else if (p->prev_gradients->elements[i][j]
                                * p->gradients->elements[i][j] < 0.0) {
                                /*
                                 * De_ij = -phi * e_ij(t)
                                 */
                                lr_delta = -n->dbd_rate_decrement
                                        * p->dynamic_params->elements[i][j];
                        }

                        /*
                         * Adjust the learning rate:
                         *
                         * e_ij = e_ij + De_ij
                         */
                        p->dynamic_params->elements[i][j] += lr_delta;

                        /*
                         * Determine the exponential average of the current
                         * and past gradients:
                         * 
                         * dE/dw_ij_bar(t) = (1 - theta) * dE/dw_ij + theta
                         *     * dE/dw_ij_bar(t-1)
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

        /*
         * Add the local status statistics to the global status statistics.
         */
        n->status->weight_cost        += weight_cost;
        n->status->gradient_linearity += gradient_linearity;
        n->status->last_deltas_length += last_deltas_length;
        n->status->gradients_length   += gradients_length;
}
