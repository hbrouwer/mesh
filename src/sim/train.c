/*
 * train.c
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

#include "act.h"
#include "error.h"
#include "ffn_unfold.h"
#include "math.h"
#include "set.h"
#include "train.h"

#include <math.h>

/*
 * This provides an implementation of the backpropagation (BP) algorithm
 * (Rumelhart, Hinton, & Williams, 1986), and its backpropagation through
 * time (BPTT) extension.
 * 
 * Let j be a unit in one of network's groups, and i a unit in a group
 * projecting to it. The net input x_j to unit j is defined as:
 *
 *     x_j = sum_i (y_i * w_ij)
 * 
 * where y_i is the activation level of unit i in the projecting group, and
 * w_ij the weight of the connection between unit j and unit i. The
 * activation level y_j of unit j is then defined as:
 *
 *    y_j = f(x_j)
 *
 * where f is a non-linear activation function, such as the commonly used
 * sigmoid function:
 *
 *    y_j = 1 / (1 + e^(-x_j))
 *
 * When activation is propagated from the input group to the output group
 * of the network, the network's error for a given input pattern is defined
 * as:
 *
 *    E = 0.5 * sum_j (o_j - t_j)^2
 *
 * where o_j is the observed activation level of unit j, and t_j its target
 * activation level.
 *
 */

/*
 * References
 *
 * Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning
 *   representations by back-propagating errors. Nature, 323, pp. 533-536.
 */

/*
 * ########################################################################
 * ## Test network                                                       ##
 * ########################################################################
 */

void train_network(struct network *n)
{
        mprintf("starting training of network: [%s]", n->name);

        n->learning_algorithm(n);
}

void test_network(struct network *n)
{
        mprintf("starting testing of network: [%s]", n->name);
        
        double me = 0.0;

        /* present all test items */
        for (int i = 0; i < n->test_set->num_elements; i++) {
                struct element *e = n->test_set->elements[i];
                
                /* reset context groups */
                if (n->srn)
                        reset_context_groups(n);

                /* present all events for this item */
                rprintf("testing item: %d -- \"%s\"", i, e->name);
                for (int j = 0; j < e->num_events; j++) {
                        copy_vector(n->input->vector, e->inputs[j]);
                        feed_forward(n, n->input);

                        if (e->targets[j] != NULL) {
                                copy_vector(n->target, e->targets[j]);
                                
                                /* compute error */
                                me += n->error->fun(n);

                                print_vector(n->target);
                                print_vector(n->output->vector);
                        }
                }
        }

        /* report error */
        me /= n->test_set->num_elements;
        pprintf("error: [%lf]", me);
}

void test_unfolded_network(struct network *n)
{
        mprintf("starting testing of network: [%s]", n->name);

        struct ffn_unfolded_network *un = n->unfolded_net;
        struct network *nsp = un->stack[0];

        double me = 0.0;
        int his = 0;

        /* present all test items */
        for (int i = 0; i < n->test_set->num_elements; i++) {
                struct element *e = n->test_set->elements[i];

                /*
                 * reset recurrent groups
                 *
                 * XXX: Does this make sense when history length is
                 *   larger then the number of events in an item?
                 */
                reset_recurrent_groups(nsp);

                /* present all events for this item */
                rprintf("testing item: %d -- \"%s\"", i, e->name);
                for (int j = 0; j < e->num_events; j++) {
                        /* cycle network stack if necessary */
                        if (his == un->stack_size) {
                                ffn_cycle_stack(un);
                                nsp = un->stack[--his];
                        } else {
                                nsp = un->stack[his];
                        }

                        copy_vector(nsp->input->vector, e->inputs[j]);
                        if (e->targets[j])
                                copy_vector(nsp->target, e->targets[j]);

                        feed_forward(nsp, nsp->input);

                        if (e->targets[j]) {
                                copy_vector(nsp->target, e->targets[j]);

                                /* compute error */
                                me += n->error->fun(nsp);

                                print_vector(nsp->target);
                                print_vector(nsp->output->vector);
                                
                        }
                        his++;
                }
        }

        /* report error */
        me /= n->test_set->num_elements;
        pprintf("error: [%lf]", me);
}

/*
 * ########################################################################
 * ## Backpropagation (BP) training                                      ##
 * ########################################################################
 */

void train_bp(struct network *n)
{
        for (int epoch = 1; epoch <= n->max_epochs; epoch++) {
                double me = 0.0;
                
                /* determine order of training items */
                struct set *training_set;
                if (n->training_order == TRAIN_ORDERED)
                        training_set = n->training_set;
                if (n->training_order == TRAIN_PERMUTED)
                        training_set = permute_set(n->training_set);
                if (n->training_order == TRAIN_RANDOMIZED)
                        training_set = randomize_set(n->training_set);

                /* present all training items */
                for (int i = 0; i < training_set->num_elements; i++) {
                        struct element *e = training_set->elements[i];

                        /* reset context groups */
                        if (n->srn)
                                reset_context_groups(n);

                        /* present all events for this item */
                        for (int j = 0; j < e->num_events; j++) {
                                copy_vector(n->input->vector, e->inputs[j]);
                                feed_forward(n, n->input);

                                /* inject error if a target is specified */
                                if (e->targets[j]) {
                                        copy_vector(n->target, e->targets[j]);

                                        /* backpropagate error */
                                        struct vector *error = n->error->deriv(n);
                                        backpropagate_error(n, n->output, error);
                                        dispose_vector(error);
                                        
                                        /* compute error */
                                        me += n->error->fun(n);
                                }
                        }
                }

                /* compute and report mean error */
                me /= training_set->num_elements;
                if (epoch == 1 || epoch % n->report_after == 0)
                        pprintf("epoch: [%d] | error: [%lf]", epoch, me);

                /* stop training if threshold is reached */
                if (me < n->error_threshold)
                        break;

                /* adjust weights */
                adjust_weights(n, n->output);

                /* scale LR and Momentum */
                scale_learning_rate(epoch,n);
                scale_momentum(epoch,n);
        }
}

/*
 * ########################################################################
 * ## Backpropagation Through Time (BPTT) training                       ##
 * ########################################################################
 */

void train_bptt(struct network *n)
{
        struct ffn_unfolded_network *un = n->unfolded_net;
        struct network *nsp = un->stack[0];

        for (int epoch = 1; epoch <= n->max_epochs; epoch++) {
                double me = 0.0;
                int his = 0;

                /* determine order of training items */
                struct set *training_set;
                if (n->training_order == TRAIN_ORDERED)
                        training_set = n->training_set;
                if (n->training_order == TRAIN_PERMUTED)
                        training_set = permute_set(n->training_set);
                if (n->training_order == TRAIN_RANDOMIZED)
                        training_set = randomize_set(n->training_set);

                /* present all training items */
                for (int i = 0; i < training_set->num_elements; i++) {
                        struct element *e = training_set->elements[i];

                        /*
                         * reset recurrent groups
                         *
                         * XXX: Does this make sense when history length is
                         *   larger then the number of events in an item?
                         */
                        reset_recurrent_groups(nsp);

                        /* present all events for this item */
                        for (int j = 0; j < e->num_events; j++) {
                                /* cycle network stack if necessary */
                                if (his == un->stack_size) {
                                        ffn_cycle_stack(un);
                                        nsp = un->stack[--his];
                                } else {
                                        nsp = un->stack[his];
                                }

                                copy_vector(nsp->input->vector, e->inputs[j]);
                                if (e->targets[j])
                                        copy_vector(nsp->target, e->targets[j]);

                                feed_forward(nsp, nsp->input);

                                his++;
                        }

                        if (his == un->stack_size) {
                                /* backpropagate error */
                                struct vector *error = nsp->error->deriv(nsp);
                                backpropagate_error(nsp, nsp->output, error);
                                dispose_vector(error);

                                /* compute error */
                                me += n->error->fun(nsp);

                                /* sum deltas over unfolded network */
                                ffn_sum_deltas(un);

                                /* adjust weights */
                                adjust_weights(un->stack[0], un->stack[0]->output);
                        }
                }

                /* compute and report mean error */
                me /= training_set->num_elements;
                if (epoch == 1 || epoch % n->report_after == 0)
                        pprintf("epoch: [%d] | error: [%lf]", epoch, me);

                /* stop training if threshold is reached */
                if (me < n->error_threshold)
                        break;

                /* scale LR and Momentum */
                scale_learning_rate(epoch,n);
                scale_momentum(epoch,n);
        }
}

/*
 * ########################################################################
 * ## Learning rate and momentum scaling                                 ##
 * ########################################################################
 */

/* learning rate rescaling */
void scale_learning_rate(int epoch, struct network *n)
{
        int scale_after = n->lr_scale_after * n->max_epochs;
        if (scale_after > 0 && epoch % scale_after == 0) {
                double lr = n->learning_rate;
                n->learning_rate = n->lr_scale_factor * n->learning_rate;
                mprintf("scaled learning rate: [%lf --> %lf]",
                                lr, n->learning_rate);
        }

}

/* momentum rescaling */
void scale_momentum(int epoch, struct network *n)
{
        int scale_after = n->mn_scale_after * n->max_epochs;
        if (scale_after > 0 && epoch % scale_after == 0) {
                double mn = n->momentum;
                n->momentum = n->mn_scale_factor * n->momentum;
                mprintf("scaled momentum: [%lf --> %lf]",
                                mn, n->momentum);
        }
}

/*
 * ########################################################################
 * ## Backpropagate error                                                ##
 * ########################################################################
 */

void backpropagate_error(struct network *n, struct group *g, 
                struct vector *error)
{
        /* compute deltas and error for all incoming projections */
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *p = g->inc_projs->elements[i];
                zero_out_vector(p->error);
                comp_proj_deltas_and_error(n, p, error);
        }
        
        /* sum and backpropagate error for all incoming projections */
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct group *ng = g->inc_projs->elements[i]->to;
                struct vector *grp_error = group_error(n, ng);
                backpropagate_error(n, ng, grp_error);
                dispose_vector(grp_error);
        }
}

void comp_proj_deltas_and_error(struct network *n, struct projection *p,
                struct vector *error)
{
        for (int i = 0; i < p->to->vector->size; i++) {
                for (int j = 0; j < error->size; j++) {
                        p->error->elements[i] += p->weights->elements[i][j]
                                * error->elements[j];
                        p->deltas->elements[i][j] += p->to->vector->elements[i]
                                * error->elements[j];
                }
        }
}

struct vector *group_error(struct network *n, struct group *g)
{
        struct vector *error = create_vector(g->vector->size);

        for (int i = 0; i < g->vector->size; i++) {
                for (int j = 0; j < g->out_projs->num_elements; j++) {
                        struct projection *p = g->out_projs->elements[j];
                        error->elements[i] += p->error->elements[i];
                }

                double act_deriv;
                if (g != n->input)
                        act_deriv = g->act->deriv(g->vector, i);
                else 
                        act_deriv = g->vector->elements[i];

                error->elements[i] *= act_deriv;
        }

        return error;
}

/*
 * ########################################################################
 * ## Adjust weights                                                     ##
 * ########################################################################
 */

void adjust_weights(struct network *n, struct group *g)
{
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                adjust_projection_weights(n, g, g->inc_projs->elements[i]);
                if (!g->inc_projs->elements[i]->recurrent)
                        adjust_weights(n, g->inc_projs->elements[i]->to);
        }
}

void adjust_projection_weights(struct network *n, struct group *g,
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
