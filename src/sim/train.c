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

#include "error.h"
#include "ffn_unfold.h"
#include "math.h"
#include "set.h"
#include "train.h"

#include <math.h>

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
                
                /* reset Elman groups */
                if (n->srn)
                        reset_elman_groups(n);

                /* present all events for this item */
                rprintf("testing item: %d -- \"%s\"", i, e->name);
                for (int j = 0; j < e->num_events; j++) {
                        copy_vector(n->input->vector, e->inputs[j]);
                        feed_forward(n, n->input);

                        if (e->targets[j] != NULL) {
                                copy_vector(n->target, e->targets[j]);
                                
                                /* compute error */
                                me += n->error_fun(n);

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
                                me += n->error_fun(n);

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

void report_training_status(int epoch, double me, struct network *n)
{
        pprintf("epoch: [%d] | error: [%lf]", epoch, me);
}

/*
 * ########################################################################
 * ## Feed forward                                                       ##
 * ########################################################################
 */

void feed_forward(struct network *n, struct group *g)
{
        /* copy previous vector to Elman group (if there is one)*/
        if (g->elman_proj)
                copy_vector(g->elman_proj->vector, g->vector);

        /* propagate to outgoing projections */
        for (int i = 0; i < g->out_projs->num_elements; i++) {
                /* skip recurrent groups */
                if (g->out_projs->elements[i]->recurrent)
                        continue;

                /* compute unit activations */
                struct group *rg = g->out_projs->elements[i]->to;
                for (int j = 0; j < rg->vector->size; j++) {
                        rg->vector->elements[j] = unit_activation(n, rg, j);
                        if (rg != n->output) {
                                rg->vector->elements[j] =
                                        n->act_fun(rg->vector, j);
                        } else {
                                rg->vector->elements[j] =
                                        n->out_act_fun(rg->vector, j);
                        }
                }
        }

        /* recursively repeat the above for all outgoing projections*/
        for (int i = 0; i < g->out_projs->num_elements; i++)
                if (!g->out_projs->elements[i]->recurrent)
                        feed_forward(n, g->out_projs->elements[i]->to);
}

double unit_activation(struct network *n, struct group *g, int u)
{
        /* sum the weighted net activation of a unit */
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

                        /* reset Elman groups */
                        if (n->srn)
                                reset_elman_groups(n);

                        /* present all events for this item */
                        for (int j = 0; j < e->num_events; j++) {
                                copy_vector(n->input->vector, e->inputs[j]);
                                feed_forward(n, n->input);

                                /* inject error if a target is specified */
                                if (e->targets[j]) {
                                        copy_vector(n->target, e->targets[j]);

                                        /* backpropagate error */
                                        struct vector *error = n->error_fun_deriv(n);
                                        backpropagate_error(n, n->output, error);
                                        dispose_vector(error);
                                        
                                        /* compute error */
                                        me += n->error_fun(n);
                                }
                        }
                }

                /* compute and report mean error */
                me /= training_set->num_elements;
                if (epoch == 1 || epoch % n->report_after == 0)
                        report_training_status(epoch, me, n);

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
                                struct vector *error = nsp->error_fun_deriv(nsp);
                                backpropagate_error(nsp, nsp->output, error);
                                dispose_vector(error);

                                /* compute error */
                                me += n->error_fun(nsp);

                                /* sum deltas over unfolded network */
                                ffn_sum_deltas(un);

                                /* adjust weights */
                                adjust_weights(un->stack[0], un->stack[0]->output);
                        }
                }

                /* compute and report mean error */
                me /= training_set->num_elements;
                if (epoch == 1 || epoch % n->report_after == 0)
                        report_training_status(epoch, me, n);

                /* stop training if threshold is reached */
                if (me < n->error_threshold)
                        break;

                /* scale LR and Momentum */
                scale_learning_rate(epoch,n);
                scale_momentum(epoch,n);
        }
}

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
                        act_deriv = n->act_fun_deriv(g->vector, i);
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
