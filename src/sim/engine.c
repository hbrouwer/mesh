/*
 * engine.c
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
#include "bp.h"
#include "engine.h"
#include "set.h"
#include "pprint.h"

/*
 * ########################################################################
 * ## Network training                                                   ##
 * ########################################################################
 */

void train_network(struct network *n)
{
        mprintf("starting training of network: [%s]", n->name);

        n->learning_algorithm(n);
}

/*
 * This function implements backpropagation (BP) training.
 */

void train_network_bp(struct network *n)
{
        int elem = 0;

        for (int epoch = 1; epoch <= n->max_epochs; epoch++) {
                double me = 0.0;

                printf("\n");

                /* determine order of training items */
                if (elem == 0) {
                        if (n->training_order == TRAIN_PERMUTED)
                                permute_set(n->training_set);
                        if (n->training_order == TRAIN_RANDOMIZED)
                                randomize_set(n->training_set);
                }

                /* present all training items */
                for (int i = 0; i < n->batch_size; i++) {
                        int elem_idx = n->training_set->order[elem++];
                        struct element *e = n->training_set->elements[elem_idx];

                        printf("%s\n", e->name);

                        /* 
                         * Restart at the beginning of the training
                         * set, if required.
                         */
                        if (elem == n->training_set->num_elements)
                                elem = 0;
                        
                        /* reset context groups */
                        if (n->type == TYPE_SRN)
                                reset_context_groups(n);

                        /* present all events for this item */
                        for (int j = 0; j < e->num_events; j++) {
                                copy_vector(n->input->vector, e->inputs[j]);
                                feed_forward(n, n->input);

                                /* inject error if a target is specified */
                                if (e->targets[j]) {
                                        reset_error_signals(n);
                                        bp_output_error(n->output, e->targets[j]);
                                        bp_backpropagate_error(n, n->output);
                                        
                                        /* compute error */
                                        me += n->output->err_fun->fun(n->output, e->targets[j]);
                                }
                        }
                }

                /* 
                 * Compute mean error and stop training 
                 * if threshold is reached.
                 */
                me /= n->batch_size;
                if (me < n->error_threshold) {
                        n->status->epoch = epoch;
                        n->status->prev_error = n->status->error;
                        n->status->error = me;
                        print_training_progress(n);
                        break;
                }

                /* update weights */
                n->update_algorithm(n);

                /* report progress */
                if (epoch == 1 || epoch % n->report_after == 0) {
                        n->status->epoch = epoch;
                        n->status->prev_error = n->status->error;
                        n->status->error = me;
                        print_training_progress(n);
                }

                /* scale LR and Momentum */
                scale_learning_rate(epoch, n);
                scale_momentum(epoch, n);
        }
}

/*
 * This function implements backpropagation through time (BPTT) training.
 */

/*
 * XXX: Epochwise versus truncated... check Haykin...
 *
 * XXX: Check Williams & Peng (1990): BPTT(h;h') that captures both
 * epochwise/BPTT(h;h) and trucated/BPTT(h;1) backprop through time.
 */
void train_network_bptt(struct network *n)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        struct network *nsp = un->stack[0];

        int elem = 0;
        for (int epoch = 1; epoch <= n->max_epochs; epoch++) {
                double me = 0.0;
                int his = 0;

                /* determine order of training items */
                if (elem == 0) {
                        if (n->training_order == TRAIN_PERMUTED)
                                permute_set(n->training_set);
                        if (n->training_order == TRAIN_RANDOMIZED)
                                randomize_set(n->training_set);
                }

                /* present all training items */
                for (int i = 0; i < n->batch_size; i++) {
                        int elem_idx = n->training_set->order[elem++];
                        struct element *e = n->training_set->elements[elem_idx];
                        // struct element *e = n->training_set->elements[elem++];

                        /* 
                         * Restart at the beginning of the training
                         * set, if required.
                         */
                        if (elem == n->training_set->num_elements)
                                elem = 0;

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
                                        rnn_cycle_stack(un);
                                        nsp = un->stack[--his];
                                } else {
                                        nsp = un->stack[his];
                                }

                                copy_vector(nsp->input->vector, e->inputs[j]);
                                feed_forward(nsp, nsp->input);

                                /* 
                                 * Inject error if a target is specified
                                 * and history is full.
                                 */
                                if (e->targets[j]) {
                                        reset_error_signals(nsp);
                                        bp_output_error(nsp->output, e->targets[j]);

                                        if (his == un->stack_size - 1)
                                                bp_backpropagate_error(nsp, nsp->output);

                                        /* compute error */
                                        me += n->output->err_fun->fun(nsp->output, e->targets[j]);
                                }

                                his++;
                        }
                }

                /* 
                 * Compute mean error and stop training 
                 * if threshold is reached.
                 */                
                me /= n->batch_size;
                if (me < n->error_threshold)
                        break;

                /* sum gradients over unfolded network */
                rnn_sum_gradients(un);

                /* update weights */
                n->update_algorithm(un->stack[0]);

                /* report progress */
                if (epoch == 1 || epoch % n->report_after == 0) {
                        n->status->epoch = epoch;
                        n->status->prev_error = n->status->error;
                        n->status->error = me;
                        print_training_progress(n);
                }

                /* scale LR and Momentum */
                scale_learning_rate(epoch, n);
                scale_momentum(epoch, n);
        }
}

/*
 * Print training progress.
 */

void print_training_progress(struct network *n)
{
        pprintf("epoch: %d | error: %lf | wc: %lf | gl: %lf",
                        n->status->epoch,
                        n->status->error,
                        n->status->weight_cost,
                        n->status->gradient_linearity);
}

/*
 * ########################################################################
 * ## Learning rate and Momentum scaling                                 ##
 * ########################################################################
 */

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
 * ## Network testing                                                    ##
 * ########################################################################
 */

void test_network(struct network *n)
{
        mprintf("starting testing of network: [%s]", n->name);

        double me = 0.0;

        /* present all test items */
        for (int i = 0; i < n->test_set->num_elements; i++) {
                struct element *e = n->test_set->elements[i];

                /* reset context groups */
                if (n->type == TYPE_SRN)
                        reset_context_groups(n);

                /* present all events for this item */
                rprintf("\nI: \"%s\"", e->name);
                for (int j = 0; j < e->num_events; j++) {
                        copy_vector(n->input->vector, e->inputs[j]);
                        feed_forward(n, n->input);

                        if (e->targets[j] != NULL) {
                                /* compute error */
                                me += n->output->err_fun->fun(n->output, e->targets[j]);

                                printf("T: ");
                                pprint_vector(e->targets[j]);
                                printf("O: ");
                                pprint_vector(n->output->vector);
                        }
                }
        }

        /* report error */
        me /= n->test_set->num_elements;
        pprintf("error: [%lf]", me);
}

// XXX: As always -- check logic...

void test_unfolded_network(struct network *n)
{
        mprintf("starting testing of network: [%s]", n->name);

        struct rnn_unfolded_network *un = n->unfolded_net;
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
                rprintf("\nI: \"%s\"", e->name);
                for (int j = 0; j < e->num_events; j++) {
                        /* cycle network stack if necessary */
                        if (his == un->stack_size) {
                                rnn_cycle_stack(un);
                                nsp = un->stack[--his];
                        } else {
                                nsp = un->stack[his];
                        }

                        copy_vector(nsp->input->vector, e->inputs[j]);
                        feed_forward(nsp, nsp->input);

                        if (e->targets[j] && his == un->stack_size - 1) {
                                /* compute error */
                                me += n->output->err_fun->fun(nsp->output, e->targets[j]);

                                printf("T: ");
                                pprint_vector(e->targets[j]);
                                printf("O: ");
                                pprint_vector(nsp->output->vector);

                        }

                        his++;
                }
        }

        /* report error */
        me /= n->test_set->num_elements;
        pprintf("error: [%lf]", me);
}
