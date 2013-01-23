/*
 * engine.c
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

#include "act.h"
#include "bp.h"
#include "engine.h"
#include "pprint.h"

/*
 * ########################################################################
 * ## Training                                                           ##
 * ########################################################################
 */

/*
 * Train network.
 */

void train_network(struct network *n)
{
        n->learning_algorithm(n);
}

/*
 * Report training progress.
 */

void print_training_progress(struct network *n)
{
        if (n->status->epoch == 1 || n->status->epoch % n->report_after == 0)
                pprintf("epoch: %d | error: %lf | wc: %lf | gl: %lf",
                                n->status->epoch,
                                n->status->error,
                                n->status->weight_cost,
                                n->status->gradient_linearity);
}

/*
 * Print training summary.
 */

void print_training_summary(struct network *n)
{
        mprintf("training finished after [%d] epoch(s) | network error: [%f]",
                        n->status->epoch,
                        n->status->error);
}

/*
 * Scale learning rate.
 */

void scale_learning_rate(struct network *n)
{
        int sa = n->lr_scale_after * n->max_epochs;
        if (sa > 0 && n->status->epoch % sa == 0) {
                double lr = n->learning_rate;
                n->learning_rate *= n->lr_scale_factor;
                mprintf("scaled learning rate: [%lf -> %lf]",
                                lr, n->learning_rate);
        }
}

/*
 * Scale momentum factor.
 */

void scale_momentum(struct network *n)
{
        int sa = n->mn_scale_after * n->max_epochs;
        if (sa > 0 && n->status->epoch % sa == 0) {
                double mn = n->momentum;
                n->momentum *= n->mn_scale_factor;
                mprintf("scaled momentum: [%lf -> %lf]",
                                mn, n->momentum);
        }
}

/*
 * ########################################################################
 * ## Backpropagation training                                           ##
 * ########################################################################
 */

/*
 * Backpropagation (BP) training.
 */

void train_network_with_bp(struct network *n)
{
        if (n->training_order == TRAIN_ORDERED)
                order_set(n->training_set);

        /* train for a maximum number of epochs */
        int elem_itr = 0;
        for (n->status->epoch = 1;
                        n->status->epoch <= n->max_epochs;
                        n->status->epoch++) {
                n->status->prev_error = n->status->error;
                n->status->error = 0.0;

                /* determine training order */
                if (elem_itr == 0) {
                        if (n->training_order == TRAIN_PERMUTED)
                                permute_set(n->training_set);
                        if (n->training_order == TRAIN_RANDOMIZED)
                                randomize_set(n->training_set);
                }

                /* train a batch of items */
                for (int i = 0; i < n->batch_size; i++) {
                        /* select item */
                        int elem_idx = n->training_set->order[elem_itr++];
                        struct element *e = n->training_set->elements[elem_idx];

                        /* restart training set (if required) */
                        if (elem_itr == n->training_set->num_elements)
                                elem_itr = 0;

                        /* reset context groups (for SRNs) */
                        if (n->type == TYPE_SRN)
                                reset_context_groups(n);

                        /* present all events of the current item */
                        for (int j = 0; j < e->num_events; j++) {
                                copy_vector(n->input->vector, e->inputs[j]);
                                feed_forward(n, n->input);

                                /* inject error if an event has a target */
                                if (e->targets[j]) {
                                        reset_error_signals(n);
                                        bp_output_error(n->output, e->targets[j]);
                                        bp_backpropagate_error(n, n->output);
                                        n->status->error += n->output->err_fun->fun(
                                                        n->output, e->targets[j])
                                                / n->batch_size;
                                }
                        }
                }

                /* stop training if threshold is reaced */
                if (n->status->error < n->error_threshold) {
                        print_training_summary(n);
                        break;
                }

                /* update weights */
                n->update_algorithm(n);

                /* scale learning rate and momentum */
                scale_learning_rate(n);
                scale_momentum(n);

                print_training_progress(n);
        }
}

/*
 * Backpropagation Through Time (BPTT) training.
 */

void train_network_with_bptt(struct network *n)
{
        struct rnn_unfolded_network *un = n->unfolded_net;

        /* order training items (if required) */
        if (n->training_order == TRAIN_ORDERED)
                order_set(n->training_set);

        /* train for a maximum number of epochs */
        int elem_itr = 0;
        for (n->status->epoch = 1; 
                        n->status->epoch <= n->max_epochs;
                        n->status->epoch++) {
                n->status->prev_error = n->status->error;
                n->status->error = 0.0;

                /* reset context groups and error signals */
                for (int j = 0; j < un->stack_size; j++) {
                        reset_recurrent_groups(un->stack[j]);
                        //reset_context_groups(un->stack[j]);
                        reset_error_signals(un->stack[j]);
                }

                /* stack pointer */
                int sp = 0;

                /* determine training order */
                if (elem_itr == 0) {
                        if (n->training_order == TRAIN_PERMUTED)
                                permute_set(n->training_set);
                        if (n->training_order == TRAIN_RANDOMIZED)
                                randomize_set(n->training_set);
                }

                /* select item */
                int elem_idx = n->training_set->order[elem_itr++];
                struct element *e = n->training_set->elements[elem_idx];

                /* restart training set (if required) */
                if (elem_itr == n->training_set->num_elements)
                        elem_itr = 0;

                /* present all events of the current item */
                for (int j = 0; j < e->num_events; j++) {
                        copy_vector(un->stack[sp]->input->vector, e->inputs[j]);
                        feed_forward(un->stack[sp], un->stack[sp]->input);

                        /* 
                         * Inject error if a target is specified for the
                         * current event, and backpropagate error if history
                         * is full.
                         */
                        if (e->targets[j]) {
                                reset_error_signals(un->stack[sp]);
                                bp_output_error(un->stack[sp]->output, e->targets[j]);
                                if (sp == un->stack_size - 1) {
                                        bp_backpropagate_error(un->stack[sp],
                                                        un->stack[sp]->output);
                                        n->status->error += n->output->err_fun->fun(
                                                        un->stack[sp]->output, e->targets[j]);
                                        rnn_sum_gradients(un);
                                        n->update_algorithm(un->stack[0]);
                                }
                        }

                        /*
                         * Cycle stack, if required. Otherwise,
                         * increase stack pointer.
                         */
                        if (sp == un->stack_size - 1) {
                                rnn_cycle_stack(un);
                        } else {
                                sp++;
                        }
                }

                /* stop training if threshold is reaced */
                if (n->status->error < n->error_threshold) {
                        print_training_summary(n);
                        break;
                }

                /* scale learning rate and momentum */
                scale_learning_rate(n);
                scale_momentum(n);

                print_training_progress(n);
        }
}

/*
 * ########################################################################
 * ## Testing                                                            ##
 * ########################################################################
 */

/*
 * Test network.
 */

void test_network(struct network *n)
{
        if (n->type == TYPE_FFN)
                test_ffn_network(n);
        if (n->type == TYPE_SRN)
                test_ffn_network(n);
        if (n->type == TYPE_RNN)
                test_rnn_network(n);
}

/*
 * Test feed forward network.
 */

void test_ffn_network(struct network *n)
{
        double total_error = 0.0;

        /* present all test items */
        for (int i = 0; i < n->test_set->num_elements; i++) {
                struct element *e = n->test_set->elements[i];
                test_ffn_network_with_item(n, e);
                total_error += n->status->error / n->test_set->num_elements;
        }

        pprintf("total error: [%lf]", total_error);
}

/*
 * Test recurrent network.
 */

void test_rnn_network(struct network *n)
{
        double total_error = 0.0;

        /* present all test items */
        for (int i = 0; i < n->test_set->num_elements; i++) {
                struct element *e = n->test_set->elements[i];
                test_rnn_network_with_item(n, e);
                total_error += n->status->error / n->test_set->num_elements;
        }

        pprintf("total error: [%lf]", total_error);
}

void test_network_with_item(struct network *n, struct element *e)
{
        if (n->type == TYPE_FFN)
                test_ffn_network_with_item(n, e);
        if (n->type == TYPE_SRN)
                test_ffn_network_with_item(n, e);
        if (n->type == TYPE_RNN)
                test_rnn_network_with_item(n, e);
}

/*
 * Test a feed forward network with a single item.
 */

void test_ffn_network_with_item(struct network *n, struct element *e)
{
        n->status->error = 0.0;

        /* reset context groups (for SRNs) */
        if (n->type == TYPE_SRN)
                reset_context_groups(n);

        printf("Item: \"%s\"\n", e->name);

        /* present all events of the current item */
        for (int j = 0; j < e->num_events; j++) {
                printf("\nEvent: %d", j);
                printf("\nI: ");
                pprint_vector(e->inputs[j]);

                copy_vector(n->input->vector, e->inputs[j]);
                feed_forward(n, n->input);

                /* compute error if an event has a target */
                if (e->targets[j]) {
                        n->status->error += n->output->err_fun->fun(
                                        n->output, e->targets[j]);
                        printf("\nT: ");
                        pprint_vector(e->targets[j]);
                        printf("O: ");
                        pprint_vector(n->output->vector);
                        pprintf("error: [%lf]\n", n->status->error);
                }
        }
}

/*
 * Test a recurrent network with a single item.
 */

void test_rnn_network_with_item(struct network *n, struct element *e)
{
        n->status->error = 0.0;

        /* unfolded network */
        struct rnn_unfolded_network *un = n->unfolded_net;

        /* reset context groups and error signals */
        for (int j = 0; j < un->stack_size; j++) {
                reset_recurrent_groups(un->stack[j]);
                //reset_context_groups(un->stack[j]);
                //reset_error_signals(un->stack[j]);
        }

        /* stack pointer */
        int sp = 0;

        printf("Item: \"%s\"\n", e->name);

        /* present all events of the current item */
        for (int j = 0; j < e->num_events; j++) {
                printf("\nEvent: %d", j);
                printf("\nI: ");
                pprint_vector(e->inputs[j]);

                copy_vector(un->stack[sp]->input->vector, e->inputs[j]);
                feed_forward(un->stack[sp], un->stack[sp]->input);

                if (e->targets[j]) {
                        n->status->error += n->output->err_fun->fun(
                                        un->stack[sp]->output, e->targets[j]);
                        printf("\nT: ");
                        pprint_vector(e->targets[j]);
                        printf("O: ");
                        pprint_vector(un->stack[sp]->output->vector);
                        pprintf("error: [%lf]\n", n->status->error);
                }

                /*
                 * Cycle stack, if required. Otherwise,
                 * increase stack pointer.
                 */
                if (sp == un->stack_size - 1) {
                        rnn_cycle_stack(un);
                } else {
                        sp++;
                }
        }
}
