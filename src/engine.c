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
#include "math.h"
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
        pprintf("Epoch \t Error \t\t Weight Cost \t Gradient Lin.");
        pprintf("----- \t ----- \t\t ----------- \t -------------");

        n->learning_algorithm(n);
}



/*
 * Report training progress.
 */

void print_training_progress(struct network *n)
{
        if (n->status->epoch == 1 || n->status->epoch % n->report_after == 0)
                pprintf("%.4d \t %lf \t %lf \t %lf",
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
        mprintf(" ");
        mprintf("Training finished after %d epoch(s) -- Network error: %f",
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
                mprintf("Scaled learning rate \t\t  ( %lf => %lf )",
                                lr, n->learning_rate);
        }
}

/*
 * Scale momentum.
 */

void scale_momentum(struct network *n)
{
        int sa = n->mn_scale_after * n->max_epochs;
        if (sa > 0 && n->status->epoch % sa == 0) {
                double mn = n->momentum;
                n->momentum *= n->mn_scale_factor;
                mprintf("Scaled momentum \t\t ( %lf => %lf )",
                                mn, n->momentum);
        }
}

/*
 * Scale weight decay
 */

void scale_weight_decay(struct network *n)
{
        int sa = n->wd_scale_after * n->max_epochs;
        if (sa > 0 && n->status->epoch % sa == 0) {
                double wd = n->weight_decay;
                n->weight_decay *= n->wd_scale_factor;
                mprintf("Scale weight decay \t\t ( %lf => %lf)",
                                wd, n->weight_decay);
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
                order_set(n->asp);

        /* train for a maximum number of epochs */
        int item_itr = 0;
        for (n->status->epoch = 1;
                        n->status->epoch <= n->max_epochs;
                        n->status->epoch++) {
                n->status->prev_error = n->status->error;
                n->status->error = 0.0;

                /* determine training order */
                if (item_itr == 0) {
                        if (n->training_order == TRAIN_PERMUTED)
                                permute_set(n->asp);
                        if (n->training_order == TRAIN_RANDOMIZED)
                                randomize_set(n->asp);
                }

                /* train a batch of items */
                for (int i = 0; i < n->batch_size; i++) {
                        /* select item */
                        int item_idx = n->asp->order[item_itr++];
                        struct item *item = (struct item *)n->asp->items->elements[item_idx];

                        /* restart training set (if required) */
                        if (item_itr == n->asp->items->num_elements)
                                item_itr = 0;

                        /* reset context groups (for SRNs) */
                        if (n->type == TYPE_SRN)
                                reset_context_groups(n);

                        /* present all events of the current item */
                        for (int j = 0; j < item->num_events; j++) {
                                /* shift context groups (for SRNs) */
                                if (j > 0 && n->type == TYPE_SRN)
                                        shift_context_groups(n);

                                copy_vector(n->input->vector, item->inputs[j]);
                                feed_forward(n, n->input);

                                /* inject error if an event has a target */
                                if (item->targets[j]) {
                                        reset_error_signals(n);
                                        bp_output_error(n->output,
                                                        item->targets[j],
                                                        n->target_radius,
                                                        n->zero_error_radius);
                                        bp_backpropagate_error(n, n->output);
                                        if (j == item->num_events - 1) {
                                                n->status->error += n->output->err_fun->fun(
                                                                n->output,
                                                                item->targets[j],
                                                                n->target_radius,
                                                                n->zero_error_radius)
                                                        / n->batch_size;
                                        }
                                }
                        }
                }

                /* stop training if threshold is reached */
                if (n->status->error < n->error_threshold) {
                        print_training_summary(n);
                        break;
                }

                /* update weights */
                n->update_algorithm(n);

                /* scale learning rate, momentum and weight decay */
                scale_learning_rate(n);
                scale_momentum(n);
                scale_weight_decay(n);

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
                order_set(n->asp);

        /* train for a maximum number of epochs */
        int item_itr = 0;
        for (n->status->epoch = 1; 
                        n->status->epoch <= n->max_epochs;
                        n->status->epoch++) {
                n->status->prev_error = n->status->error;
                n->status->error = 0.0;

                /* reset recurrent groups and error signals */
                for (int j = 0; j < un->stack_size; j++) {
                        reset_recurrent_groups(un->stack[j]);
                        reset_error_signals(un->stack[j]);
                }

                /* stack pointer */
                int sp = 0;

                /* determine training order */
                if (item_itr == 0) {
                        if (n->training_order == TRAIN_PERMUTED)
                                permute_set(n->asp);
                        if (n->training_order == TRAIN_RANDOMIZED)
                                randomize_set(n->asp);
                }

                /* select item */
                int item_idx = n->asp->order[item_itr++];
                struct item *item = n->asp->items->elements[item_idx];

                /* restart training set (if required) */
                if (item_itr == n->asp->items->num_elements)
                        item_itr = 0;

                /* present all events of the current item */
                for (int j = 0; j < item->num_events; j++) {
                        copy_vector(un->stack[sp]->input->vector, item->inputs[j]);
                        feed_forward(un->stack[sp], un->stack[sp]->input);

                        /* 
                         * Inject error if a target is specified for the
                         * current event, and backpropagate error if history
                         * is full.
                         */
                        if (item->targets[j]) {
                                reset_error_signals(un->stack[sp]);
                                bp_output_error(un->stack[sp]->output,
                                                item->targets[j],
                                                n->target_radius,
                                                n->zero_error_radius);
                                if (sp == un->stack_size - 1) {
                                        bp_backpropagate_error(un->stack[sp],
                                                        un->stack[sp]->output);
                                        n->status->error += n->output->err_fun->fun(
                                                        un->stack[sp]->output,
                                                        item->targets[j],
                                                        n->target_radius,
                                                        n->zero_error_radius);
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

                /* stop training if threshold is reached */
                if (n->status->error < n->error_threshold) {
                        print_training_summary(n);
                        break;
                }

                /* scale learning rate, momentum and weight decay */
                scale_learning_rate(n);
                scale_momentum(n);
                scale_weight_decay(n);

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
        n->status->error = 0.0;
        int threshold_reached = 0;

        for (int i = 0; i < n->asp->items->num_elements; i++) {
                struct item *item = (struct item *)n->asp->items->elements[i];

                /* reset context groups (for SRNs) */
                if (n->type == TYPE_SRN)
                        reset_context_groups(n);

                /* present all events of the current item */
                for (int j = 0; j < item->num_events; j++) {
                        /* shift context groups (for SRNs) */
                        if (j > 0 && n->type == TYPE_SRN)
                                shift_context_groups(n);

                        copy_vector(n->input->vector, item->inputs[j]);
                        feed_forward(n, n->input);

                        /* compute error if an event has a target */
                        if (item->targets[j] && j == item->num_events - 1) {
                                double error = n->output->err_fun->fun(
                                                n->output,
                                                item->targets[j],
                                                n->target_radius,
                                                n->zero_error_radius);
                                n->status->error += error;

                                if (error <= n->error_threshold)
                                        threshold_reached++;
                        }
                        
                }
                
        }

        mprintf(" ");
        mprintf("Number of items: \t\t %d",
                        n->asp->items->num_elements);
        mprintf("Total error: \t\t\t %lf",
                        n->status->error);
        mprintf("Error per example:\t\t %lf",
                        n->status->error / n->asp->items->num_elements);
        mprintf("# Items reached threshold: \t %d (%.2lf%%)",
                        threshold_reached,
                        ((double)threshold_reached / n->asp->items->num_elements) * 100.0);

        mprintf(" ");
}

/*
 * Test recurrent network.
 */

void test_rnn_network(struct network *n)
{
        n->status->error = 0.0;
        int threshold_reached = 0;

        /* unfolded network */
        struct rnn_unfolded_network *un = n->unfolded_net;

        for (int i = 0; i < n->asp->items->num_elements; i++) {
                struct item *item = (struct item *)n->asp->items->elements[i];

                /* reset context groups and error signals */
                for (int j = 0; j < un->stack_size; j++)
                        reset_recurrent_groups(un->stack[j]);

                /* stack pointer */
                int sp = 0;
 
                /* present all events of the current item */
                for (int j = 0; j < item->num_events; j++) {
                        copy_vector(un->stack[sp]->input->vector, item->inputs[j]);
                        feed_forward(un->stack[sp], un->stack[sp]->input);

                        if (item->targets[j]) {
                                double error = n->output->err_fun->fun(
                                                un->stack[sp]->output,
                                                item->targets[j],
                                                n->target_radius,
                                                n->zero_error_radius);
                                n->status->error += error;

                                if (error < n->error_threshold)
                                        threshold_reached++;
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

        mprintf(" ");
        mprintf("Number of items: \t\t %d",
                        n->asp->items->num_elements);
        mprintf("Total error: \t\t\t %lf",
                        n->status->error);
        mprintf("Error per example:\t\t %lf",
                        n->status->error / n->asp->items->num_elements);
        mprintf("# Items reached threshold: \t %d (%.2lf%%)",
                        threshold_reached,
                        ((double)threshold_reached / n->asp->items->num_elements) * 100.0);
        mprintf(" ");
}

void test_network_with_item(struct network *n, struct item *item, bool pprint, int scheme)
{
        if (n->type == TYPE_FFN)
                test_ffn_network_with_item(n, item, pprint, scheme);
        if (n->type == TYPE_SRN)
                test_ffn_network_with_item(n, item, pprint, scheme);
        if (n->type == TYPE_RNN)
                test_rnn_network_with_item(n, item, pprint, scheme);
}

/*
 * Test a feed forward network with a single item.
 */

void test_ffn_network_with_item(struct network *n, struct item *item, bool pprint, int scheme)
{
        n->status->error = 0.0;

        /* reset context groups (for SRNs) */
        if (n->type == TYPE_SRN)
                reset_context_groups(n);

        printf("\nItem: \"%s\" --  \"%s\"\n", item->name, item->meta);

        /* present all events of the current item */
        for (int j = 0; j < item->num_events; j++) {
                printf("\nEvent: \t%d\n", j);
                printf("\nInput: \t");
                if (pprint) {
                        pprint_vector(item->inputs[j], scheme);
                } else {
                        print_vector(item->inputs[j]);
                }

                /* shift context groups (for SRNs) */
                if (j > 0 && n->type == TYPE_SRN)
                        shift_context_groups(n);

                copy_vector(n->input->vector, item->inputs[j]);
                feed_forward(n, n->input);

                /* compute error if an event has a target */
                if (item->targets[j]) {
                        printf("\nTarget: ");
                        if (pprint) {
                                pprint_vector(item->targets[j], scheme);
                        } else  {
                                print_vector(item->targets[j]);
                        }
                        printf("Output: ");
                        if (pprint) {
                                pprint_vector(n->output->vector, scheme);
                        } else {
                                print_vector(n->output->vector);
                        }

                        if (j == item->num_events - 1) {
                                n->status->error += n->output->err_fun->fun(
                                        n->output,
                                        item->targets[j],
                                        n->target_radius,
                                        n->zero_error_radius);
                                pprintf("\nError: \t%lf\n", n->status->error);
                        }
                }
        }
}

/*
 * Test a recurrent network with a single item.
 */

void test_rnn_network_with_item(struct network *n, struct item *item, bool pprint, int scheme)
{
        n->status->error = 0.0;

        /* unfolded network */
        struct rnn_unfolded_network *un = n->unfolded_net;

        /* reset context groups and error signals */
        for (int j = 0; j < un->stack_size; j++)
                reset_recurrent_groups(un->stack[j]);

        /* stack pointer */
        int sp = 0;

        printf("\nItem: \"%s\" --  \"%s\"\n", item->name, item->meta);

        /* present all events of the current item */
        for (int j = 0; j < item->num_events; j++) {
                printf("\nEvent: \t%d\n", j);
                printf("\nInput: \t");
                if (pprint) {
                        pprint_vector(item->inputs[j], scheme);
                } else {
                        print_vector(item->inputs[j]);
                }

                copy_vector(un->stack[sp]->input->vector, item->inputs[j]);
                feed_forward(un->stack[sp], un->stack[sp]->input);

                if (item->targets[j]) {
                        n->status->error += n->output->err_fun->fun(
                                        un->stack[sp]->output,
                                        item->targets[j],
                                        n->target_radius,
                                        n->zero_error_radius);
                        printf("\nTarget: \t");
                        if (pprint) {
                                pprint_vector(item->targets[j], scheme);
                        } else {
                                print_vector(item->targets[j]);
                        }
                        printf("Output: \t");
                        if (pprint) {
                                pprint_vector(un->stack[sp]->output->vector, scheme);
                        } else {
                                print_vector(un->stack[sp]->output->vector);
                        }
                        pprintf("\nError: \t%lf", n->status->error);
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
