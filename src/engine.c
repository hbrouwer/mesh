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

/**************************************************************************
 *************************************************************************/
void train_network(struct network *n)
{
        pprintf("Epoch \t Error \t\t Weight Cost \t Gradient Lin.\n");
        pprintf("----- \t ----- \t\t ----------- \t -------------\n");

        n->learning_algorithm(n);
}

/**************************************************************************
 * Backpropagation training
 *************************************************************************/
void train_network_with_bp(struct network *n)
{
        /* make sure training set is order, if required */
        if (n->training_order == TRAIN_ORDERED)
                order_set(n->asp);

        uint32_t item_itr = 0;
        for (uint32_t epoch = 1; epoch <= n->max_epochs; epoch++) {
                n->status->epoch = epoch;
                n->status->prev_error = n->status->error;
                n->status->error = 0.0;

                /* reorder training set, if required */
                if (item_itr == 0) {
                        if (n->training_order == TRAIN_PERMUTED)
                                permute_set(n->asp);
                        if (n->training_order == TRAIN_RANDOMIZED)
                                randomize_set(n->asp);
                }

                /* train network on one batch */
                for (uint32_t i = 0; i < n->batch_size; i++) {
                        /* 
                         * Select training item, and set training item
                         * iterator to the beginning of the training set
                         * if necessary.
                         */
                        uint32_t item_idx = n->asp->order[item_itr++];
                        struct item *item = n->asp->items->elements[item_idx];
                        if (item_itr == n->asp->items->num_elements)
                                item_itr = 0;

                        /* train network on item */
                        train_ffn_network_with_item(n, item);
                }

                /* stop training, if threshold was reached */
                if (n->status->error < n->error_threshold) {
                        print_training_summary(n);
                        break;
                }

                /* update network's weights */
                n->update_algorithm(n);

                /* scale learning parameters */
                scale_learning_rate(n);
                scale_momentum(n);
                scale_weight_decay(n);

                print_training_progress(n);
        }
}

/**************************************************************************
 *************************************************************************/
void train_ffn_network_with_item(struct network *n, struct item *item)
{
        /* reset context groups */
        if (n->type == TYPE_SRN)
                reset_context_groups(n);

        for (uint32_t i = 0; i < item->num_events; i++) {
                /* 
                 * Shift context group chain, in case of 
                 * "Elman-towers".
                 */
                if (i > 0 && n->type == TYPE_SRN)
                        shift_context_groups(n);
                
                /* feed activation forward */
                copy_vector(n->input->vector, item->inputs[i]);
                feed_forward(n, n->input);

                /* 
                 * Skip error backpropagation, if there is no 
                 * target for the current event.
                 */
                if (!item->targets[i])
                        continue;

                struct group *g = n->output;
                struct vector *tv = item->targets[i];
                double tr = n->target_radius;
                double zr = n->zero_error_radius;

                /* backpropagate error */
                reset_ffn_error_signals(n);
                bp_output_error(g, tv, tr, zr);
                bp_backpropagate_error(n, g);
                
                /* 
                 * Update network error if all of the
                 * item's events have been processed.
                 */
                if (i == item->num_events - 1) {
                        double error = n->output->err_fun->fun(g, tv, tr, zr);
                        error /= n->batch_size;
                        n->status->error += error;
                }
        }
}

/**************************************************************************
 * Backpropagation Through Time (BPTT) training
 *************************************************************************/
void train_network_with_bptt(struct network *n)
{
        /* make sure training set is order, if required */
        if (n->training_order == TRAIN_ORDERED)
                order_set(n->asp);

        uint32_t item_itr = 0;
        for (uint32_t epoch = 1; epoch <= n->max_epochs; epoch++) {
                n->status->epoch = epoch;
                n->status->prev_error = n->status->error;
                n->status->error = 0.0;

                /* reorder training set, if required */
                if (item_itr == 0) {
                        if (n->training_order == TRAIN_PERMUTED)
                                permute_set(n->asp);
                        if (n->training_order == TRAIN_RANDOMIZED)
                                randomize_set(n->asp);
                }

                /* 
                 * Select training item, and set training item
                 * iterator to the beginning of the training set
                 * if necessary.
                 */
                uint32_t item_idx = n->asp->order[item_itr++];
                struct item *item = n->asp->items->elements[item_idx];
                if (item_itr == n->asp->items->num_elements)
                        item_itr = 0;
                
                /* train network on item */
                train_rnn_network_with_item(n, item);

                /* stop training, if threshold was reached */ 
                if (n->status->error < n->error_threshold) {
                        print_training_summary(n);
                        break;
                }

                /* scale learning parameters */
                scale_learning_rate(n);
                scale_momentum(n);
                scale_weight_decay(n);

                print_training_progress(n);
        }
}

/**************************************************************************
 *************************************************************************/
void train_rnn_network_with_item(struct network *n, struct item *item)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        un->sp = 0;

        /* reset recurrent groups, and error signals */
        reset_recurrent_groups(un->stack[un->sp]);
        reset_rnn_error_signals(n);

        for (uint32_t i = 0; i < item->num_events; i++) {
                /* feed activation forward */
                copy_vector(un->stack[un->sp]->input->vector, item->inputs[i]);
                feed_forward(un->stack[un->sp], un->stack[un->sp]->input);

                /* 
                 * Skip error backpropagation, if there is no 
                 * target for the current event.
                 */                
                if (!item->targets[i])
                        goto shift_stack;

                struct group *g = un->stack[un->sp]->output;
                struct vector *tv = item->targets[i];
                double tr = n->target_radius;
                double zr = n->zero_error_radius;
                
                /* compute error for current item */
                bp_output_error(g, tv, tr, zr);

                /* backpropagate error through time (if stack is full) */
                if (un->sp == un->stack_size - 1) {
                        /* backpropagate error */
                        bp_backpropagate_error(un->stack[un->sp], g);

                        /* update network's error */
                        n->status->error += n->output->err_fun->fun(g, tv, tr, zr);

                        /* sum gradients, and update weights */
                        rnn_sum_gradients(un);
                        n->update_algorithm(un->stack[0]);
                }

shift_stack:
                if (un->sp == un->stack_size - 1) {
                        rnn_shift_stack(un);
                } else {
                        un->sp++;
                }
        }
}

/**************************************************************************
 *************************************************************************/
void test_network(struct network *n)
{
        if (n->type == TYPE_FFN)
                test_ffn_network(n);
        if (n->type == TYPE_SRN)
                test_ffn_network(n);
        if (n->type == TYPE_RNN)
                test_rnn_network(n);
}

/**************************************************************************
 *************************************************************************/
void test_ffn_network(struct network *n)
{
        n->status->error = 0.0;
        uint32_t threshold_reached = 0;

        /* test network on all items in the current set */
        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                struct item *item = n->asp->items->elements[i];

                /* reset context groups */
                if (n->type == TYPE_SRN)
                        reset_context_groups(n);

                for (uint32_t j = 0; j < item->num_events; j++) {
                        /* 
                         * Shift context group chain, in case of 
                         * "Elman-towers".
                         */
                        if (j > 0 && n->type == TYPE_SRN)
                                shift_context_groups(n);

                        /* feed activation forward */
                        copy_vector(n->input->vector, item->inputs[j]);
                        feed_forward(n, n->input);

                        /* only compute network error for last event */
                        if (!(item->targets[j] && j == item->num_events - 1))
                                continue;

                        struct group *g = n->output;
                        struct vector *tv = item->targets[j];
                        double tr = n->target_radius;
                        double zr = n->zero_error_radius;

                        /* compute error */
                        double error = n->output->err_fun->fun(g, tv, tr, zr);
                        n->status->error += error;
                        if (error <= n->error_threshold)
                                threshold_reached++;
                }
                
        }

        print_testing_summary(n, threshold_reached);
}

/**************************************************************************
 *************************************************************************/
void test_rnn_network(struct network *n)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        n->status->error = 0.0;
        uint32_t threshold_reached = 0;

        /* test network on all items in the current set */
        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                struct item *item = n->asp->items->elements[i];

                /* reset recurrent groups */
                reset_recurrent_groups(un->stack[un->sp]);

                for (uint32_t j = 0; j < item->num_events; j++) {
                        /* feed activation forward */
                        copy_vector(un->stack[un->sp]->input->vector, item->inputs[j]);
                        feed_forward(un->stack[un->sp], un->stack[un->sp]->input);

                        /*
                         * Only compute error if there is a target
                         * for the current event.
                         */
                        if (!item->targets[j])
                                goto shift_stack;

                        struct group *g = un->stack[un->sp]->output;
                        struct vector *tv = item->targets[j];
                        double tr = n->target_radius;
                        double zr = n->zero_error_radius;

                        /* compute error */
                        double error = n->output->err_fun->fun(g, tv, tr, zr);
                        n->status->error += error;
                        if (error < n->error_threshold)
                                threshold_reached++;

shift_stack:
                        if (un->sp == un->stack_size - 1) {
                                rnn_shift_stack(un);
                        } else {
                                un->sp++;
                        }
                }
        }

        print_testing_summary(n, threshold_reached);
}

/**************************************************************************
 *************************************************************************/
void test_network_with_item(struct network *n, struct item *item,
                bool pprint, uint32_t scheme)
{
        if (n->type == TYPE_FFN)
                test_ffn_network_with_item(n, item, pprint, scheme);
        if (n->type == TYPE_SRN)
                test_ffn_network_with_item(n, item, pprint, scheme);
        if (n->type == TYPE_RNN)
                test_rnn_network_with_item(n, item, pprint, scheme);
}

/**************************************************************************
 *************************************************************************/
void test_ffn_network_with_item(struct network *n, struct item *item,
                bool pprint, uint32_t scheme)
{
        n->status->error = 0.0;

        /* reset context groups */
        if (n->type == TYPE_SRN)
                reset_context_groups(n);

        pprintf("Item: \t\"%s\" --  \"%s\"\n", item->name, item->meta);
        for (uint32_t i = 0; i < item->num_events; i++) {
                /* print event number, and input vector */
                pprintf("\n");
                pprintf("Event: \t%d\n", i);
                pprintf("Input: \t");
                if (pprint) {
                        pprint_vector(item->inputs[i], scheme);
                } else {
                        print_vector(item->inputs[i]);
                }

                /*
                 * Shift context group chain, in case of 
                 * "Elman-towers".
                 */
                if (i > 0 && n->type == TYPE_SRN)
                        shift_context_groups(n);

                /* feed activation forward */
                copy_vector(n->input->vector, item->inputs[i]);
                feed_forward(n, n->input);

                /* 
                 * Print target vector, if the current 
                 * event has one.
                 */
                if (item->targets[i]) {
                        pprintf("\n");
                        pprintf("Target: \t");
                        if (pprint) {
                                pprint_vector(item->targets[i], scheme);
                        } else  {
                                print_vector(item->targets[i]);
                        }
                }

                /* print output vector */
                pprintf("Output: \t");
                if (pprint) {
                        pprint_vector(n->output->vector, scheme);
                } else {
                        print_vector(n->output->vector);
                }

                /* only compute and print error for last event */
                if (!(i == item->num_events - 1) || !item->targets[i])
                        continue;

                struct group *g = n->output;
                struct vector *tv = item->targets[i];
                double tr = n->target_radius;
                double zr = n->zero_error_radius; 

                /* compute and print error */
                n->status->error += n->output->err_fun->fun(g, tv, tr, zr);
                pprintf("\n");
                pprintf("Error: \t%lf\n", n->status->error);
        }
}

/**************************************************************************
 *************************************************************************/
void test_rnn_network_with_item(struct network *n, struct item *item,
                bool pprint, uint32_t scheme)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        un->sp = 0;
        n->status->error = 0.0;
        
        /* reset recurrent groups */
        reset_recurrent_groups(un->stack[un->sp]);

        pprintf("Item: \t\"%s\" --  \"%s\"\n", item->name, item->meta);
        for (uint32_t i = 0; i < item->num_events; i++) {
                /* print event number, and input vector */
                pprintf("\n");
                pprintf("Event: \t%d\n", i);
                pprintf("Input: \t");
                if (pprint) {
                        pprint_vector(item->inputs[i], scheme);
                } else {
                        print_vector(item->inputs[i]);
                }

                /* feed activation vector */
                copy_vector(un->stack[un->sp]->input->vector, item->inputs[i]);
                feed_forward(un->stack[un->sp], un->stack[un->sp]->input);

                /* 
                 * Print target vector, if the current 
                 * event has one.
                 */                
                if (item->targets[i]) {
                        pprintf("\n");
                        pprintf("Target: \t");
                        if (pprint) {
                                pprint_vector(item->targets[i], scheme);
                        } else {
                                print_vector(item->targets[i]);
                        }
                }

                /* print output vector */
                pprintf("Output: \t");
                if (pprint) {
                        pprint_vector(un->stack[un->sp]->output->vector, scheme);
                } else {
                        print_vector(un->stack[un->sp]->output->vector);
                }

                /*
                 * Only compute and print error if there 
                 * is a target for the current event.
                 */
                if (!item->targets[i])
                        goto shift_stack;

                struct group *g = un->stack[un->sp]->output;
                struct vector *tv = item->targets[i];
                double tr = n->target_radius;
                double zr = n->zero_error_radius;

                /* compute and print error */
                n->status->error += n->output->err_fun->fun(g, tv, tr, zr);
                pprintf("\n");
                pprintf("Error: \t%lf\n", n->status->error);

shift_stack:
                if (un->sp == un->stack_size - 1) {
                        rnn_shift_stack(un);
                } else {
                        un->sp++;
                }
        }
}

/**************************************************************************
 *************************************************************************/
void print_training_progress(struct network *n)
{
        if (n->status->epoch == 1 || n->status->epoch % n->report_after == 0)
                pprintf("%.4d \t %lf \t %lf \t %lf\n",
                                n->status->epoch,
                                n->status->error,
                                n->status->weight_cost,
                                n->status->gradient_linearity);
}

/**************************************************************************
 *************************************************************************/
void print_training_summary(struct network *n)
{
        pprintf("\n");
        pprintf("Training finished after %d epoch(s) -- Network error: %f\n",
                        n->status->epoch, n->status->error);
}

/**************************************************************************
 *************************************************************************/
void print_testing_summary(struct network *n, uint32_t tr)
{
        pprintf("Number of items: \t\t %d\n",
                        n->asp->items->num_elements);
        pprintf("Total error: \t\t %lf\n",
                        n->status->error);
        pprintf("Error per example:\t\t %lf\n",
                        n->status->error / n->asp->items->num_elements);
        pprintf("# Items reached threshold:  %d (%.2lf%%)\n",
                        tr, ((double)tr / n->asp->items->num_elements) * 100.0);
}

/**************************************************************************
 *************************************************************************/
void scale_learning_rate(struct network *n)
{
        uint32_t sa = n->lr_scale_after * n->max_epochs;
        if (sa > 0 && n->status->epoch % sa == 0) {
                double lr = n->learning_rate;
                n->learning_rate *= n->lr_scale_factor;
                mprintf("Scaled learning rate ... \t  ( %lf => %lf )",
                                lr, n->learning_rate);
        }
}

/**************************************************************************
 *************************************************************************/
void scale_momentum(struct network *n)
{
        uint32_t sa = n->mn_scale_after * n->max_epochs;
        if (sa > 0 && n->status->epoch % sa == 0) {
                double mn = n->momentum;
                n->momentum *= n->mn_scale_factor;
                mprintf("Scaled momentum ... \t ( %lf => %lf )",
                                mn, n->momentum);
        }
}

/**************************************************************************
 *************************************************************************/
void scale_weight_decay(struct network *n)
{
        uint32_t sa = n->wd_scale_after * n->max_epochs;
        if (sa > 0 && n->status->epoch % sa == 0) {
                double wd = n->weight_decay;
                n->weight_decay *= n->wd_scale_factor;
                mprintf("Scaled weight decay ... \t ( %lf => %lf)",
                                wd, n->weight_decay);
        }
}
