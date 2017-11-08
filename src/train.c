/*
 * Copyright 2012-2017 Harm Brouwer <me@hbrouwer.eu>
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

#include <signal.h>
#include <stdbool.h>
#include <stdio.h>

#include "act.h"
#include "bp.h"
#include "main.h"
#include "train.h"

static bool keep_running = true;

void train_network(struct network *n)
{
        pprintf("Epoch \t Error \t\t Weight Cost \t Gradient Lin.\n");
        pprintf("----- \t ----- \t\t ----------- \t -------------\n");

        struct sigaction sa;
        sa.sa_handler = training_signal_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        sigaction(SIGINT, &sa, NULL);

        keep_running = true;

        n->learning_algorithm(n);

        sa.sa_handler = SIG_DFL;
        sigaction(SIGINT, &sa, NULL);
        
        cprintf("\n");
}

                /*************************
                 **** backpropagation ****
                 *************************/

void train_network_with_bp(struct network *n)
{
        /* make sure training set is ordered, if required */
        if (n->training_order == train_ordered)
                order_set(n->asp);

        uint32_t item_itr = 0;
        for (uint32_t epoch = 1; epoch <= n->max_epochs; epoch++) {
                n->status->epoch = epoch;
                n->status->prev_error = n->status->error;
                n->status->error = 0.0;

                /* reorder training set, if required */
                if (item_itr == 0) {
                        if (n->training_order == train_permuted)
                                permute_set(n->asp);
                        if (n->training_order == train_randomized)
                                randomize_set(n->asp);
                }

                /* train network on one batch */
                for (uint32_t i = 0; i < n->batch_size; i++) {
                        /* abort after signal */
                        if (!keep_running)
                                return;

                        /* 
                         * Select training item, and set training item
                         * iterator to the beginning of the training set if
                         * necessary.
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

void train_ffn_network_with_item(struct network *n, struct item *item)
{
        if (n->type == ntype_srn)
                reset_context_groups(n);
        for (uint32_t i = 0; i < item->num_events; i++) {
                /* feed activation forward */
                if (i > 0 && n->type == ntype_srn)
                        shift_context_groups(n);
                copy_vector(n->input->vector, item->inputs[i]);
                feed_forward(n, n->input);

                /* 
                 * Skip error backpropagation, if there is no target for the
                 * current event.
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
                 * Update network error if all of the item's events have
                 * been processed.
                 */
                if (i == item->num_events - 1) {
                        double error = n->output->err_fun->fun(g, tv, tr, zr);
                        error /= n->batch_size;
                        n->status->error += error;
                }

                /*
                 * In case of multi-stage training, clamp the desired target
                 * vector for the current stage, to the input group of the
                 * previous stage, and feed forward activation.
                 */
                if (n->ms_input) {
                        struct item *ms_item = find_array_element_by_name(
                                        n->ms_set->items, item->name);
                        if (!ms_item) {
                                eprintf("No matching item in multi-stage training\n");
                                continue;
                        }
                        copy_vector(n->ms_input->vector, ms_item->inputs[i]);
                        feed_forward(n, n->ms_input);
                }
        }
}

                /**************************************
                 **** backpropagation through time ****
                 **************************************/

void train_network_with_bptt(struct network *n)
{
        /* make sure training set is ordered, if required */
        if (n->training_order == train_ordered)
                order_set(n->asp);

        uint32_t item_itr = 0;
        for (uint32_t epoch = 1; epoch <= n->max_epochs; epoch++) {
                n->status->epoch = epoch;
                n->status->prev_error = n->status->error;
                n->status->error = 0.0;

                /* abort after signal */
                if (!keep_running)
                        return;

                /* reorder training set, if required */
                if (item_itr == 0) {
                        if (n->training_order == train_permuted)
                                permute_set(n->asp);
                        if (n->training_order == train_randomized)
                                randomize_set(n->asp);
                }

                /* 
                 * Select training item, and set training item iterator to
                 * the beginning of the training set if necessary.
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

void train_rnn_network_with_item(struct network *n, struct item *item)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        un->sp = 0;

        reset_recurrent_groups(un->stack[un->sp]);
        reset_rnn_error_signals(n);
        for (uint32_t i = 0; i < item->num_events; i++) {
                /* feed activation forward */
                copy_vector(un->stack[un->sp]->input->vector, item->inputs[i]);
                feed_forward(un->stack[un->sp], un->stack[un->sp]->input);

                /* 
                 * Skip error backpropagation, if there is no target for the
                 * current event.
                 */                
                if (!item->targets[i])
                        goto next_tick;

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

                // XXX: What about multi-stage training?

next_tick:
                shift_pointer_or_stack(n);
        }
}

void print_training_progress(struct network *n)
{
        if (n->status->epoch == 1 || n->status->epoch % n->report_after == 0)
                pprintf("%.4d \t\t %lf \t %lf \t %lf\n",
                        n->status->epoch,
                        n->status->error,
                        n->status->weight_cost,
                        n->status->gradient_linearity);
}

void print_training_summary(struct network *n)
{
        pprintf("\n");
        pprintf("Training finished after %d epoch(s) -- Network error: %f\n",
                        n->status->epoch, n->status->error);
}

void scale_learning_rate(struct network *n)
{
        uint32_t sa = n->lr_scale_after * n->max_epochs;
        if (sa > 0 && n->status->epoch % sa == 0) {
                double lr = n->learning_rate;
                n->learning_rate *= n->lr_scale_factor;
                mprintf("scaled learning rate ... \t ( %lf => %lf )\n",
                                lr, n->learning_rate);
        }
}

void scale_momentum(struct network *n)
{
        uint32_t sa = n->mn_scale_after * n->max_epochs;
        if (sa > 0 && n->status->epoch % sa == 0) {
                double mn = n->momentum;
                n->momentum *= n->mn_scale_factor;
                mprintf("scaled momentum ... \t ( %lf => %lf )\n",
                                mn, n->momentum);
        }
}

void scale_weight_decay(struct network *n)
{
        uint32_t sa = n->wd_scale_after * n->max_epochs;
        if (sa > 0 && n->status->epoch % sa == 0) {
                double wd = n->weight_decay;
                n->weight_decay *= n->wd_scale_factor;
                mprintf("scaled weight decay ... \t ( %lf => %lf)\n",
                                wd, n->weight_decay);
        }
}

void training_signal_handler(int32_t signal)
{
        cprintf("Training interrupted. Abort [y/n]");
        int32_t c = getc(stdin);
        getc(stdin); /* get newline */
        if (c == 'y' || c == 'Y')
                keep_running = false;
}
