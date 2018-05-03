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
        cprintf("\n");
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
        uint32_t item_itr = 0;
        for (uint32_t epoch = 1; epoch <= n->pars->max_epochs; epoch++) {
                n->status->epoch      = epoch;
                n->status->prev_error = n->status->error;
                n->status->error      = 0.0;

                /* reorder training set, if required */
                if (item_itr == 0) reorder_training_set(n);

                /* train network on one batch */
                for (uint32_t i = 0; i < n->pars->batch_size; i++) {
                        if (!keep_running)
                                return;
                
                        /* select next item */
                        uint32_t item_idx = n->asp->order[item_itr++];
                        struct item *item = n->asp->items->elements[item_idx];
                        if (item_itr == n->asp->items->num_elements)
                                item_itr = 0;
                
                        /* train network with item */
                        train_ffn_network_with_item(n, item);
                }

                /* stop training, if threshold was reached */
                if (n->status->error < n->pars->error_threshold) {
                        print_training_summary(n);
                        break;
                }

                /* update weights, and parameters */
                n->update_algorithm(n);
                scale_learning_rate(n);
                scale_momentum(n);
                scale_weight_decay(n);

                print_training_progress(n);
        }
}

void train_ffn_network_with_item(struct network *n, struct item *item)
{
        if (n->flags->type == ntype_srn)
                reset_context_groups(n);
        for (uint32_t i = 0; i < item->num_events; i++) {
                if (i > 0 && n->flags->type == ntype_srn)
                        shift_context_groups(n);
                copy_vector(n->input->vector, item->inputs[i]);
                feed_forward(n, n->input);

                /* skip backpropagation, if there is no target */
                if (!item->targets[i])
                        continue;

                struct group *g   = n->output;
                struct vector *tv = item->targets[i];
                double tr         = n->pars->target_radius;
                double zr         = n->pars->zero_error_radius;

                /* backpropagate error */
                reset_ffn_error_signals(n);
                bp_output_error(g, tv, tr, zr);
                bp_backpropagate_error(n, g);

                /* update network error at the last event */
                if (i == item->num_events - 1) {
                        double error = n->output->err_fun->fun(g, tv, tr, zr);
                        error /= n->pars->batch_size;
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
                        if (ms_item) {
                                copy_vector(n->ms_input->vector,
                                        ms_item->inputs[i]);
                                feed_forward(n, n->ms_input);       
                        } else {
                                eprintf("No matching item in multi-stage training\n");
                        }
                }
        }
}

                /**************************************
                 **** backpropagation through time ****
                 **************************************/

/*
 * TODO: Check logic.
 */
void train_network_with_bptt(struct network *n)
{
        uint32_t item_itr = 0;
        for (uint32_t epoch = 1; epoch <= n->pars->max_epochs; epoch++) {
                if (!keep_running)
                        return;

                n->status->epoch      = epoch;
                n->status->prev_error = n->status->error;
                n->status->error      = 0.0;
                
                /* reorder training set, if required */
                if (item_itr == 0) reorder_training_set(n);

                /* select next item */
                uint32_t item_idx = n->asp->order[item_itr++];
                struct item *item = n->asp->items->elements[item_idx];
                if (item_itr == n->asp->items->num_elements)
                        item_itr = 0;

                /* train network with item */
                train_rnn_network_with_item(n, item);

                /* stop training, if threshold was reached */ 
                if (n->status->error < n->pars->error_threshold) {
                        print_training_summary(n);
                        break;
                }

                /* update learning parameters */
                scale_learning_rate(n);
                scale_momentum(n);
                scale_weight_decay(n);

                print_training_progress(n);
        }
}

/*
 * TODO: Check logic.
 */
void train_rnn_network_with_item(struct network *n, struct item *item)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        
        reset_stack_pointer(n);
        reset_recurrent_groups(un->stack[un->sp]);
        reset_rnn_error_signals(n);
        for (uint32_t i = 0; i < item->num_events; i++) {
                copy_vector(
                        un->stack[un->sp]->input->vector,
                        item->inputs[i]);
                feed_forward(
                        un->stack[un->sp],
                        un->stack[un->sp]->input);

                /* skip backpropagation, if there is no target */             
                if (!item->targets[i])
                        goto next_tick;

                struct group *g   = un->stack[un->sp]->output;
                struct vector *tv = item->targets[i];
                double tr         = n->pars->target_radius;
                double zr         = n->pars->zero_error_radius;

                /* compute error for current item */
                bp_output_error(g, tv, tr, zr);

                /* 
                 * If unfolded network stack is full, backpropagate error
                 * through time, update network error, and update weights.
                 */
                if (un->sp == un->stack_size - 1) {
                        /* backpropagate error */
                        bp_backpropagate_error(un->stack[un->sp], g);
                        /* update network error */
                        n->status->error +=
                                n->output->err_fun->fun(g, tv, tr, zr);
                        /* update weights */        
                        rnn_sum_gradients(un);
                        n->update_algorithm(un->stack[0]);
                }

                // TODO: Multi-stage training.

next_tick:
                shift_pointer_or_stack(n);
        }
}

void reorder_training_set(struct network *n)
{
        switch (n->flags->training_order) {
        case train_ordered:
                order_set(n->asp);
                break;
        case train_permuted:
                permute_set(n->asp);
                break;
        case train_randomized:
                randomize_set(n->asp);
                break;
        }
}

void print_training_progress(struct network *n)
{
        if (n->status->epoch == 1 || 
                n->status->epoch % n->pars->report_after == 0)
                pprintf("%.4d \t\t %lf \t %lf \t %lf\n",
                        n->status->epoch,
                        n->status->error,
                        n->status->weight_cost,
                        n->status->gradient_linearity);
}

void print_training_summary(struct network *n)
{
        cprintf("\nTraining finished after %d epoch(s) -- Network error: %f\n",
                        n->status->epoch,
                        n->status->error);
}

void scale_learning_rate(struct network *n)
{
        uint32_t sa = n->pars->lr_scale_after * n->pars->max_epochs;
        if (sa > 0 && n->status->epoch % sa == 0) {
                double lr = n->pars->learning_rate;
                n->pars->learning_rate *= n->pars->lr_scale_factor;
                mprintf("Scaled learning rate ... \t ( %lf => %lf )\n",
                        lr, n->pars->learning_rate);
        }
}

void scale_momentum(struct network *n)
{
        uint32_t sa = n->pars->mn_scale_after * n->pars->max_epochs;
        if (sa > 0 && n->status->epoch % sa == 0) {
                double mn = n->pars->momentum;
                n->pars->momentum *= n->pars->mn_scale_factor;
                mprintf("Scaled momentum ... \t ( %lf => %lf )\n",
                        mn, n->pars->momentum);
        }
}

void scale_weight_decay(struct network *n)
{
        uint32_t sa = n->pars->wd_scale_after * n->pars->max_epochs;
        if (sa > 0 && n->status->epoch % sa == 0) {
                double wd = n->pars->weight_decay;
                n->pars->weight_decay *= n->pars->wd_scale_factor;
                mprintf("Scaled weight decay ... \t ( %lf => %lf)\n",
                        wd, n->pars->weight_decay);
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
