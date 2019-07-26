/*
 * Copyright 2012-2019 Harm Brouwer <me@hbrouwer.eu>
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
#include "engine.h"
#include "main.h"
#include "rnn_unfold.h"
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

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Standard backpropagation training. For each event of an item, error is
injected and backpropagated if a target pattern is present. Weights are
updated after each batch.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

void train_network_with_bp(struct network *n)
{
        uint32_t z = 0;
        for (uint32_t epoch = 1; epoch <= n->pars->max_epochs; epoch++) {
                n->status->epoch      = epoch;
                n->status->prev_error = n->status->error;
                n->status->error      = 0.0;
                if (z == 0)
                        reorder_training_set(n);
                for (uint32_t i = 0; i < n->pars->batch_size; i++) {
                        if (!keep_running)
                                return;
                        uint32_t x = n->asp->order[z++];
                        struct item *item = n->asp->items->elements[x];
                        if (z == n->asp->items->num_elements)
                                z = 0;
                        reset_ticks(n);
                        for (uint32_t j = 0; j < item->num_events; j++) {
                                if (j > 0)
                                        next_tick(n);
                                clamp_input_vector(n, item->inputs[j]);
                                forward_sweep(n);
                                if (!item->targets[j])
                                        continue;
                                reset_error_signals(n);
                                inject_error(n, item->targets[j]);
                                backward_sweep(n);
                                if (n->ts_bw_group) /* two-stage backward sweep */
                                        two_stage_backward_sweep(n, item, j);
                                if (j == item->num_events - 1)
                                        n->status->error += output_error(n,
                                                item->targets[j])
                                                / n->pars->batch_size;
                                if (n->ts_fw_group) /* two-stage forward sweep */
                                        two_stage_forward_sweep(n, item, j);
                        }
                }
                if (n->status->error < n->pars->error_threshold) {
                        print_training_summary(n);
                        break;
                }
                update_weights(n);
                scale_learning_rate(n);
                scale_momentum(n);
                scale_weight_decay(n);
                print_training_progress(n);
        }
}

                /**************************************
                 **** backpropagation through time ****
                 **************************************/

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Backpropagation Through Time (BPTT_ training. For each event of an item,
error is injected if a target pattern is present. Error is only
backpropagated once all events of an item have been processed. Weights are
updated after each batch.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

void train_network_with_bptt(struct network *n)
{
        uint32_t z = 0;
        for (uint32_t epoch = 1; epoch <= n->pars->max_epochs; epoch++) {
                if (!keep_running)
                        return;
                n->status->epoch      = epoch;
                n->status->prev_error = n->status->error;
                n->status->error      = 0.0;
                if (z == 0)
                        reorder_training_set(n);
                for (uint32_t i = 0; i < n->pars->batch_size; i++) {
                        if (!keep_running)
                                return;                 
                        uint32_t x = n->asp->order[z++];
                        struct item *item = n->asp->items->elements[x];
                        if (z == n->asp->items->num_elements)
                                z = 0;
                        reset_ticks(n);
                        reset_error_signals(n);
                        for (uint32_t j = 0; j < item->num_events; j++) {
                                if (j > 0)
                                        next_tick(n);
                                clamp_input_vector(n, item->inputs[j]);
                                forward_sweep(n);
                                if (!item->targets[j])
                                        continue;
                                inject_error(n, item->targets[j]);
                                if (n->unfolded_net->sp
                                        == n->unfolded_net->stack_size - 1
                                        || j == item->num_events - 1) {
                                        // inject_error(n, item->targets[j]);
                                        backward_sweep(n);
                                        if (n->ts_bw_group) /* two-stage backward sweep */
                                                two_stage_backward_sweep(n, item, j);
                                        n->status->error += output_error(n,
                                                item->targets[j])
                                                / n->pars->batch_size;                                          
                                }
                                if (n->ts_fw_group) /* two-stage forward sweep */
                                        two_stage_forward_sweep(n, item, j);  
                        }
                }
                if (n->status->error < n->pars->error_threshold) {
                        print_training_summary(n);
                        break;
                }                       
                update_weights(n);
                scale_learning_rate(n);
                scale_momentum(n);
                scale_weight_decay(n);
                print_training_progress(n);
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
