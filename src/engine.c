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

/**************************************************************************
 *************************************************************************/
void train_network_with_bp(struct network *n)
{
        if (n->training_order == TRAIN_ORDERED)
                order_set(n->asp);

        uint32_t item_itr = 0;
        for (uint32_t epoch = 1; epoch <= n->max_epochs; epoch++) {
                n->status->epoch = epoch;
                n->status->prev_error = n->status->error;
                n->status->error = 0.0;

                if (item_itr == 0) {
                        if (n->training_order == TRAIN_PERMUTED)
                                permute_set(n->asp);
                        if (n->training_order == TRAIN_RANDOMIZED)
                                randomize_set(n->asp);
                }

                for (uint32_t i = 0; i < n->batch_size; i++) {
                        uint32_t item_idx = n->asp->order[item_itr++];
                        if (item_itr == n->asp->items->num_elements)
                                item_itr = 0;
                        struct item *item = n->asp->items->elements[item_idx];

                        if (n->type == TYPE_SRN)
                                reset_context_groups(n);
                        
                        train_ffn_network_with_item(n, item);
                }

                if (n->status->error < n->error_threshold) {
                        print_training_summary(n);
                        break;
                }

                n->update_algorithm(n);

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
        for (uint32_t i = 0; i < item->num_events; i++) {
                if (i > 0 && n->type == TYPE_SRN)
                        shift_context_groups(n);
                
                copy_vector(n->input->vector, item->inputs[i]);
                feed_forward(n, n->input);

                if (!item->targets[i])
                        continue;

                reset_error_signals(n);

                struct group *g = n->output;
                struct vector *tv = item->targets[i];
                double tr = n->target_radius;
                double zr = n->zero_error_radius;

                bp_output_error(g, tv, tr, zr);
                bp_backpropagate_error(n, g);
                
                if (i == item->num_events - 1) {
                        double error = n->output->err_fun->fun(g, tv, tr, zr);
                        error /= n->batch_size;
                        n->status->error += error;
                }
        }
}

/**************************************************************************
 *************************************************************************/
void train_network_with_bptt(struct network *n)
{
        struct rnn_unfolded_network *un = n->unfolded_net;

        if (n->training_order == TRAIN_ORDERED)
                order_set(n->asp);

        uint32_t item_itr = 0;
        for (uint32_t epoch = 1; epoch <= n->max_epochs; epoch++) {
                n->status->epoch = epoch;
                n->status->prev_error = n->status->error;
                n->status->error = 0.0;

                if (item_itr == 0) {
                        if (n->training_order == TRAIN_PERMUTED)
                                permute_set(n->asp);
                        if (n->training_order == TRAIN_RANDOMIZED)
                                randomize_set(n->asp);
                }

                uint32_t item_idx = n->asp->order[item_itr++];
                if (item_itr == n->asp->items->num_elements)
                        item_itr = 0;
                struct item *item = n->asp->items->elements[item_idx];

                un->sp = 0;
                for (uint32_t i = 0; i < un->stack_size; i++) {
                        reset_recurrent_groups(un->stack[i]);
                        reset_error_signals(un->stack[i]);
                }

                train_rnn_network_with_item(n, item);
                
                if (n->status->error < n->error_threshold) {
                        print_training_summary(n);
                        break;
                }

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

        for (uint32_t i = 0; i < item->num_events; i++) {
                uint32_t sp = un->sp;

                copy_vector(un->stack[sp]->input->vector, item->inputs[i]);
                feed_forward(un->stack[sp], un->stack[sp]->input);

                if (!item->targets[i])
                        goto cycle_stack;

                reset_error_signals(un->stack[sp]);
                        
                struct group *g = un->stack[sp]->output;
                struct vector *tv = item->targets[i];
                double tr = n->target_radius;
                double zr = n->zero_error_radius;
                       
                bp_output_error(g, tv, tr, zr);
                if (sp == un->stack_size - 1) {
                        bp_backpropagate_error(un->stack[un->sp], g);
                        n->status->error += n->output->err_fun->fun(g, tv, tr, zr);
                        rnn_sum_gradients(un);
                        n->update_algorithm(un->stack[0]);
                }
              
cycle_stack:
                if (un->sp == un->stack_size - 1) {
                        rnn_cycle_stack(un);
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

        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                struct item *item = n->asp->items->elements[i];

                if (n->type == TYPE_SRN)
                        reset_context_groups(n);

                for (uint32_t j = 0; j < item->num_events; j++) {
                        if (j > 0 && n->type == TYPE_SRN)
                                shift_context_groups(n);

                        copy_vector(n->input->vector, item->inputs[j]);
                        feed_forward(n, n->input);

                        if (!(item->targets[j] && j == item->num_events - 1))
                                continue;

                        struct group *g = n->output;
                        struct vector *tv = item->targets[j];
                        double tr = n->target_radius;
                        double zr = n->zero_error_radius;

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

        un->sp = 0;
        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                uint32_t sp = un->sp;
                struct item *item = n->asp->items->elements[i];

                for (uint32_t j = 0; j < un->stack_size; j++)
                        reset_recurrent_groups(un->stack[j]);

                for (uint32_t j = 0; j < item->num_events; j++) {
                        copy_vector(un->stack[sp]->input->vector, item->inputs[j]);
                        feed_forward(un->stack[sp], un->stack[sp]->input);

                        if (!item->targets[j])
                                goto cycle_stack;

                        struct group *g = un->stack[sp]->output;
                        struct vector *tv = item->targets[j];
                        double tr = n->target_radius;
                        double zr = n->zero_error_radius;

                        double error = n->output->err_fun->fun(g, tv, tr, zr);
                        n->status->error += error;
                        if (error < n->error_threshold)
                                threshold_reached++;
                }
                       
cycle_stack:
                if (sp == un->stack_size - 1) {
                        rnn_cycle_stack(un);
                } else {
                        un->sp++;
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

        if (n->type == TYPE_SRN)
                reset_context_groups(n);

        pprintf("Item: \t\"%s\" --  \"%s\"\n", item->name, item->meta);

        for (uint32_t i = 0; i < item->num_events; i++) {
                pprintf("\n");
                pprintf("Event: \t%d\n", i);
                pprintf("Input: \t");
                if (pprint) {
                        pprint_vector(item->inputs[i], scheme);
                } else {
                        print_vector(item->inputs[i]);
                }

                if (i > 0 && n->type == TYPE_SRN)
                        shift_context_groups(n);

                copy_vector(n->input->vector, item->inputs[i]);
                feed_forward(n, n->input);

                if (!item->targets[i])
                        continue;

                pprintf("\n");
                pprintf("Target: \t");
                if (pprint) {
                        pprint_vector(item->targets[i], scheme);
                } else  {
                        print_vector(item->targets[i]);
                }
                pprintf("Output: \t");
                if (pprint) {
                        pprint_vector(n->output->vector, scheme);
                } else {
                        print_vector(n->output->vector);
                }

                if (!(i == item->num_events - 1))
                        continue;

                struct group *g = n->output;
                struct vector *tv = item->targets[i];
                double tr = n->target_radius;
                double zr = n->zero_error_radius; 

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
        
        n->status->error = 0.0;
        
        un->sp = 0;
        for (uint32_t i = 0; i < un->stack_size; i++)
                reset_recurrent_groups(un->stack[i]);

        pprintf("Item: \t\"%s\" --  \"%s\"\n", item->name, item->meta);


        for (uint32_t i = 0; i < item->num_events; i++) {
                uint32_t sp = un->sp;

                pprintf("\n");
                pprintf("Event: \t%d\n", i);
                pprintf("Input: \t");
                if (pprint) {
                        pprint_vector(item->inputs[i], scheme);
                } else {
                        print_vector(item->inputs[i]);
                }

                copy_vector(un->stack[sp]->input->vector, item->inputs[i]);
                feed_forward(un->stack[sp], un->stack[sp]->input);

                if (!item->targets[i])
                        goto cycle_stack;
             
                pprintf("\n");
                pprintf("Target: \t");
                if (pprint) {
                        pprint_vector(item->targets[i], scheme);
                } else {
                        print_vector(item->targets[i]);
                }
                pprintf("Output: \t");
                if (pprint) {
                        pprint_vector(un->stack[sp]->output->vector, scheme);
                } else {
                        print_vector(un->stack[sp]->output->vector);
                }

                struct group *g = un->stack[sp]->output;
                struct vector *tv = item->targets[i];
                double tr = n->target_radius;
                double zr = n->zero_error_radius;

                n->status->error += n->output->err_fun->fun(g, tv, tr, zr);
                pprintf("\n");
                pprintf("Error: \t%lf\n", n->status->error);

cycle_stack:
                un->sp++;
                if (sp == un->stack_size - 1) {
                        rnn_cycle_stack(un);
                } else {
                        un->sp++;
                }
        }
}
