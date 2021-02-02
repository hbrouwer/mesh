/*
 * Copyright 2012-2021 Harm Brouwer <me@hbrouwer.eu>
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
#include "rnn_unfold.h"
#include "modules/dss.h"

                /****************
                 **** engine ****
                 ****************/

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
This glues together the processing logic for different network types.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */                

void clamp_input_vector(struct network *n, struct vector *input)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        switch(n->flags->type) {
        case ntype_ffn:
                /* fall through */
        case ntype_srn:
                copy_vector(input, n->input->vector);
                break;
        case ntype_rnn:
                copy_vector(input, un->stack[un->sp]->input->vector);
                break;
        }
}

void reset_ticks(struct network *n)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        if (n->flags->dcs)
                reset_dcs_vectors(n);
        switch(n->flags->type) {
        case ntype_ffn:
                break;
        case ntype_srn:
                reset_context_groups(n);
                break;
        case ntype_rnn:
                reset_stack_pointer(n);
                reset_recurrent_groups(un->stack[0]);
                break;
        }
}

void next_tick(struct network *n)
{
        if (n->flags->dcs)
                update_dcs_vectors(n);
        switch(n->flags->type) {
        case ntype_ffn:
                break;
        case ntype_srn:
                shift_context_groups(n);
                break;
        case ntype_rnn:
                shift_pointer_or_stack(n);
                break;
        }
}

void forward_sweep(struct network *n)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        switch(n->flags->type) {
        case ntype_ffn:
                /* fall through */
        case ntype_srn:
                feed_forward(n, n->input);
                break;
        case ntype_rnn:
                feed_forward(un->stack[un->sp], un->stack[un->sp]->input);
                break;
        }
}

double output_error(struct network *n, struct vector *target)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        double error = 0.0;
        switch(n->flags->type) {
        case ntype_ffn:
                /* fall through */
        case ntype_srn:
                error = n->output->err_fun->fun(n, n->output, target);
                break;
        case ntype_rnn:
                error = un->stack[un->sp]->output->err_fun->fun(
                        n, un->stack[un->sp]->output, target);
                break;
        }
        return error;
}

struct vector *output_vector(struct network *n)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        struct vector *v = NULL;
        switch(n->flags->type) {
        case ntype_ffn:
                /* fall through */
        case ntype_srn:
                v = n->output->vector;
                break;
        case ntype_rnn:
                v = un->stack[un->sp]->output->vector;
                break;     
        }
        return v;
}

struct group *find_network_group_by_name(struct network *n, char *name)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        struct group *g = NULL;
        switch(n->flags->type) {
        case ntype_ffn:
                /* fall through */
        case ntype_srn:
                g = find_array_element_by_name(n->groups, name);
                break;
        case ntype_rnn:
                g = find_array_element_by_name(
                        un->stack[un->sp]->groups, name);
                break;
        }
        return g;
}

void reset_error_signals(struct network *n)
{
        switch(n->flags->type) {
        case ntype_ffn:
                /* fall through */
        case ntype_srn:
                reset_ffn_error_signals(n);
                break;
        case ntype_rnn:
                reset_rnn_error_signals(n);
                break;
        }
}

void backward_sweep(struct network *n)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        switch(n->flags->type) {
        case ntype_ffn:
                /* fall through */
        case ntype_srn:
                bp_backpropagate_error(n, n->output);
                break;
        case ntype_rnn:
                for (int32_t i = un->sp; i >= 0; i--)
                        bp_backpropagate_error(
                                un->stack[i], 
                                un->stack[i]->output);
                break;     
        }
}

void update_weights(struct network *n)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        switch(n->flags->type) {
        case ntype_ffn:
                /* fall through */
        case ntype_srn:
                n->update_algorithm(n);
                break;
        case ntype_rnn:
                rnn_sum_and_reset_gradients(un);
                n->update_algorithm(un->stack[0]);
                break;
        }
}

void inject_error(struct network *n, struct vector *target)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        switch(n->flags->type) {
        case ntype_ffn:
                /* fall through */
        case ntype_srn:
                bp_output_error(n, n->output, target);
                break;
        case ntype_rnn:
                bp_output_error(n, un->stack[un->sp]->output, target);
                break;
        } 
}

void two_stage_forward_sweep(struct network *n, struct item *item,
        uint32_t event)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        struct network *np = NULL;
        switch(n->flags->type) {
        case ntype_ffn:
                /* fall through */
        case ntype_srn:
                np = n;
                break;
        case ntype_rnn:
                np = un->stack[un->sp];
                break;
        }
        struct group *ts_fw_group = find_network_group_by_name(
                np,
                n->ts_fw_group->name);
        struct item  *ts_fw_item  = find_array_element_by_name(
                n->ts_fw_set->items,
                item->name);
        copy_vector(ts_fw_item->inputs[event], ts_fw_group->vector);
        feed_forward(np, ts_fw_group);
}

void two_stage_backward_sweep(struct network *n, struct item *item,
        uint32_t event)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        struct network *np = NULL;
        switch(n->flags->type) {
        case ntype_ffn:
                /* fall through */
        case ntype_srn:
                np = n;
                break;
        case ntype_rnn:
                np = un->stack[un->sp];
                break;
        }
        struct group *ts_bw_group = find_network_group_by_name(
                np,
                n->ts_bw_group->name);
        struct item  *ts_bw_item  = find_array_element_by_name(
                n->ts_bw_set->items,
                item->name);
        bp_output_error(n, ts_bw_group, ts_bw_item->targets[event]);
        bp_backpropagate_error(np, ts_bw_group);
}
