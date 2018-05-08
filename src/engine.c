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

#include "act.h"
#include "bp.h"
#include "engine.h"
#include "rnn_unfold.h"

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
        case ntype_ffn: /* fall through */
        case ntype_srn:
                copy_vector(n->input->vector, input);
                break;
        case ntype_rnn:
                copy_vector(un->stack[un->sp]->input->vector, input);
                break;
        }
}

void reset_ticks(struct network *n)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
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

void reset_error_signals(struct network *n)
{
        switch(n->flags->type) {
        case ntype_ffn: /* fall through */
        case ntype_srn:
                reset_ffn_error_signals(n);
                break;
        case ntype_rnn:
                reset_rnn_error_signals(n);
                break;
        }
}

void forward_sweep(struct network *n)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        switch(n->flags->type) {
        case ntype_ffn: /* fall through */
        case ntype_srn:
                feed_forward(n, n->input);
                break;
        case ntype_rnn:
                feed_forward(un->stack[un->sp], un->stack[un->sp]->input);
                break;
        }
}

void next_tick(struct network *n)
{
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

void inject_error(struct network *n, struct vector *target)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        double tr = n->pars->target_radius;
        double zr = n->pars->zero_error_radius;
        switch(n->flags->type) {
        case ntype_ffn: /* fall through */
        case ntype_srn:
                bp_output_error(n->output, target, tr, zr);
                break;
        case ntype_rnn:
                bp_output_error(un->stack[un->sp]->output, target, tr, zr);
                break;
        } 
}

double output_error(struct network *n, struct vector *target)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        double tr = n->pars->target_radius;
        double zr = n->pars->zero_error_radius;
        switch(n->flags->type) {
        case ntype_ffn: /* fall through */
        case ntype_srn:
                return n->output->err_fun->fun(n->output, target, tr, zr);
        case ntype_rnn:
                return un->stack[un->sp]->output->err_fun->fun(
                        un->stack[un->sp]->output, target, tr, zr);
        }
}

struct vector *output_vector(struct network *n)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        switch(n->flags->type) {
        case ntype_ffn: /* fall through */
        case ntype_srn:
                return(n->output->vector);
                break;
        case ntype_rnn:
                return(un->stack[un->sp]->output->vector);
                break;     
        }
}

struct group *find_network_group_by_name(struct network *n, char *name)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        struct group *g;
        switch(n->flags->type) {
        case ntype_ffn: /* fall through */
        case ntype_srn:
                g = find_array_element_by_name(n->groups, name);
                break;
        case ntype_rnn:
                g = find_array_element_by_name(un->stack[un->sp]->groups,
                        name);
                break;
        }
        return g;
}

void backward_sweep(struct network *n)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        switch(n->flags->type) {
        case ntype_ffn: /* fall through */
        case ntype_srn:
                bp_backpropagate_error(n, n->output);
                break;
        case ntype_rnn:
                for (int32_t i = un->sp; i >= 0; i--)
                        bp_backpropagate_error(un->stack[i], 
                                un->stack[i]->output);
                break;     
        }
}

void update_weights(struct network *n)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        switch(n->flags->type) {
        case ntype_ffn: /* fall through */
        case ntype_srn:
                n->update_algorithm(n);
                break;
        case ntype_rnn:
                rnn_sum_and_reset_gradients(un);
                n->update_algorithm(un->stack[0]);
                break;
        }
}
