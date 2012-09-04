/*
 * train.c
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

#include "ffn_unfold.h"
#include "math.h"
#include "set.h"
#include "train.h"

#define REPORT_AFTER_PERCENTAGE 0.05

void train_network(struct network *n)
{
        mprintf("starting training of network: [%s]", n->name);

        if ((n->learning_algorithm == train_bptt_epochwise)
                        || (n->learning_algorithm == train_bptt_truncated))
                n->unfolded_net = ffn_init_unfolded_network(n);

        n->learning_algorithm(n);
}

void test_network(struct network *n)
{
        mprintf("starting testing of network: [%s]", n->name);

        for (int i = 0; i < n->test_set->num_elements; i++) {
                struct element *e = n->test_set->elements[i];

                rprintf("testing item: %d -- \"%s\"", i, e->name);

                for (int j = 0; j < e->num_events; j++) {
                        copy_vector(n->input->vector, e->inputs[j]);
                        if (e->targets[j] != NULL)
                                copy_vector(n->target, e->targets[j]);
                
                        rprintf("");

                        if (e->targets[j] != NULL) {
                                rprintf("target vector:");
                                print_vector(n->target);
                        }

                        feed_forward(n, n->input);

                        print_units(n);
                }
        }

        print_weights(n);
        print_weight_stats(n);

        double mse = mean_squared_error(n);
        pprintf("MSE: [%lf]", mse);
}

/*
 * Total network error (or mean squared error), as described in:
 *
 * Williams R. J. and Zipser. D. (1990). Gradient-based learning algoritms
 *   for recurrent connectionist networks. (Technical Report NU-CCS-90-0).
 *   Boston: Northeastern University, College of Computer Science.
 */
double mean_squared_error(struct network *n)
{
        double mse = 0.0;
        for (int i = 0; i < n->training_set->num_elements; i++) {
                struct element *e = n->training_set->elements[i];

                for (int j = 0; j < e->num_events; j++) {
                        copy_vector(n->input->vector, e->inputs[j]);
                        feed_forward(n, n->input);

                        if (e->targets[j]) {
                                copy_vector(n->target, e->targets[j]);
                                mse += squared_error(n->output->vector, n->target);
                        }
                }
        }

        return mse / n->training_set->num_elements;
}

void report_mean_squared_error(int epoch, double mse)
{
        pprintf("epoch: [%d] | MSE: [%lf]", epoch, mse);
}

/*
 * Feed activation forward. For a specified group G, the activation is
 * computed for each unit in all of the groups towards which G maintains an
 * outgoing projection. This process is recursively repeated for all groups
 * that have a projection coming in from G. If the first G is the input
 * group of the network, activation should be propagated forwar through all
 * of the network's hidden groups to the output group:
 *
 * ##########    ##########    ##########
 * # input  #---># hidden #---># output #
 * ##########    ##########    ##########
 *
 * The activation of a unit in a group G' is computed by summing the 
 * weighted activations of all of the the incoming projections of G', and
 * applying an activation (or squashing) function to the summed total. Note
 * that activation functions may be different for hidden and output units.
 *
 * If G has an Elman-type projection to group G', copy the vector of G into
 * G'.
 */

void feed_forward(struct network *n, struct group *g)
{
        if (g->elman_proj)
                copy_vector(g->elman_proj->vector, g->vector);

        for (int i = 0; i < g->out_projs->num_elements; i++) {
                struct group *rg = g->out_projs->elements[i]->to;
                for (int j = 0; j < rg->vector->size; j++) {
                        rg->vector->elements[j] = unit_activation(n, rg, j);

                        if (rg != n->output)
                                rg->vector->elements[j] =
                                        n->act_fun(rg->vector, j);
                        else
                                rg->vector->elements[j] =
                                        n->out_act_fun(rg->vector, j);
                }
        }

        for (int i = 0; i < g->out_projs->num_elements; i++)
                feed_forward(n, g->out_projs->elements[i]->to);
}

double unit_activation(struct network *n, struct group *g, int u)
{
        double act = 0.0;
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct group *pg = g->inc_projs->elements[i]->to;
                struct matrix *w = g->inc_projs->elements[i]->weights;
                for (int j = 0; j < pg->vector->size; j++)
                        act += w->elements[j][u] * pg->vector->elements[j];
        }

        return act;
}

/*
 * Backpropagation training. Three types of backpropagation are provided:
 *
 * (1) Standard or regular backpropagation;
 *
 *     For each training element, activation is fed forward to the
 *     output group, and error is backpropagated from the output to the
 *     input group. Weights are adjusted after a specified training epoch
 *     length. If the epoch length is one iteration, this is similar to
 *     adjusting weights after each training instance.
 *
 * (2) Epochwise backpropagation through time
 *
 *     Activation of all training items in an epoch is fed forward to the 
 *     output group. For each item, the activation of recurrent groups as
 *     evoked by the previous training item is also taken into account.
 *     When the last item of an epoch is reached, the error gradient is
 *     computed for each time step in the epoch, by means of propagating
 *     error at the time step's output group to the time step's input group
 *     (i.e., this is essentially the standard backpropagation procedure 
 *     applied to the individual time steps). The error gradient for the
 *     entire epoch is then determined by summing the gradients at the
 *     different time steps, and weights are adjusted according to this
 *     total error gradient.
 *
 * (3) Truncated backpropagation through time.
 *
 *     This training procedure is essentially similar to that of epochwise
 *     backpropagation through time. However, instead of presenting the
 *     network with items for the course of an epoch, a fixed history 
 *     length is used. After activation is fed forward for all items, error 
 *     is  backpropagated from the latest time step all the way down to the
 *     earliest time step. Importantly, and in contrast to epochwise
 *     backpropagation through time, only the output error at the latest 
 *     time step is taken into account. No error is injected at
 *     earlier time steps. The error gradient for the entire history length
 *     is then determined by summing individual error gradients, and weights
 *     are adjusted accordingly.
 */

void train_bp(struct network *n)
{
        int report_after = n->max_epochs * REPORT_AFTER_PERCENTAGE;

        int item = 0, event = 0;
        for (int epoch = 1; epoch <= n->max_epochs; epoch++) {
                for (int i = 0; i < n->epoch_length; i++) {
                        struct element *e = n->training_set->elements[item];
                        
                        copy_vector(n->input->vector, e->inputs[event]);
                        feed_forward(n, n->input);
                        
                        if (e->targets[event]) {
                                copy_vector(n->target, e->targets[event]);
                                struct vector *error = n->error_measure(n);
                                backpropagate_error(n, n->output, error);
                                dispose_vector(error);
                        }

                        event++;
                        if (event == e->num_events) {
                                event = 0;
                                item++;
                        }
                        if (item == n->training_set->num_elements)
                                item = 0;
                }
                adjust_weights(n, n->output);

                double mse = mean_squared_error(n);
                if (epoch == 1 || epoch % report_after == 0)
                        report_mean_squared_error(epoch, mse);
                if (mse < n->mse_threshold)
                        break;
        }
}

void train_bptt_epochwise(struct network *n)
{
        int item = 0, event = 0;
        for (int epoch = 1; epoch <= n->max_epochs; epoch++) {
                for (int i = 0; i < n->epoch_length; i++) {
                        struct element *e = n->training_set->elements[item];

                        if (i > 0)
                                ffn_connect_duplicate_networks(
                                                n->unfolded_net,
                                                n->unfolded_net->stack[i - 1],
                                                n->unfolded_net->stack[i]);
                        
                        copy_vector(n->unfolded_net->stack[i]->input->vector, 
                                        e->inputs[event]);
                        feed_forward(n->unfolded_net->stack[i],
                                        n->unfolded_net->stack[i]->input);
                        
                        if (e->targets[event])
                                copy_vector(n->unfolded_net->stack[i]->target,
                                                e->targets[event]);

                        event++;
                        if (event == e->num_events) {
                                event = 0;
                                item++;
                        }
                        if (item == n->training_set->num_elements)
                                item = 0;
                }

                for (int i = n->epoch_length - 1; i >= 0; i--) {
                        struct network *ns = n->unfolded_net->stack[i];
                        struct vector *error = n->error_measure(ns);
                        backpropagate_error(ns, ns->output, error);
                        dispose_vector(error);
                }

                ffn_sum_deltas(n->unfolded_net);
                adjust_weights(n->unfolded_net->stack[0],
                                n->unfolded_net->stack[0]->output);
                
                for (int i = 1; i < n->epoch_length; i++)
                        ffn_disconnect_duplicate_networks(
                                        n->unfolded_net,
                                        n->unfolded_net->stack[i - 1],
                                        n->unfolded_net->stack[i]);
        }
}

void train_bptt_truncated(struct network *n)
{
        int item = 0, event = 0, h = 0;
        for (int epoch = 1; epoch <= n->max_epochs; epoch++) {
                for (int i = h; i < n->history_length + 1; i++, h++) {
                        struct element *e = n->training_set->elements[item];

                        if (i > 0)
                                ffn_connect_duplicate_networks(
                                                n->unfolded_net,
                                                n->unfolded_net->stack[i - 1],
                                                n->unfolded_net->stack[i]);

                        copy_vector(n->unfolded_net->stack[i]->input->vector, 
                                        e->inputs[event]);
                        feed_forward(n->unfolded_net->stack[i],
                                        n->unfolded_net->stack[i]->input);
                        
                        if (e->targets[event])
                                copy_vector(n->unfolded_net->stack[i]->target,
                                                e->targets[event]);

                        event++;
                        if (event == e->num_events) {
                                event = 0;
                                item++;
                        }
                        if (item == n->training_set->num_elements)
                                item = 0;
                }

                struct network *ns = n->unfolded_net->stack[n->history_length];
                struct vector *error = n->error_measure(ns);
                backpropagate_error(ns, ns->output, error);
                dispose_vector(error);
                
                ffn_sum_deltas(n->unfolded_net);
                adjust_weights(n->unfolded_net->stack[0],
                                n->unfolded_net->stack[0]->output);

                ffn_cycle_stack(n->unfolded_net);

                h--;
        }

        for (int i = 1; i < n->history_length; i++)
                ffn_disconnect_duplicate_networks(
                                n->unfolded_net,
                                n->unfolded_net->stack[i - 1],
                                n->unfolded_net->stack[i]);
}

struct vector *ss_output_error(struct network *n)
{
        struct vector *error = create_vector(n->target->size);

        for (int i = 0; i < error->size; i++) {
                double act = n->output->vector->elements[i];
                double err = n->target->elements[i] - act;
                error->elements[i] = err
                        * n->out_act_fun_deriv(n->output->vector, i);
        }

        return error;
}

struct vector *ce_output_error(struct network *n)
{
        struct vector *error = create_vector(n->target->size);

        for (int i = 0; i < error->size; i++) {
                double act = n->output->vector->elements[i];
                double err = n->target->elements[i] - act;
                error->elements[i] = err;
        }

        return error;
}

/*
 * Backpropagate error. For a specified group G, backpropagate its error E
 * to all of its incoming projections. For each projection to a group G', 
 * this involves computation of:
 *
 * (1) The weight deltas for that projection. The delta for a connection
 *     between a unit U in G and U' in G' is the error at U' multiplied
 *     by the activation of U.
 *
 * (2) The error of that projection. Let W be the weight matrix between G
 *     and G'. The error for the projection between G and G' relative to a
 *     unit U' in G', is the sum of the weighted error at each of units U
 *     in G.
 *
 *     Note that this the 'absolute' error that needs to be multiplied by
 *     the derivative of the activation of U' to obtain the 'real' error.
 *     We will, however, want to do this later, as we will first sum the 
 *     error of all of the outgoing projections of G' to obtain the total
 *     error at G'. This is needed for backpropagation through time.
 *
 * Once the deltas and errors have been computed for each of the incoming
 * projections of G, the total error at a group G' is recursively propagated
 * to all of the incoming projections of G'. This is true for all 
 * projections, unless it is a recurrent one and the training algorithm at
 * hand is not truncated backpropgation through time.
 */

void backpropagate_error(struct network *n, struct group *g, 
                struct vector *error)
{
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *p = g->inc_projs->elements[i];
                zero_out_vector(p->error);
                comp_proj_deltas_and_error(n, p, error);
        }
        
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                if (n->learning_algorithm != train_bptt_truncated &&
                                g->inc_projs->elements[i]->recurrent)
                        continue;

                struct group *ng = g->inc_projs->elements[i]->to;

                struct vector *grp_error = group_error(n, ng);
                backpropagate_error(n, ng, grp_error);
                dispose_vector(grp_error);
        }
}

void comp_proj_deltas_and_error(struct network *n, struct projection *p,
                struct vector *error)
{
        for (int i = 0; i < p->to->vector->size; i++) {
                for (int j = 0; j < error->size; j++) {
                        p->error->elements[i] += p->weights->elements[i][j]
                                * error->elements[j];
                        p->deltas->elements[i][j] += p->to->vector->elements[i]
                                * error->elements[j];
                }
        }
}

struct vector *group_error(struct network *n, struct group *g)
{
        struct vector *error = create_vector(g->vector->size);

        for (int i = 0; i < error->size; i++) {
                for (int j = 0; j < g->out_projs->num_elements; j++) {
                        struct projection *p = g->out_projs->elements[j];
                        error->elements[i] += p->error->elements[i];
                }
                
                double act_deriv;
                if (g != n->input)
                        act_deriv = n->act_fun_deriv(g->vector, i);
                else 
                        act_deriv = g->vector->elements[i];

                error->elements[i] *= act_deriv;
        }

        return error;
}

/*
 * Adjust projection weights. For a specified group G, adjust the weights
 * of all of its incoming projections. The new weight of a connection within
 * this projection is this the old weight of this connection plus the
 * learning rate of the network multiplied by the connection's weight delta.
 * In addition, if a value larger than zero is specified for the momentum
 * parameter of the network, the previous weight delta of the connection
 * multiplied by the momentum parameter's value is added to the new weight.
 *
 * Once all of a projection's weights have been adjusted, its weight deltas
 * are set to zero for the next training iteration, but a copy is stored for
 * the application of momentum. After updating the projection weights of all
 * of the incoming projections of G, the process is recursively repeated for
 * all of the incoming projection of the groups that project to G. Hence,
 * if the first G is the output group of the network, the weights of all
 * projections from the output to the input group should be adjusted:
 *
 * ##########    ##########    ##########
 * # input  #<---# hidden #<---# output #
 * ##########    ##########    ##########
 */

void adjust_weights(struct network *n, struct group *g)
{
        for (int i = 0; i < g->inc_projs->num_elements; i++)
                adjust_projection_weights(n, g, g->inc_projs->elements[i]);
        
        for (int i = 0; i < g->inc_projs->num_elements; i++)
                adjust_weights(n, g->inc_projs->elements[i]->to);
}

void adjust_projection_weights(struct network *n, struct group *g,
                struct projection *p)
{
        for (int i = 0; i < p->to->vector->size; i++)
                for (int j = 0; j < g->vector->size; j++)
                        p->weights->elements[i][j] += 
                                n->learning_rate
                                * p->deltas->elements[i][j]
                                - n->weight_decay
                                * p->prev_deltas->elements[i][j]
                                + n->momentum
                                * p->prev_deltas->elements[i][j];
        
        copy_matrix(p->prev_deltas, p->deltas);
        zero_out_matrix(p->deltas);
}
