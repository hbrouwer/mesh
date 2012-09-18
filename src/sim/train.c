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

#include <math.h>

#define REPORT_AFTER_PERCENTAGE 0.01

void train_network(struct network *n)
{
        mprintf("starting training of network: [%s]", n->name);

        if (n->learning_algorithm == train_bptt)
                n->unfolded_net = ffn_init_unfolded_network(n);

        n->learning_algorithm(n);
}

/* test 'normal' network */
void test_network(struct network *n)
{
        mprintf("starting testing of network: [%s]", n->name);
        
        for (int i = 0; i < n->test_set->num_elements; i++) {
                struct element *e = n->test_set->elements[i];
                
                /* reset Elman groups */
                if (n->srn)
                        reset_elman_groups(n);

                rprintf("testing item: %d -- \"%s\"", i, e->name);
                for (int j = 0; j < e->num_events; j++) {
                        copy_vector(n->input->vector, e->inputs[j]);
                        feed_forward(n, n->input);

                        /* 
                         * If there is target vector for this event,
                         * print this vector as well as the network's
                         * output
                         */
                        if (e->targets[j] != NULL) {
                                copy_vector(n->target, e->targets[j]);
                                print_vector(n->target);
                                print_vector(n->output->vector);
                                rprintf("");
                        }
                }
        }

        /*
        print_weights(n);
        print_weight_stats(n);
        */

        double mse = mean_squared_error(n);
        pprintf("MSE: [%lf]", mse);
}

/* test an unfolded network */
void test_unfolded_network(struct network *n)
{
        mprintf("starting testing of network: [%s]", n->name);

        int h = 0;
        struct ffn_unfolded_network *un = n->unfolded_net;
        reset_recurrent_groups(un->stack[h]);

        for (int i = 0; i < n->training_set->num_elements; i++) {
                struct element *e = n->training_set->elements[i];

                rprintf("testing item: %d -- \"%s\"", i, e->name);
                for (int j = 0; j < e->num_events; j++) {
                        /* cycle network stack, if full */
                        if (h == un->stack_size) {
                                ffn_cycle_stack(un);
                                h--;
                        }

                        copy_vector(un->stack[h]->input->vector, e->inputs[j]);
                        if (e->targets[j])
                                copy_vector(un->stack[h]->target, e->targets[j]);
                        feed_forward(un->stack[h], un->stack[h]->input);

                        /* 
                         * If there is target vector for this event,
                         * print this vector as well as the network's
                         * output
                         */
                        if (e->targets[j] != NULL) {
                                print_vector(un->stack[h]->target);
                                print_vector(un->stack[h]->output->vector);
                                rprintf("");
                        }

                        h++;
                }

                // XXX: does this make sense??
                reset_recurrent_groups(un->stack[h - 1]);
        }

        /*
        print_weights(un->stack[0]);
        print_weight_stats(un->stack[0]);
        */

        double mse = mean_squared_error_un(n);
        pprintf("MSE: [%lf]", mse);
}

/*
 * Total network error (or mean squared error).
 */
double mean_squared_error(struct network *n)
{
        double mse = 0.0;
        for (int i = 0; i < n->training_set->num_elements; i++) {
                struct element *e = n->training_set->elements[i];
                
                /* reset Elman groups */
                if (n->srn)
                        reset_elman_groups(n);

                for (int j = 0; j < e->num_events; j++) {
                        copy_vector(n->input->vector, e->inputs[j]);
                        feed_forward(n, n->input);

                        /* 
                         * Compute MSE on basis of the current item's
                         * last event
                         */
                        if (j == e->num_events - 1) {
                                copy_vector(n->target, e->targets[j]);
                                mse += squared_error(n->output->vector, n->target);
                        }
                }
        }

        return mse / n->training_set->num_elements;
}

double mean_squared_error_un(struct network *n)
{
        double mse = 0.0;
        int h = 0;
        struct ffn_unfolded_network *un = n->unfolded_net;
        reset_recurrent_groups(un->stack[h]);

        for (int i = 0; i < n->training_set->num_elements; i++) {
                struct element *e = n->training_set->elements[i];

                for (int j = 0; j < e->num_events; j++) {
                        /* cycle network stack, if full */
                        if (h == un->stack_size) {
                                ffn_cycle_stack(un);
                                h--;
                        }

                        copy_vector(un->stack[h]->input->vector, e->inputs[j]);
                        feed_forward(un->stack[h], un->stack[h]->input);

                        /* 
                         * Compute MSE on basis of the current item's
                         * last event
                         */
                        if (j == e->num_events - 1) {
                                copy_vector(un->stack[h]->target, e->targets[j]);
                                mse += squared_error(un->stack[h]->output->vector,
                                                un->stack[h]->target);
                        }

                        h++;
                }

                // XXX: does this make sense??
                reset_recurrent_groups(un->stack[h - 1]);
        }

        return mse / n->training_set->num_elements;
}

void report_error(int epoch, double mse, struct network *n)
{
        int report_after = REPORT_AFTER_PERCENTAGE * n->max_epochs;

        if (epoch == 1 || epoch % report_after == 0) {
                double rms = sqrt(mse);
                pprintf("epoch: [%d] | MSE: [%lf] | RMS: [%lf]", epoch, mse, rms);
        }
}

/*
 * Feed activation forward. For a specified group G, the activation is
 * computed for each unit in all of the groups towards which G maintains an
 * outgoing projection. This process is recursively repeated for all groups
 * that have a projection coming in from G. If the first G is the input
 * group of the network, activation should be propagated forward through all
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
                if (g->out_projs->elements[i]->recurrent)
                        continue;

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

        for (int i = 0; i < g->out_projs->num_elements; i++) {
                if (!g->out_projs->elements[i]->recurrent)
                        feed_forward(n, g->out_projs->elements[i]->to);
        }
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
 * Backpropagation training
 */

void train_bp(struct network *n)
{
        for (int epoch = 1; epoch <= n->max_epochs; epoch++) {
                for (int i = 0; i < n->training_set->num_elements; i++) {
                        struct element *e = n->training_set->elements[i];

                        /* reset Elman groups */
                        if (n->srn)
                                reset_elman_groups(n);

                        for (int j = 0; j < e->num_events; j++) {
                                copy_vector(n->input->vector, e->inputs[j]);
                                feed_forward(n, n->input);

                                /* inject error if a target is specified */
                                if (e->targets[j]) {
                                        copy_vector(n->target, e->targets[j]);
                                        struct vector *error = n->error_measure(n);
                                        backpropagate_error(n, n->output, error);
                                        dispose_vector(error);
                                }
                        }
                }
                adjust_weights(n, n->output);

                /* compute and report MSE */
                double mse = mean_squared_error(n);
                report_error(epoch, mse, n);
                if (mse < n->mse_threshold)
                        break;

                /* scale LR and Momentum */
                scale_learning_rate(epoch,n);
                scale_momentum(epoch,n);
        }
}

/*
 * Backpropagation through time training
 */
void train_bptt(struct network *n)
{
        struct ffn_unfolded_network *un = n->unfolded_net;

        for (int epoch = 1; epoch <= n->max_epochs; epoch++) {
                int h = 0;
                reset_recurrent_groups(un->stack[h]);
                
                for (int i = 0; i < n->training_set->num_elements; i++) {
                        struct element *e = n->training_set->elements[i];

                        for (int j = 0; j < e->num_events; j++) {
                                /* cycle network stack if full */
                                if (h == un->stack_size) {
                                        ffn_cycle_stack(un);
                                        h--;
                                }

                                copy_vector(un->stack[h]->input->vector, e->inputs[j]);
                                if (e->targets[j])
                                        copy_vector(un->stack[h]->target, e->targets[j]);
                                feed_forward(un->stack[h], un->stack[h]->input);
                        
                                h++;
                        }

                        /* inject error and adjust weights */
                        if (h == un->stack_size) {
                                struct network *ns = un->stack[n->history_length];
                                struct vector *error = n->error_measure(ns);
                                backpropagate_error(ns, ns->output, error);
                                dispose_vector(error);
                                ffn_sum_deltas(un);
                                adjust_weights(un->stack[0], un->stack[0]->output);
                        }

                        // XXX: but what if h > num_events ?
                        reset_recurrent_groups(un->stack[h - 1]);
                }

                /* compute and report MSE */
                double mse = mean_squared_error_un(n);
                report_error(epoch, mse, n);
                if (mse < n->mse_threshold)
                        break;

                /* scale LR and Momentum */
                scale_learning_rate(epoch,n);
                scale_momentum(epoch,n);
        }
}


/* sum of squares error */
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

/* cross-entropy error */
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

void scale_learning_rate(int epoch, struct network *n)
{
        int scale_after = n->lr_scale_after * n->max_epochs;
        if (scale_after > 0 && epoch % scale_after == 0) {
                double lr = n->learning_rate;
                n->learning_rate = n->lr_scale_factor * n->learning_rate;
                mprintf("scaled learning rate: [%lf --> %lf]",
                                lr, n->learning_rate);
        }

}

void scale_momentum(int epoch, struct network *n)
{
        int scale_after = n->mn_scale_after * n->max_epochs;
        if (scale_after > 0 && epoch % scale_after == 0) {
                double mn = n->momentum;
                n->momentum = n->mn_scale_factor * n->momentum;
                mprintf("scaled momentum: [%lf --> %lf]",
                                mn, n->momentum);
        }
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
 * (2) The error of that projection. The error for the projection between G 
 *     and G' relative to a unit U' in G', is the sum of the weighted error 
 *     at each of units U in G.
 *
 *     Note that this the 'absolute' error that needs to be multiplied by
 *     the derivative of the activation of U' to obtain the 'real' error.
 *     We will, however, want to do this later, as we will first sum the 
 *     error of all of the outgoing projections of G' to obtain the total
 *     error at G'. This is needed for backpropagation through time.
 *
 * Once the deltas and errors have been computed for each of the incoming
 * projections of G, the total error at a group G' is recursively propagated
 * to all of the incoming projections of G'. 
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
 * learning rate of the network multiplied by the connection's weight delta,
 * minus the weight decay multiplied by the previous weight delta, and
 * plus the momentum times the previous weight delta.
 *
 * Once all of a projection's weights have been adjusted, its weight deltas
 * are set to zero for the next training iteration, but a copy is stored for
 * the application of weight decay and momentum. After updating the projection 
 * weights of all of the incoming projections of G, the process is recursively 
 * repeated for all of the incoming projection of the groups that project to 
 * G. Hence, if the first G is the output group of the network, the weights 
 * of all projections from the output to the input group should be adjusted:
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
