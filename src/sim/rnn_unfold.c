/*
 * rnn_unfold.c
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

#include "vector.h"
#include "rnn_unfold.h"

/* 
 * Unfolding of recurrent neural networks for backpropagation through time.
 * Assume a network with the following typology:
 * 
 * ###########
 * # output1 #
 * ###########
 *      |
 * ###########
 * # hidden1 # <-- recurrent group
 * ###########
 *      |
 * ###########
 * # input1  #
 * ###########
 *
 * where [hidden1] is a recurrent group. The aim is to unfold this network 
 * in time such that its states at different timesteps are connected through
 * recurrent connections:
 *
 *                                                       ...........   .
 *                                                            |        |
 *                                     ###########       ###########   |
 *                                     # output3 #   +--># hidden4 #<--+
 *                                     ###########   |   ###########
 *                                          |       [W]       |
 *                   ###########       ###########   |   ###########
 *                   # output2 #   +--># hidden3 #<--+   # input4  #
 *                   ###########   |   ###########       ###########
 *                         |      [W]       |
 * ###########       ###########   |   ###########
 * # output1 #   +--># hidden2 #<--+   # input3  #
 * ###########   |   ###########       ############
 *      |       [W]       |
 * ###########   |   ###########
 * # hidden1 #<--+   # input2  #
 * ###########       ###########
 *      |
 * ###########
 * # input1  #
 * ###########
 *
 * The weight matrix [W] is the same across the recurrent projections.
 */
struct rnn_unfolded_network *rnn_init_unfolded_network(struct network *n)
{
        struct rnn_unfolded_network *un;
        if (!(un = malloc(sizeof(struct rnn_unfolded_network))))
                goto error_out;
        memset(un, 0, sizeof(struct rnn_unfolded_network));

        un->recur_groups = rnn_recurrent_groups(n);

        int block_size = un->recur_groups->num_elements * sizeof(struct matrix *);
        if (!(un->recur_weights = malloc(block_size)))
                goto error_out;
        memset(un->recur_weights, 0, block_size);

        if (!(un->recur_prev_weight_deltas = malloc(block_size)))
                goto error_out;
        memset(un->recur_prev_weight_deltas, 0, block_size);

        if (!(un->recur_dyn_learning_pars = malloc(block_size)))
                goto error_out;
        memset(un->recur_dyn_learning_pars, 0, block_size);

        for (int i = 0; i < un->recur_groups->num_elements; i++) {
                struct group *g = un->recur_groups->elements[i];
                un->recur_weights[i] = create_matrix(
                                g->vector->size,
                                g->vector->size);
                randomize_matrix(un->recur_weights[i],
                                n->random_mu, n->random_sigma);

                un->recur_prev_weight_deltas[i] = create_matrix(
                                g->vector->size,
                                g->vector->size);

                un->recur_dyn_learning_pars[i] = create_matrix(
                                g->vector->size,
                                g->vector->size);
                fill_matrix_with_value(un->recur_dyn_learning_pars[i],
                                n->rp_init_update);
        }

        un->stack_size = n->history_length + 1;
        block_size = un->stack_size * sizeof(struct network *);
        if (!(un->stack = malloc(block_size)))
                goto error_out;
        memset(un->stack, 0, block_size);

        for (int i = 0; i < un->stack_size; i++) {
                un->stack[i] = rnn_duplicate_network(n);
                if (i == 0) {
                        rnn_attach_recurrent_groups(un, un->stack[0]);
                } else {
                        rnn_connect_duplicate_networks(un, un->stack[i - 1],
                                        un->stack[i]);
                }
        }

        return un;

error_out:
        perror("[rnn_init_unfolded_network()]");
        return NULL;
}

void rnn_dispose_unfolded_network(struct rnn_unfolded_network *un)
{
        for (int i = 1; i < un->stack_size; i++)
                rnn_disconnect_duplicate_networks(un, un->stack[i - 1], un->stack[i]);

        rnn_detach_recurrent_groups(un, un->stack[0]);

        for (int i = 0; i < un->recur_groups->num_elements; i++) {
                dispose_matrix(un->recur_weights[i]);
                dispose_matrix(un->recur_prev_weight_deltas[i]);
                dispose_matrix(un->recur_dyn_learning_pars[i]);
        }
        free(un->recur_weights);
        free(un->recur_prev_weight_deltas);

        dispose_group_array(un->recur_groups);
        
        for (int i = 0; i < un->stack_size; i++)
                rnn_dispose_duplicate_network(un->stack[i]);
        free(un->stack);

        free(un);
}

struct network *rnn_duplicate_network(struct network *n)
{
        struct network *dn;
        if (!(dn = malloc(sizeof(struct network))))
                goto error_out;
        memset(dn, 0, sizeof(struct network));
        memcpy(dn, n, sizeof(struct network));

        dn->groups = create_group_array(n->groups->max_elements);
        rnn_duplicate_groups(n, dn, n->input);

        return dn;

error_out:
        perror("[rnn_duplicate_network()])");
        return NULL;
}

void rnn_dispose_duplicate_network(struct network *dn)
{
        // rnn_dispose_duplicate_groups(dn->input);
        rnn_dispose_duplicate_groups(dn->output);
        dispose_group_array(dn->groups);

        free(dn);
}

struct group *rnn_duplicate_group(struct group *g)
{
        struct group *dg;

        if (!(dg = malloc(sizeof(struct group))))
                goto error_out;
        memset(dg, 0, sizeof(struct group));

        int block_size = (strlen(g->name) + 1) * sizeof(char);
        if (!(dg->name = malloc(block_size)))
                goto error_out;
        memset(dg->name, 0, block_size);
        strncpy(dg->name, g->name, strlen(g->name));

        dg->vector = create_vector(g->vector->size);
        dg->error = create_vector(g->vector->size);
        dg->act_fun = g->act_fun;
        dg->err_fun = g->err_fun;

        dg->inc_projs = create_projs_array(g->inc_projs->max_elements);
        dg->inc_projs->num_elements = g->inc_projs->num_elements;
        dg->out_projs = create_projs_array(g->out_projs->max_elements);
        dg->out_projs->num_elements = g->out_projs->num_elements;

        dg->bias = g->bias;
        dg->recurrent = g->recurrent;

        return dg;

error_out:
        perror("[rnn_duplicate_group()]");
        return NULL;
}

struct group *rnn_duplicate_groups(struct network *n, struct network *dn, 
                struct group *g)
{
        struct group *dg = rnn_duplicate_group(g);

        dn->groups->elements[dn->groups->num_elements++] = dg;
        if (dn->groups->num_elements == dn->groups->max_elements)
                increase_group_array_size(dn->groups);

        /* duplicate bias groups */
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct group *bg = g->inc_projs->elements[i]->to;

                if (!bg->bias)
                        continue;
                
                struct group *dbg = rnn_duplicate_group(bg);
                dn->groups->elements[dn->groups->num_elements++] = dbg;
                if (dn->groups->num_elements == dn->groups->max_elements)
                        increase_group_array_size(dn->groups);

                /*
                 * Note: weight matrices are shared among recurrent
                 *   projections.
                 */
                struct vector *error = create_vector(
                                bg->vector->size);
                struct matrix *gradients = create_matrix(
                                bg->vector->size,
                                g->vector->size);
                struct matrix *prev_gradients = create_matrix(
                                bg->vector->size,
                                g->vector->size);

                dg->inc_projs->elements[i] = rnn_duplicate_projection(
                                g->inc_projs->elements[i], error, gradients,
                                prev_gradients);
                dg->inc_projs->elements[i]->to = dbg;

                dbg->out_projs->elements[0] = rnn_duplicate_projection(
                                bg->out_projs->elements[0], error, gradients,
                                prev_gradients);
                dbg->out_projs->elements[0]->to = dg;
        }

        /* duplicate outgoing projections */
        for (int i = 0; i < g->out_projs->num_elements; i++) {
                struct group *g2 = g->out_projs->elements[i]->to;

                if (g->out_projs->elements[i]->recurrent)
                        continue;

                /*
                 * Note: weight matrices are shared among recurrent
                 *   projections.
                 */
                struct vector *error = create_vector(
                                g->vector->size);
                struct matrix *gradients = create_matrix(
                                g->vector->size,
                                g2->vector->size);
                struct matrix *prev_gradients = create_matrix(
                                g->vector->size,
                                g2->vector->size);

                dg->out_projs->elements[i] = rnn_duplicate_projection(
                                g->out_projs->elements[i], error, gradients, 
                                prev_gradients);
                struct group *rg = rnn_duplicate_groups(n, dn, g2);
                dg->out_projs->elements[i]->to = rg;

                for (int j = 0; j < g2->inc_projs->num_elements; j++) {
                        if (g2->inc_projs->elements[j]->to == g) {
                                rg->inc_projs->elements[j] = 
                                        rnn_duplicate_projection(
                                                        g2->inc_projs->elements[j], 
                                                        error, gradients, prev_gradients);
                                rg->inc_projs->elements[j]->to = dg;
                        }
                 }
        }

        if (n->input == g)
                dn->input = dg;
        if (n->output == g)
                dn->output = dg;

        return dg;
}

void rnn_dispose_duplicate_groups(struct group *dg)
{
        for (int i = 0; i < dg->inc_projs->num_elements; i++) {
                rnn_dispose_duplicate_groups(dg->inc_projs->elements[i]->to);
                rnn_dispose_duplicate_projection(dg->inc_projs->elements[i]);
        }
        dispose_projs_array(dg->inc_projs);

        for (int i = 0; i < dg->out_projs->num_elements; i++) {
                free(dg->out_projs->elements[i]);
        }
        dispose_projs_array(dg->out_projs);

        free(dg->name);
        dispose_vector(dg->vector);

        free(dg);
}

struct projection *rnn_duplicate_projection(
                struct projection *p,
                struct vector *error,
                struct matrix *gradients,
                struct matrix *prev_gradients)
{
        struct projection *dp;
        if (!(dp = malloc(sizeof(struct projection))))
                goto error_out;
        memset(dp, 0, sizeof(struct projection));

        dp->weights = p->weights; /* <-- shared weights */
        dp->error = error;
        dp->gradients = gradients;
        dp->prev_gradients = prev_gradients;
        dp->prev_weight_deltas = p->prev_weight_deltas;
        dp->dyn_learning_pars = p->dyn_learning_pars;

        return dp;

error_out:
        perror("[rnn_duplicate_projection()]");
        return NULL;
}

void rnn_dispose_duplicate_projection(struct projection *dp)
{
        dispose_vector(dp->error);
        dispose_matrix(dp->gradients);
        dispose_matrix(dp->prev_gradients);
        
        free(dp);
}

struct group_array *rnn_recurrent_groups(struct network *n)
{
        struct group_array *gs = create_group_array(MAX_GROUPS);
        
        rnn_collect_recurrent_groups(n->input, gs);
        
        return gs;
}

void rnn_collect_recurrent_groups(struct group *g, struct group_array *gs)
{
        if (g->recurrent) {
                gs->elements[gs->num_elements++] = g;
                if (gs->num_elements == gs->max_elements)
                        increase_group_array_size(gs);
        }

        for (int i = 0; i < g->out_projs->num_elements; i++)
                rnn_collect_recurrent_groups(g->out_projs->elements[i]->to, gs);
}

void rnn_attach_recurrent_groups(struct rnn_unfolded_network *un,
                struct network *n)
{
        for (int i = 0; i < un->recur_groups->num_elements; i++) {
                char *name = un->recur_groups->elements[i]->name;
                struct group *g1 = find_group_by_name(n, name);
                struct group *g2 = create_group(g1->name, g1->act_fun, g1->err_fun,
                                g1->vector->size, false, true);

                /*
                 * Note: weight matrices are shared among recurrent
                 *   projections.
                 */
                struct vector *error = create_vector(
                                g1->vector->size);
                struct matrix *gradients = create_matrix(
                                g1->vector->size,
                                g2->vector->size);
                struct matrix *prev_gradients = create_matrix(
                                g1->vector->size,
                                g2->vector->size);

                g2->out_projs->elements[g2->out_projs->num_elements++] =
                        create_projection(g1, un->recur_weights[i], error,
                                        gradients, prev_gradients, un->recur_prev_weight_deltas[i],
                                        un->recur_dyn_learning_pars[i], true);
                if (g2->out_projs->num_elements == g2->out_projs->max_elements)
                        increase_projs_array_size(g2->out_projs);

                g1->inc_projs->elements[g1->inc_projs->num_elements++] = 
                        create_projection(g2, un->recur_weights[i], error,
                                        gradients, prev_gradients, un->recur_prev_weight_deltas[i],
                                        un->recur_dyn_learning_pars[i], true);
                if (g1->inc_projs->num_elements == g1->inc_projs->max_elements)
                        increase_projs_array_size(g1->inc_projs);
        }
}

void rnn_detach_recurrent_groups(struct rnn_unfolded_network *un,
                struct network *n)
{
        for (int i = 0; i < un->recur_groups->num_elements; i++) {
                char *name = un->recur_groups->elements[i]->name;
                struct group *g1 = find_group_by_name(n, name);

                int z = g1->inc_projs->num_elements - 1;
                struct group *g2 = g1->inc_projs->elements[z]->to;

                g1->inc_projs->num_elements--;
                g2->out_projs->num_elements--;

                struct projection *p1 = 
                        g1->inc_projs->elements[g1->inc_projs->num_elements];
                struct projection *p2 =
                        g2->out_projs->elements[g2->out_projs->num_elements];
                rnn_dispose_duplicate_projection(p1);
                free(p2);

                g1->inc_projs->elements[g1->inc_projs->num_elements] = NULL;
                g2->out_projs->elements[g2->out_projs->num_elements] = NULL;

                rnn_dispose_duplicate_groups(g2);
        }
}

void rnn_connect_duplicate_networks(struct rnn_unfolded_network *un,
                struct network *n1, struct network *n2)
{
        for (int i = 0; i < un->recur_groups->num_elements; i++) {
                char *name = un->recur_groups->elements[i]->name;
                struct group *g1 = find_group_by_name(n1, name);
                struct group *g2 = find_group_by_name(n2, name);

                /*
                 * Note: weight matrices are shared among recurrent
                 *   projections.
                 */                
                struct vector *error = create_vector(
                                g1->vector->size);
                struct matrix *gradients = create_matrix(
                                g1->vector->size,
                                g2->vector->size);
                struct matrix *prev_gradients = create_matrix(
                                g1->vector->size,
                                g2->vector->size);

                g1->out_projs->elements[g1->out_projs->num_elements++] =
                        create_projection(g2, un->recur_weights[i], error,
                                        gradients, prev_gradients, un->recur_prev_weight_deltas[i],
                                        un->recur_dyn_learning_pars[i], true);
                if (g1->out_projs->num_elements == g1->out_projs->max_elements)
                        increase_projs_array_size(g1->out_projs);

                g2->inc_projs->elements[g2->inc_projs->num_elements++] = 
                        create_projection(g1, un->recur_weights[i], error,
                                        gradients, prev_gradients, un->recur_prev_weight_deltas[i],
                                        un->recur_dyn_learning_pars[i], true);
                if (g2->inc_projs->num_elements == g2->inc_projs->max_elements)
                        increase_projs_array_size(g2->inc_projs);
        }
}

void rnn_disconnect_duplicate_networks(struct rnn_unfolded_network *un,
                struct network *n1, struct network *n2)
{
        for (int i = 0; i < un->recur_groups->num_elements; i++) {
                 char *name = un->recur_groups->elements[i]->name;
                 struct group *g1 = find_group_by_name(n1, name);
                 struct group *g2 = find_group_by_name(n2, name);

                 g1->out_projs->num_elements--;
                 g2->inc_projs->num_elements--;

                 struct projection *p1 = 
                         g1->out_projs->elements[g1->out_projs->num_elements];
                 struct projection *p2 =
                         g2->inc_projs->elements[g2->inc_projs->num_elements];
                 rnn_dispose_duplicate_projection(p1);
                 free(p2);

                 g1->out_projs->elements[g1->out_projs->num_elements] = NULL;
                 g2->inc_projs->elements[g2->inc_projs->num_elements] = NULL;
        }
}

void rnn_sum_gradients(struct rnn_unfolded_network *un)
{
        for (int i = 1; i < un->stack_size; i++)
                rnn_add_gradients(un->stack[0]->output, un->stack[i]->output);
}

void rnn_add_gradients(struct group *g1, struct group *g2)
{
        for (int i = 0; i < g1->inc_projs->num_elements; i++) {
                struct projection *p1 = g1->inc_projs->elements[i];
                struct projection *p2 = g2->inc_projs->elements[i];
                
                for (int r = 0; r < p1->gradients->rows; r++)
                        for (int c = 0; c < p1->gradients->cols; c++)
                                p1->gradients->elements[r][c] +=
                                        p2->gradients->elements[r][c];

                copy_matrix(p2->gradients, p2->prev_gradients);
                zero_out_matrix(p2->gradients);

                if (!p1->recurrent)
                        rnn_add_gradients(p1->to, p2->to);
        }
}

/*
 * Cycle the network stack. Assume the following unfolded network:
 *
 *                                     ...........   .
 *                                          |        |
 *                   ###########       ###########   |
 *                   # output2 #   +--># hidden3 #<--+
 *                   ###########   |   ###########
 *                        |        |        |
 *                   ###########   |   ###########
 *               +--># hidden2 #<--+   # input3  #
 *               |   ###########       ###########
 *               |        |
 * ###########   |   ###########
 * # hidden1 #<--+   # input2  #
 * ###########       ###########
 *
 *                    stack[0]          stack[1]   ....   stack[n]
 *
 * In brief, we want to completely isolate stack[0] and move it into
 * stack[n]. We accomplish this by conducting the following steps:
 *
 * (1) Store a reference to [hidden1]. Next, disconnect [hidden1] from 
 *     [hidden2], dispose the corresponding projection error vector, and 
 *     delta matrices, and remove [hidden1] from the incoming projection 
 *     array of [hidden2].
 *
 * (2) Disconnect [hidden2] from [hidden3], preserve the projection's
 *     error vector, delta matrices, and remove [hidden3] from the outgoing
 *     projections of [hidden2].
 *
 * (3) Copy the activation vector of [hidden2] into [hidden1].
 *
 * (4) Set [hidden1] as the recurrent group of [hidden3], reusing the
 *     error vector, and delta matrices that were used for the previous
 *     projection between [hidden2] and [hidden3].
 *
 * (5) Store a reference to the network state at stack[0]. Next, shift
 *     stack[1] into stack[0], stack[2] into stack[1], until stack[n] has 
 *     been shifted into stack[n - 1]. Finally, set the stack[n] to refer
 *     to the previous stack[0].
 *
 * (6) Connect the recurrent groups of stack[n] to those of stack[n-1].
 *
 * Note: The above procedure extends to multiple recurrent groups per 
 *   network.
 */
void rnn_cycle_stack(struct rnn_unfolded_network *un)
{
        for (int i = 0; i < un->recur_groups->num_elements; i++) {
                char *name = un->recur_groups->elements[i]->name;

                struct group *g1 = find_group_by_name(un->stack[0], name);
                struct group *g2 = find_group_by_name(un->stack[1], name);

                /* step 1 */
                int j = g1->inc_projs->num_elements - 1;
                struct projection *p = g1->inc_projs->elements[j];
                struct group *g = g1->inc_projs->elements[j]->to;
                rnn_dispose_duplicate_projection(p);
                g1->inc_projs->elements[j] = NULL;
                g1->inc_projs->num_elements--;

                /* step 2 */
                j = g1->out_projs->num_elements - 1;
                free(g1->out_projs->elements[j]);
                g1->out_projs->elements[j] = NULL;
                g1->out_projs->num_elements--;

                /* step 3 */
                copy_vector(g->vector, g1->vector);

                /* step 4 */
                j = g2->inc_projs->num_elements - 1;
                g2->inc_projs->elements[j]->to = g;
                p = g2->inc_projs->elements[j];
                j = g->out_projs->num_elements - 1;
                g->out_projs->elements[j]->to = g2;
                g->out_projs->elements[j]->error = p->error;
                g->out_projs->elements[j]->gradients = p->gradients;
                g->out_projs->elements[j]->prev_gradients = p->prev_gradients;
        }
        
        /* step 5 */
        struct network *n = un->stack[0];
        for (int i = 0; i < un->stack_size - 1; i++) {
                un->stack[i] = un->stack[i + 1];
        }
        un->stack[un->stack_size - 1] = n;

        /* step 6 */
        rnn_connect_duplicate_networks(
                        un, 
                        un->stack[un->stack_size - 2],
                        un->stack[un->stack_size - 1]);
}
