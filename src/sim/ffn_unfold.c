/*
 * ffn_unfold.c
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
#include "train.h"
#include "vector.h"

/*
 * ############################## WARNING #################################
 * ## Unfolding is only guaranteed to work properly for feed forward     ##
 * ## networks. Behavior is ill-defined when used on other topologies.   ##
 * ########################################################################
 */

/* 
 * Unfolding of feed forward networks for backpropagation through time.
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
struct ffn_unfolded_network *ffn_init_unfolded_network(struct network *n)
{
        struct ffn_unfolded_network *un;
        if (!(un = malloc(sizeof(struct ffn_unfolded_network))))
                goto error_out;
        memset(un, 0, sizeof(struct ffn_unfolded_network));

        un->recur_groups = ffn_recurrent_groups(n);

        int block_size = un->recur_groups->num_elements
                * sizeof(struct matrix *);
        if (!(un->recur_weights = malloc(block_size)))
                goto error_out;
        memset(un->recur_weights, 0, block_size);

        for (int i = 0; i < un->recur_groups->num_elements; i++) {
                struct group *g = un->recur_groups->elements[i];
                un->recur_weights[i] = create_matrix(
                                g->vector->size,
                                g->vector->size);
                randomize_matrix(un->recur_weights[i],
                                n->random_mu, n->random_sigma);
        }

        if (n->learning_algorithm == train_bptt_epochwise)
                un->stack_size = n->epoch_length;
        else
                un->stack_size = n->history_length + 1;
        block_size = un->stack_size * sizeof(struct network *);
        if (!(un->stack = malloc(block_size)))
                goto error_out;
        memset(un->stack, 0, block_size);

        for (int i = 0; i < un->stack_size; i++)
                un->stack[i] = ffn_duplicate_network(n);
        ffn_attach_recurrent_groups(un, un->stack[0]);

        return un;

error_out:
        perror("[ffn_init_unfolded_network()]");
        return NULL;
}

void ffn_dispose_unfolded_network(struct ffn_unfolded_network *un)
{
        ffn_detach_recurrent_groups(un, un->stack[0]);

        for (int i = 0; i < un->recur_groups->num_elements; i++)
                dispose_matrix(un->recur_weights[i]);
        free(un->recur_weights);

        dispose_group_array(un->recur_groups);
        
        for (int i = 0; i < un->stack_size; i++)
                ffn_dispose_duplicate_network(un->stack[i]);
        free(un->stack);

        free(un);
}

struct network *ffn_duplicate_network(struct network *n)
{
        struct network *dn;
        if (!(dn = malloc(sizeof(struct network))))
                goto error_out;
        memset(dn, 0, sizeof(struct network));
        memcpy(dn, n, sizeof(struct network));

        dn->groups = create_group_array(n->groups->max_elements);
        ffn_duplicate_groups(n, dn, n->input);

        return dn;

error_out:
        perror("[ffn_duplicate_network()])");
        return NULL;
}

void ffn_dispose_duplicate_network(struct network *dn)
{
        // ffn_dispose_duplicate_groups(dn->input);
        ffn_dispose_duplicate_groups(dn->output);
        dispose_group_array(dn->groups);

        free(dn);
}

struct group *ffn_duplicate_group(struct group *g)
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

        dg->inc_projs = create_projs_array(g->inc_projs->max_elements);
        dg->inc_projs->num_elements = g->inc_projs->num_elements;
        dg->out_projs = create_projs_array(g->out_projs->max_elements);
        dg->out_projs->num_elements = g->out_projs->num_elements;

        dg->bias = g->bias;
        dg->recurrent = g->recurrent;

        return dg;

error_out:
        perror("[ffn_duplicate_group()]");
        return NULL;
}

struct group *ffn_duplicate_groups(struct network *n, struct network *dn, 
                struct group *g)
{
        struct group *dg = ffn_duplicate_group(g);

        dn->groups->elements[dn->groups->num_elements++] = dg;
        if (dn->groups->num_elements == dn->groups->max_elements)
                increase_group_array_size(dn->groups);

        /* duplicate bias groups */
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct group *bg = g->inc_projs->elements[i]->to;

                if (!bg->bias)
                        continue;
                
                struct group *dbg = ffn_duplicate_group(bg);
                dn->groups->elements[dn->groups->num_elements++] = dbg;
                if (dn->groups->num_elements == dn->groups->max_elements)
                        increase_group_array_size(dn->groups);

                /*
                 * Note: weight matrices are shared among recurrent
                 *   projections.
                 */
                struct vector *error = create_vector(
                                bg->vector->size);
                struct matrix *deltas = create_matrix(
                                bg->vector->size,
                                g->vector->size);
                struct matrix *prev_deltas = create_matrix(
                                bg->vector->size,
                                        g->vector->size);

                dg->inc_projs->elements[i] = ffn_duplicate_projection(
                                g->inc_projs->elements[i], error, deltas, prev_deltas);
                dg->inc_projs->elements[i]->to = dbg;

                dbg->out_projs->elements[0] = ffn_duplicate_projection(
                                bg->out_projs->elements[0], error, deltas, prev_deltas);
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
                struct matrix *deltas = create_matrix(
                                g->vector->size,
                                g2->vector->size);
                struct matrix *prev_deltas = create_matrix(
                                g->vector->size,
                                g2->vector->size);

                dg->out_projs->elements[i] = ffn_duplicate_projection(
                                g->out_projs->elements[i], error, deltas, 
                                prev_deltas);
                struct group *rg = ffn_duplicate_groups(n, dn, g2);
                dg->out_projs->elements[i]->to = rg;

                for (int j = 0; j < g2->inc_projs->num_elements; j++) {
                        if (g2->inc_projs->elements[j]->to == g) {
                                rg->inc_projs->elements[j] = 
                                        ffn_duplicate_projection(
                                                        g2->inc_projs->elements[j], 
                                                        error, deltas, prev_deltas);
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

void ffn_dispose_duplicate_groups(struct group *dg)
{
        for (int i = 0; i < dg->inc_projs->num_elements; i++) {
                ffn_dispose_duplicate_groups(dg->inc_projs->elements[i]->to);
                ffn_dispose_duplicate_projection(dg->inc_projs->elements[i]);
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

/*
void ffn_dispose_duplicate_groups(struct group *dg)
{
        for (int i = 0; i < dg->out_projs->num_elements; i++) {
                ffn_dispose_duplicate_groups(dg->out_projs->elements[i]->to);
                ffn_dispose_duplicate_projection(dg->out_projs->elements[i]);
        }
        dispose_projs_array(dg->out_projs);

        for (int i = 0; i < dg->inc_projs->num_elements; i++)
                free(dg->inc_projs->elements[i]);
        dispose_projs_array(dg->inc_projs);

        free(dg->name);
        dispose_vector(dg->vector);

        free(dg);
}
*/

struct projection *ffn_duplicate_projection(
                struct projection *p,
                struct vector *error,
                struct matrix *deltas,
                struct matrix *prev_deltas)
{
        struct projection *dp;
        if (!(dp = malloc(sizeof(struct projection))))
                goto error_out;
        memset(dp, 0, sizeof(struct projection));

        dp->weights = p->weights; /* <-- shared weights */
        dp->error = error;
        dp->deltas = deltas;
        dp->prev_deltas = prev_deltas;

        return dp;

error_out:
        perror("[ffn_duplicate_projection()]");
}

void ffn_dispose_duplicate_projection(struct projection *dp)
{
        dispose_vector(dp->error);
        dispose_matrix(dp->deltas);
        dispose_matrix(dp->prev_deltas);
        
        free(dp);
}

struct group_array *ffn_recurrent_groups(struct network *n)
{
        struct group_array *gs = create_group_array(MAX_GROUPS);
        
        ffn_collect_recurrent_groups(n->input, gs);
        
        return gs;
}

void ffn_collect_recurrent_groups(struct group *g, struct group_array *gs)
{
        if (g->recurrent) {
                gs->elements[gs->num_elements++] = g;
                if (gs->num_elements == gs->max_elements)
                        increase_group_array_size(gs);
        }

        for (int i = 0; i < g->out_projs->num_elements; i++)
                ffn_collect_recurrent_groups(g->out_projs->elements[i]->to, gs);
}

void ffn_attach_recurrent_groups(struct ffn_unfolded_network *un,
                struct network *n)
{
        for (int i = 0; i < un->recur_groups->num_elements; i++) {
                char *name = un->recur_groups->elements[i]->name;
                struct group *g1 = find_group_by_name(n, name);
                struct group *g2 = create_group(g1->name, g1->vector->size, false, true);

                /*
                 * Note: weight matrices are shared among recurrent
                 *   projections.
                 */
                struct vector *error = create_vector(
                                g1->vector->size);
                struct matrix *deltas = create_matrix(
                                g1->vector->size,
                                g2->vector->size);
                struct matrix *prev_deltas = create_matrix(
                                g1->vector->size,
                                g2->vector->size);

                g2->out_projs->elements[g2->out_projs->num_elements++] =
                        create_projection(g1, un->recur_weights[i], error,
                                        deltas, prev_deltas, true);
                if (g2->out_projs->num_elements == g2->out_projs->max_elements)
                        increase_projs_array_size(g2->out_projs);

                g1->inc_projs->elements[g1->inc_projs->num_elements++] = 
                        create_projection(g2, un->recur_weights[i], error,
                                        deltas, prev_deltas, true);
                if (g1->inc_projs->num_elements == g1->inc_projs->max_elements)
                        increase_projs_array_size(g1->inc_projs);
        }
}

void ffn_detach_recurrent_groups(struct ffn_unfolded_network *un,
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
                ffn_dispose_duplicate_projection(p1);
                free(p2);

                g1->inc_projs->elements[g1->inc_projs->num_elements] = NULL;
                g2->out_projs->elements[g2->out_projs->num_elements] = NULL;

                ffn_dispose_duplicate_groups(g2);
        }
}

void ffn_connect_duplicate_networks(struct ffn_unfolded_network *un,
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
                struct matrix *deltas = create_matrix(
                                g1->vector->size,
                                g2->vector->size);
                struct matrix *prev_deltas = create_matrix(
                                g1->vector->size,
                                g2->vector->size);

                g1->out_projs->elements[g1->out_projs->num_elements++] =
                        create_projection(g2, un->recur_weights[i], error,
                                        deltas, prev_deltas, true);
                if (g1->out_projs->num_elements == g1->out_projs->max_elements)
                        increase_projs_array_size(g1->out_projs);

                g2->inc_projs->elements[g2->inc_projs->num_elements++] = 
                        create_projection(g1, un->recur_weights[i], error,
                                        deltas, prev_deltas, true);
                if (g2->inc_projs->num_elements == g2->inc_projs->max_elements)
                        increase_projs_array_size(g2->inc_projs);
        }
}

void ffn_disconnect_duplicate_networks(struct ffn_unfolded_network *un,
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
                 ffn_dispose_duplicate_projection(p1);
                 free(p2);

                 g1->out_projs->elements[g1->out_projs->num_elements] = NULL;
                 g2->inc_projs->elements[g2->inc_projs->num_elements] = NULL;
        }
}

void ffn_sum_deltas(struct ffn_unfolded_network *un)
{
        for (int i = 1; i < un->stack_size; i++)
                ffn_add_deltas(un->stack[0]->output, un->stack[i]->output);
}

void ffn_add_deltas(struct group *g1, struct group *g2)
{
        for (int i = 0; i < g1->inc_projs->num_elements; i++) {
                struct projection *p1 = g1->inc_projs->elements[i];
                struct projection *p2 = g2->inc_projs->elements[i];
                
                for (int r = 0; r < p1->deltas->rows; r++)
                        for (int c = 0; c < p1->deltas->cols; c++)
                                p1->deltas->elements[r][c] +=
                                        p2->deltas->elements[r][c];

                copy_matrix(p2->deltas, p2->prev_deltas);
                zero_out_matrix(p2->deltas);

                if (!p1->recurrent)
                        ffn_add_deltas(p1->to, p2->to);
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
 * Note: The above procedure extends to multiple recurrent groups per 
 *   network.
 */
void ffn_cycle_stack(struct ffn_unfolded_network *un)
{
        for (int i = 0; i < un->recur_groups->num_elements; i++) {
                char *name = un->recur_groups->elements[i]->name;

                struct group *g1 = find_group_by_name(un->stack[0], name);
                struct group *g2 = find_group_by_name(un->stack[1], name);

                /* step 1 */
                int j = g1->inc_projs->num_elements - 1;
                struct projection *p = g1->inc_projs->elements[j];
                struct group *g = g1->inc_projs->elements[j]->to;
                ffn_dispose_duplicate_projection(p);
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
                g->out_projs->elements[j]->deltas = p->deltas;
                g->out_projs->elements[j]->prev_deltas = p->prev_deltas;
        }
        
        /* step 5 */
        struct network *n = un->stack[0];
        for (int i = 0; i < un->stack_size - 1; i++) {
                un->stack[i] = un->stack[i + 1];
        }
        un->stack[un->stack_size - 1] = n;
}
