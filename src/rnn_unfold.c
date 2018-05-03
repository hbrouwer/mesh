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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bp.h"
#include "rnn_unfold.h"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
This implements the unfolding of recurrent neural networks (RNNs) for
backpropagation through time (BPTT; Rumelhart, Hinton, & Williams, 1986),
such that they effectively become feed forward networks (FFNs) that can be
trained with standard backpropagation (BP). 

Assume an RNN with the following topology:

        +---------+
        | output1 |
        +---------+
             |
        +---------+
        | hidden1 | <-- recurrent group
        +---------+
             |
        +---------+
        | input1  |
        +---------+

where _hidden1_ is a recurrent group. The aim is to unfold this network in
time such that its states at different timesteps are connected through
recurrent projections:

        ...........
             |
        +---------+       +---------+
        | hidden4 |<--+   | output3 |
        +---------+   |   +---------+
             |        |        |
        +---------+   |   +---------+       +---------+
        | input4  |   +-->| hidden3 |<--+   | output2 |
        +---------+       +---------+   |   +---------+
                               |        |        |
                          +---------+   |   +---------+       +---------+
                          | input3  |   +-->| hidden2 |<--+   | output1 |
                          +---------+       +---------+   |   +---------+
                                                 |        |        |
                                            +---------+   |   +---------+
                                            | input2  |   +-->| hidden1 |
                                            +---------+       +---------+
                                                                   |
                                                              +---------+
                                                              | input1  |
                                                              +---------+

Note: Weight matrices, previous weight delta matrices, and dynamic learning
parameter matrices are shared among recurrent projections.

References

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning internal
        representations by error propagation. In: D. E. Rumelhart & J. L.
        McClelland (Eds.), Parallel distributed processing: Explorations in
        the microstructure of cognition, Volume 1: Foundations, pp. 318-362,
        Cambridge, MA: MIT Press.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

struct rnn_unfolded_network *rnn_init_unfolded_network(struct network *n)
{
        struct rnn_unfolded_network *un;
        if (!(un = malloc(sizeof(struct rnn_unfolded_network))))
                goto error_out;
        memset(un, 0, sizeof(struct rnn_unfolded_network));

        /* 
         * Obtain an array of recurrent groups, and allocate arrays for
         * their shared weight matrices, previous weight delta matrices, and
         * dynamic learning parameter matrices.
         */
        un->rcr_groups = rnn_recurrent_groups(n);
        size_t block_size = un->rcr_groups->num_elements
                * sizeof(struct matrix *);
        /* weights */
        if (!(un->rcr_weights = malloc(block_size)))
                goto error_out;
        memset(un->rcr_weights, 0, block_size);
        /* previous weight deltas */
        if (!(un->rcr_prev_deltas = malloc(block_size)))
                goto error_out;
        memset(un->rcr_prev_deltas, 0, block_size);
        /* dynamic learning parameters */
        if (!(un->rcr_dynamic_params = malloc(block_size)))
                goto error_out;
        memset(un->rcr_dynamic_params, 0, block_size);

        /*
         * Fill the constructed arrays with the required matrices, and
         * initialize these appropriately.
         */
        for (uint32_t i = 0; i < un->rcr_groups->num_elements; i++) {
                struct group *g = un->rcr_groups->elements[i];
                uint32_t vz = g->vector->size;

                /* construct a random weight matrix */
                un->rcr_weights[i] = create_matrix(vz, vz);
                n->random_algorithm(un->rcr_weights[i], n);

                /* construct an empty matrix for previous weight deltas */
                un->rcr_prev_deltas[i] = create_matrix(vz, vz);

                /* 
                 * Construct a matrix for dynamic learning parameters. If
                 * the Delta-bar-Delta update algorithm is used, we set the
                 * values of this matrix to the network's learning rate. If
                 * Rprop is used, by constrast, we set its values to the
                 * initial Rprop update value.
                 */
                un->rcr_dynamic_params[i] = create_matrix(vz, vz);
                if (n->update_algorithm == bp_update_dbd)
                        fill_matrix_with_value(
                                un->rcr_dynamic_params[i],
                                n->pars->learning_rate);
                if (n->update_algorithm == bp_update_rprop)
                        fill_matrix_with_value(
                                un->rcr_dynamic_params[i],
                                n->pars->rp_init_update);
        }

        /* 
         * Allocate a stack for duplicate networks. The size of this stack
         * should be equal to the desired number of back ticks plus one
         * (current timestep plus history).
         */
        un->stack_size = n->pars->back_ticks + 1;
        block_size = un->stack_size * sizeof(struct network *);
        if (!(un->stack = malloc(block_size)))
                goto error_out;
        memset(un->stack, 0, block_size);

        /*
         * Fill the stack with duplicate networks. The first network on the
         * stack is attached a "terminal" recurrent group. All other
         * networks are connected to the network that precedes them on the
         * stack.
         */
        for (uint32_t i = 0; i < un->stack_size; i++) {
                un->stack[i] = rnn_duplicate_network(n);
                if (i == 0)
                        rnn_attach_recurrent_groups(un,
                                un->stack[i]);
                else
                        rnn_connect_duplicate_networks(un,
                                un->stack[i - 1],
                                un->stack[i]);
        }
        
        return un;

error_out:
        perror("[rnn_init_unfolded_network()]");
        return NULL;
}

void rnn_free_unfolded_network(struct rnn_unfolded_network *un)
{
        /* detach terminal recurrent groups */
        rnn_detach_recurrent_groups(un, un->stack[0]);

        /* disconnect each network from the one preceding it */
        for (uint32_t i = 1; i < un->stack_size; i++)
                rnn_disconnect_duplicate_networks(un,
                        un->stack[i - 1],
                        un->stack[i]);

        /* free recurrent matrices, and their corresponding arrays */
        for (uint32_t i = 0; i < un->rcr_groups->num_elements; i++) {
                free_matrix(un->rcr_weights[i]);
                free_matrix(un->rcr_prev_deltas[i]);
                free_matrix(un->rcr_dynamic_params[i]);
        }
        free(un->rcr_weights);
        free(un->rcr_prev_deltas);
        free(un->rcr_dynamic_params);

        /* free the array of recurrent groups */
        free_array(un->rcr_groups);

        /* free networks and stack */
        for (uint32_t i = 0; i < un->stack_size; i++)
                rnn_free_duplicate_network(un->stack[i]);
        free(un->stack);

        /* free unfolded network */
        free(un);
}

struct network *rnn_duplicate_network(struct network *n)
{
        struct network *dn;
        if (!(dn = malloc(sizeof(struct network))))
                goto error_out;
        memset(dn, 0, sizeof(struct network));
        memcpy(dn, n, sizeof(struct network));

        /* duplicate network groups */
        dn->groups = create_array(atype_groups);
        rnn_duplicate_groups(n, dn, n->input);

        return dn;

error_out:
        perror("[rnn_duplicate_network()])");
        return NULL;
}

void rnn_free_duplicate_network(struct network *dn)
{
        rnn_free_duplicate_groups(dn->output);
        free_array(dn->groups);
        free(dn);
}

struct group *rnn_duplicate_group(struct group *g)
{
        struct group *dg;
        if (!(dg = malloc(sizeof(struct group))))
                goto error_out;
        memset(dg, 0, sizeof(struct group));
        
        /* name */
        dg->name = g->name;

        /* unit and error vectors */
        dg->vector = create_vector(g->vector->size);
        dg->error  = create_vector(g->vector->size);

        /* activation function */
        dg->act_fun = g->act_fun;
        dg->err_fun = g->err_fun;
        
        /* incoming and outgoing projections */
        dg->inc_projs = create_array(atype_projs);
        dg->inc_projs->num_elements = g->inc_projs->num_elements;
        dg->out_projs = create_array(atype_projs);
        dg->out_projs->num_elements = g->out_projs->num_elements;

        /* flags and parameters */
        dg->flags = g->flags;
        dg->pars  = g->pars;

        return dg;

error_out:
        perror("[rnn_duplicate_group()]");
        return NULL;
}

struct group *rnn_duplicate_groups(struct network *n, struct network *dn, 
        struct group *g)
{
        /* duplicate groups */
        struct group *dg = rnn_duplicate_group(g);
        add_group(dn, dg);

        /* set input and output groups */
        if (n->input == g)  dn->input  = dg;
        if (n->output == g) dn->output = dg;

        /* 
         * If the current group has a bias group, duplicate it.
         */
        for (uint32_t i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *ip = g->inc_projs->elements[i];
                struct group *bg = ip->to;
                
                /* skip non-bias groups */
                if (!bg->flags->bias) continue;

                /* duplicate bias group */
                struct group *dbg = rnn_duplicate_group(bg);
                add_group(dn, dbg);

                /*
                 * Duplicate the projection between the bias group and its
                 * corresponding group. We only need a unique gradient and
                 * previous gradient matrix for this projection.
                 */
                struct matrix *gradients =
                        create_matrix(1, g->vector->size);
                struct matrix *prev_gradients =
                        create_matrix(1, g->vector->size);
                dg->inc_projs->elements[i] =
                        rnn_duplicate_projection(dbg, ip,
                                gradients, prev_gradients);
                dbg->out_projs->elements[0] = 
                        rnn_duplicate_projection(dg,
                                bg->out_projs->elements[0],
                                gradients, prev_gradients);
        }

        /* 
         * Recursively duplicate the current group's outgoing projections.
         */
        for (uint32_t i = 0; i < g->out_projs->num_elements; i++) {
                struct projection *op = g->out_projs->elements[i];
                struct group *tg      = op->to;

                /* recursively duplicate groups */
                struct group *rg = rnn_duplicate_groups(n, dn, tg);

                /*
                 * Duplicate the projection between the current group and
                 * the group to which it projects. We only need a unique
                 * gradient and previous gradient matrix for this
                 * projection.
                 */
                struct matrix *gradients =
                        create_matrix(g->vector->size, tg->vector->size);
                struct matrix *prev_gradients =
                        create_matrix(g->vector->size, tg->vector->size);
                dg->out_projs->elements[i] = 
                        rnn_duplicate_projection(rg, op,
                                gradients, prev_gradients);
                
                /*
                 * Add the required incoming projection. We want to add this
                 * projection in the same array location as the original
                 * projection.
                 */
                for (uint32_t j = 0; j < tg->inc_projs->num_elements; j++) {
                        struct projection *ip = tg->inc_projs->elements[j];
                        if (ip->to != g)
                                continue;
                        rg->inc_projs->elements[j] =
                                rnn_duplicate_projection(dg, ip,
                                        gradients, prev_gradients);
                }
        }

        return dg;
}

void rnn_free_duplicate_groups(struct group *dg)
{
        /* recursively free incoming projections */
        for (uint32_t i = 0; i < dg->inc_projs->num_elements; i++) {
                struct projection *ip = dg->inc_projs->elements[i];
                rnn_free_duplicate_groups(ip->to);
                rnn_free_duplicate_projection(ip);
        }
        free_array(dg->inc_projs);

        /* free outgoing projections */
        for (uint32_t i = 0; i < dg->out_projs->num_elements; i++)
                free(dg->out_projs->elements[i]);
        free_array(dg->out_projs);

        /* free context groups (if allocated) */
        if (dg->ctx_groups)
                free_array(dg->ctx_groups);

        free_vector(dg->vector);
        free_vector(dg->error);
        free(dg);
}

struct projection *rnn_duplicate_projection(
        struct group *to,
        struct projection *p,
        struct matrix *gradients,
        struct matrix *prev_gradients)
{
        struct projection *dp;
        if (!(dp = malloc(sizeof(struct projection))))
                goto error_out;
        memset(dp, 0, sizeof(struct projection));

        dp->to             = to;
        dp->weights        = p->weights;        /* <-- shared */
        dp->gradients      = gradients;
        dp->prev_gradients = prev_gradients;
        dp->prev_deltas    = p->prev_deltas;    /* <-- shared */
        dp->dynamic_params = p->dynamic_params; /* <-- shared */
        dp->flags          = p->flags;          /* <-- shared */

        return dp;

error_out:
        perror("[rnn_duplicate_projection()]");
        return NULL;
}

void rnn_free_duplicate_projection(struct projection *dp)
{
        free_matrix(dp->gradients);
        free_matrix(dp->prev_gradients);
        free(dp);
}

struct array *rnn_recurrent_groups(struct network *n)
{
        struct array *gs = create_array(atype_groups);
        rnn_collect_recurrent_groups(n->input, gs);
        return gs;
}

void rnn_collect_recurrent_groups(struct group *g, struct array *gs)
{
        if (g->flags->recurrent)
                add_to_array(gs, g);
        for (uint32_t i = 0; i < g->out_projs->num_elements; i++)
                rnn_collect_recurrent_groups(((struct projection *)
                        g->out_projs->elements[i])->to, gs);
}

void rnn_attach_recurrent_groups(struct rnn_unfolded_network *un,
        struct network *n)
{
        for (uint32_t i = 0; i < un->rcr_groups->num_elements; i++) {
                /* recurrent group */
                struct group *rg = un->rcr_groups->elements[i];
                struct group *tg = find_array_element_by_name(
                        n->groups, rg->name);
                uint32_t vz = tg->vector->size;

                /* terminal group */
                struct group *fg = create_group(tg->name, vz, false, true);

                fg->act_fun->fun   = tg->act_fun->fun;
                fg->err_fun->fun   = tg->err_fun->fun;
                fg->act_fun->deriv = tg->act_fun->deriv;
                fg->err_fun->deriv = tg->err_fun->deriv;

                /*
                 * Create a projection between the recurrent group, and the
                 * terminal recurrent group. We only need a unique gradient
                 * and previous gradient matrix for this projection.
                 */
                struct matrix *gradients      = create_matrix(vz, vz);
                struct matrix *prev_gradients = create_matrix(vz, vz);
                
                /* flags */
                struct projection_flags *flags;
                if (!(flags = malloc(sizeof(struct projection_flags))))
                        goto error_out;
                memset(flags, 0, sizeof(struct projection_flags));
                flags->recurrent = true;

                /* create projections */
                struct projection *op =
                        create_projection(tg, un->rcr_weights[i], gradients,
                                prev_gradients, un->rcr_prev_deltas[i],
                                un->rcr_dynamic_params[i], flags);
                struct projection *ip =
                        create_projection(fg, un->rcr_weights[i], gradients,
                                prev_gradients, un->rcr_prev_deltas[i],
                                un->rcr_dynamic_params[i], flags);
                add_projection(fg->out_projs, op);
                add_projection(tg->inc_projs, ip);
        }

        return;

error_out:
        perror("[rnn_attach_recurrent_groups()]");
        return;
}

void rnn_detach_recurrent_groups(struct rnn_unfolded_network *un,
        struct network *n)
{
        for (uint32_t i = 0; i < un->rcr_groups->num_elements; i++) {
                /* recurrent group */
                struct group *rg = un->rcr_groups->elements[i];
                struct group *tg = find_array_element_by_name(
                        n->groups, rg->name);

                /* terminal group */
                struct group *fg = ((struct projection *)
                        tg->inc_projs->elements[
                                tg->inc_projs->num_elements - 1])->to;

                /* remove incoming projection from recurrent group */
                tg->inc_projs->num_elements--;
                struct projection *ip = 
                        tg->inc_projs->elements[tg->inc_projs->num_elements];
                rnn_free_duplicate_projection(ip);
                tg->inc_projs->elements[tg->inc_projs->num_elements] = NULL;
                
                /* remove outgoing projection from terminal group */
                fg->out_projs->num_elements--;
                struct projection *op = fg->out_projs->elements[
                        fg->out_projs->num_elements];
                free(op);
                fg->out_projs->elements[fg->out_projs->num_elements] = NULL;
               
                /* remove terminal group */
                rnn_free_duplicate_groups(fg);
        }
}

void rnn_connect_duplicate_networks(struct rnn_unfolded_network *un,
        struct network *n, struct network *nn)
{
        for (uint32_t i = 0; i < un->rcr_groups->num_elements; i++) {
                /* find recurrent groups to connect */
                struct group *rg = un->rcr_groups->elements[i];
                struct group *fg = find_array_element_by_name(
                        n->groups, rg->name);
                struct group *tg = find_array_element_by_name(
                        nn->groups, rg->name);

                /*
                 * Create a projection between the recurrent groups. We only
                 * need a unique gradient and previous gradient matrix for
                 * this projection.
                 */
                struct matrix *gradients = 
                        create_matrix(fg->vector->size, tg->vector->size);
                struct matrix *prev_gradients = 
                        create_matrix(fg->vector->size, tg->vector->size);

                /* flags */
                struct projection_flags *flags;
                if (!(flags = malloc(sizeof(struct projection_flags))))
                        goto error_out;
                memset(flags, 0, sizeof(struct projection_flags));
                flags->recurrent = true;
                
                /* create projections */
                struct projection *op =
                        create_projection(tg, un->rcr_weights[i], gradients,
                                prev_gradients, un->rcr_prev_deltas[i],
                                un->rcr_dynamic_params[i], flags);
                struct projection *ip =
                        create_projection(fg, un->rcr_weights[i], gradients,
                                prev_gradients, un->rcr_prev_deltas[i],
                                un->rcr_dynamic_params[i], flags);
                add_projection(fg->out_projs, op);
                add_projection(tg->inc_projs, ip);
        }

        return;

error_out:
        perror("[rnn_connect_duplicate_networks()]");
        return;
}

void rnn_disconnect_duplicate_networks(struct rnn_unfolded_network *un,
        struct network *n, struct network *nn)
{
        for (uint32_t i = 0; i < un->rcr_groups->num_elements; i++) {
                struct group *rg = un->rcr_groups->elements[i];
                struct group *fg = find_array_element_by_name(
                        n->groups, rg->name);
                struct group *tg = find_array_element_by_name(
                        nn->groups, rg->name);

                /* remove outgoing projection from 'from' group */
                fg->out_projs->num_elements--;
                struct projection *op =
                        fg->out_projs->elements[fg->out_projs->num_elements];
                rnn_free_duplicate_projection(op);
                fg->out_projs->elements[fg->out_projs->num_elements] = NULL;

                /* remove incoming projection from 'receiving' group */
                tg->inc_projs->num_elements--;
                struct projection *ip =
                        tg->inc_projs->elements[tg->inc_projs->num_elements];
                free(ip);
                tg->inc_projs->elements[tg->inc_projs->num_elements] = NULL;
        }
}

void rnn_sum_gradients(struct rnn_unfolded_network *un)
{
        for (uint32_t i = 1; i < un->stack_size; i++)
                rnn_add_gradients(
                        un->stack[0]->output,
                        un->stack[i]->output);
}

void rnn_add_gradients(struct group *g, struct group *dg)
{
        /*
         * Provided a group g and g', add gradients for all incoming,
         * projections of g' to those in g. Recursively repeat this for all
         * non-recurrent groups projecting to g.
         */
        for (uint32_t i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *p  = g->inc_projs->elements[i];
                struct projection *dp = dg->inc_projs->elements[i];

                /* sum gradients */
                for (uint32_t r = 0; r < p->gradients->rows; r++)
                        for (uint32_t c = 0; c < p->gradients->cols; c++)
                                p->gradients->elements[r][c]
                                        += dp->gradients->elements[r][c];

                /* 
                 * Copy duplicate gradients to previous gradients matrix,
                 * and reset them.
                 */
                copy_matrix(dp->gradients, dp->prev_gradients);
                zero_out_matrix(dp->gradients);

                if (p->flags->recurrent)
                        continue;
                
                rnn_add_gradients(p->to, dp->to);
        }
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Shift the network stack. Assume the following unfolded network:

        .   ...........
        |        |
        |   +---------+       +---------+
        +-->| hidden3 |<--+   | output2 |
            +---------+   |   +---------+
                 |        |        |
            +---------+   |   +---------+
            | input3  |   +-->| hidden2 |<--+
            +---------+       +---------+   |
                                   |        |
                              +---------+   |   +---------+
                              | input2  |   +-->| hidden1 |
                              +---------+       +---------+

stack/n  ...  stack/1           stack/0          terminal

The aim is to completely isolate stack/0, and move it into stack/n.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

void rnn_shift_stack(struct rnn_unfolded_network *un)
{
        /* 
         * First, isolate the network in stack/0 by disconnecting it, and
         * rewiring all of its recurrent terminals to their corresponding
         * groups in stack/1.
         * 
         * Variables: g1 ~ terminal; g2 ~ stack/0, and g3 ~ stack/1.
         */
        for (uint32_t i = 0; i < un->rcr_groups->num_elements; i++) {
                struct group *rg = un->rcr_groups->elements[i];
                struct group *g2 = find_array_element_by_name(  /* stack/0 */
                        un->stack[0]->groups, rg->name);
                struct group *g3 = find_array_element_by_name(  /* stack/1 */
                        un->stack[1]->groups, rg->name);

                /* 
                 * Disconnect the recurrent group in stack/0 from the
                 * terminal recurrent group, and the recurrent group in
                 * stack/1. Whereas we we will free the projection gradient
                 * between stack/0 and the terminal group, we perserve the
                 * one between stack/0 and stack/1. Finally, we copy the
                 * activation pattern of the group in stack/0 into the
                 * terminal group.
                 */
                g2->inc_projs->num_elements--;  /* stack/0 -> terminal */
                struct projection *g2_to_g1 =
                        g2->inc_projs->elements[g2->inc_projs->num_elements];
                struct group *g1 = g2_to_g1->to;
                rnn_free_duplicate_projection(g2_to_g1);
                g2->inc_projs->elements
                        [g2->inc_projs->num_elements] = NULL;
                g2->out_projs->num_elements--;  /* stack0 -> stack/1 */
                free(g2->out_projs->elements[g2->out_projs->num_elements]);
                g2->out_projs->elements[g2->out_projs->num_elements] = NULL;
                copy_vector(g1->vector, g2->vector);

                /* 
                 * Connect the terminal recurrent group to the recurrent
                 * group in stack/1, reusing the gradients that were used
                 * for the projection between stack/0 and stack/1.
                 */
                struct projection *g3_to_g1 =   /* stack/1 -> terminal */
                        g3->inc_projs->elements[g3->inc_projs->num_elements - 1];
                g3_to_g1->to = g1;
                struct projection *g1_to_g3 =   /* terminal -> stack/1 */
                        g1->out_projs->elements[g1->out_projs->num_elements - 1];
                g1_to_g3->to             = g3;
                g1_to_g3->gradients      = g3_to_g1->gradients;
                g1_to_g3->prev_gradients = g3_to_g1->prev_gradients;
        }

        /*
         * Secondly, perform the actual stack shifting. Store a reference to
         * the network in stack/0, and shift stack/1 into stack/0, stack/2
         * into stack/1 until stack/n has been shifted into stack/(n-1).
         * Next, we move the network that used to be in stack/0 into
         * stack/n, and connect it to the network in stack/(n-1).
         */
        struct network *n = un->stack[0];               /* net <- stack/0 */
        for (uint32_t i = 0; i < un->stack_size - 1; i++)
                un->stack[i] = un->stack[i + 1];        /* shift stack */
        un->stack[un->stack_size - 1] = n;              /* stack/n <- net */
        rnn_connect_duplicate_networks(un,
                un->stack[un->stack_size - 2],          /* stack/(n-1) */
                un->stack[un->stack_size - 1]);         /* stack/n */
}
