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

struct rnn_unfolded_network *rnn_unfold_network(struct network *n)
{
        struct rnn_unfolded_network *un;
        if (!(un = malloc(sizeof(struct rnn_unfolded_network))))
                goto error_out;
        memset(un, 0, sizeof(struct rnn_unfolded_network));

        /*
         * Find all recurrent groups in the network, and create a "terminal"
         * group for each.
         */
        un->rcr_groups = create_array(atype_groups);
        un->trm_groups = create_array(atype_groups);        
        rnn_find_recurrent_groups(n->input, un->rcr_groups);
        for (uint32_t i = 0; i < un->rcr_groups->num_elements; i++) {
                struct group *rg = un->rcr_groups->elements[i];
                struct group *tg = rnn_duplicate_group(rg);
                add_to_array(un->trm_groups, tg);
        }

        /* 
         * Allocate a stack for duplicate networks. The size of this stack
         * should be equal to the desired number of back ticks plus one
         * (current timestep plus history).
         */
        un->stack_size = n->pars->back_ticks + 1;
        size_t block_size = un->stack_size * sizeof(struct network *);
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
                un->stack[i]->unfolded_net = un; /* <- unfolded network */
                if (i == 0) {
                        /* attach "terminal" groups */
                        rnn_attach_terminal_groups(un,
                                un->stack[i]);
                } else {
                        /* connect duplicate networks */
                        rnn_connect_duplicate_networks(un,
                                un->stack[i - 1],
                                un->stack[i]);
                }
        }

        return un;

error_out:
        perror("[rnn_init_unfolded_network()]");
        return NULL;
}

void rnn_find_recurrent_groups(struct group *g, struct array *rcr_groups)
{
        if (find_projection(g->out_projs, g))
                add_to_array(rcr_groups, g);
        for (uint32_t i = 0; i < g->out_projs->num_elements; i++) {
                struct projection *op = g->out_projs->elements[i];
                if (op->flags->recurrent)
                        continue;
                rnn_find_recurrent_groups(op->to, rcr_groups);
        }
}

void rnn_free_unfolded_network(struct rnn_unfolded_network *un)
{
        rnn_detach_terminal_groups(un, un->stack[0]);
        for (uint32_t i = 1; i < un->stack_size; i++)
                rnn_disconnect_duplicate_networks(un,
                        un->stack[i - 1],
                        un->stack[i]);
        free_array(un->rcr_groups);
        free_array(un->trm_groups);
        for (uint32_t i = 0; i < un->stack_size; i++)
                rnn_free_duplicate_network(un->stack[i]);
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
        // rnn_free_duplicate_groups(dn->output);
        rnn_free_duplicate_groups(dn->groups);
        free_array(dn->groups);
        free(dn);
}

struct group *rnn_duplicate_group(struct group *g)
{
        struct group *dg;
        if (!(dg = malloc(sizeof(struct group))))
                goto error_out;
        memset(dg, 0, sizeof(struct group));
        
        dg->name       = g->name;
        dg->vector     = create_vector(g->vector->size);
        dg->error      = create_vector(g->vector->size);
        dg->act_fun    = g->act_fun;
        dg->err_fun    = g->err_fun;
        dg->inc_projs  = create_array(atype_projs);
        dg->out_projs  = create_array(atype_projs);
        dg->ctx_groups = NULL;
        dg->flags      = g->flags;
        dg->pars       = g->pars;

        if (dg->flags->bias)
                copy_vector(dg->vector, g->vector);

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
        if (n->input == g)
                dn->input  = dg;
        if (n->output == g)
                dn->output = dg;

        /* if the current group has a bias group, duplicate it */
        for (uint32_t i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *ip = g->inc_projs->elements[i];
                struct group *bg      = ip->to;
                /* skip non-bias groups */
                if (!bg->flags->bias)
                        continue;
                /* only duplicate unseen bias groups */
                struct group *dbg = find_array_element_by_name(dn->groups, bg->name);
                if (!dbg) {
                        dbg = rnn_duplicate_group(bg);
                        add_group(dn, dbg);
                }
                /*
                 * Duplicate the projection between the current group and
                 * its bias group. We only need a unique gradient and
                 * previous gradient matrix.
                 */                
                struct matrix *gradients =
                        create_matrix(1, g->vector->size);
                struct matrix *prev_gradients =
                        create_matrix(1, g->vector->size);
                add_to_array(dg->inc_projs, rnn_duplicate_projection(
                        dbg, ip, gradients, prev_gradients));
                struct projection *op = find_projection(bg->out_projs, g);
                add_to_array(dbg->out_projs, rnn_duplicate_projection(
                        dg, op, gradients, prev_gradients));             
        }

        /* recursively duplicate the current group's outgoing projections */
        for (uint32_t i = 0; i < g->out_projs->num_elements; i++) {
                struct projection *op = g->out_projs->elements[i];
                struct group *tg      = op->to;
                /* skip reucrrent projections */
                if (op->flags->recurrent)
                        continue;
                /* recursively duplicate groups */
                struct group *rg = rnn_duplicate_groups(n, dn, tg);
                /*
                 * Duplicate the projection between the current group and
                 * the group to which it projects. We only need a unique
                 * gradient and previous gradient matrix.
                 */
                struct matrix *gradients =
                        create_matrix(g->vector->size, tg->vector->size);
                struct matrix *prev_gradients =
                        create_matrix(g->vector->size, tg->vector->size);
                add_to_array(dg->out_projs, rnn_duplicate_projection(
                        rg, op, gradients, prev_gradients));
                struct projection *ip = find_projection(tg->inc_projs, g);
                add_to_array(rg->inc_projs, rnn_duplicate_projection(
                        dg, ip, gradients, prev_gradients));
        }

        return dg;
}

void rnn_free_duplicate_group(struct group *dg)
{
        free_vector(dg->vector);
        free_vector(dg->error);
        for (uint32_t i = 0; i < dg->inc_projs->num_elements; i++)
                rnn_free_duplicate_projection(dg->inc_projs->elements[i]);
        free_array(dg->inc_projs);
        for (uint32_t i = 0; i < dg->out_projs->num_elements; i++)
                free(dg->out_projs->elements[i]);
        free_array(dg->out_projs);
        free(dg);        
}

void rnn_free_duplicate_groups(struct array *dgs)
{
        for (uint32_t i = 0; i < dgs->num_elements; i++)
                rnn_free_duplicate_group(dgs->elements[i]);
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

void rnn_attach_terminal_groups(struct rnn_unfolded_network *un,
        struct network *n)
{
        for (uint32_t i = 0; i < un->rcr_groups->num_elements; i++) {
                struct group *rg = un->rcr_groups->elements[i];
                struct group *tg = find_array_element_by_name(
                        n->groups, rg->name);
                struct group *fg = un->trm_groups->elements[i];

                /*
                 * Create a projection between the recurrent group, and the
                 * terminal recurrent group. We only need a unique gradient
                 * and previous gradient matrix.
                 */
                uint32_t vz = tg->vector->size;
                struct matrix *gradients      = create_matrix(vz, vz);
                struct matrix *prev_gradients = create_matrix(vz, vz);
                /* flags */
                struct projection_flags *flags;
                if (!(flags = malloc(sizeof(struct projection_flags))))
                        goto error_out;
                memset(flags, 0, sizeof(struct projection_flags));
                flags->recurrent = true; /* <-- flag recurrent */
                /* create projections */
                struct projection *rp = find_projection(rg->out_projs, rg);
                struct projection *op = create_projection(tg, rp->weights,
                        gradients, prev_gradients, rp->prev_deltas,
                        rp->dynamic_params, flags);
                struct projection *ip = create_projection(fg, rp->weights,
                        gradients, prev_gradients, rp->prev_deltas,
                        rp->dynamic_params, flags);
                add_projection(fg->out_projs, op);
                add_projection(tg->inc_projs, ip);
        }

        return;

error_out:
        perror("[rnn_attach_recurrent_groups()]");
        return;
}

void rnn_detach_terminal_groups(struct rnn_unfolded_network *un,
        struct network *n)
{
        for (uint32_t i = 0; i < un->rcr_groups->num_elements; i++) {
                struct group *rg = un->rcr_groups->elements[i];
                struct group *tg = find_array_element_by_name(
                        n->groups, rg->name);
                struct group *fg = un->trm_groups->elements[i];
                /* remove incoming projection from recurrent group */
                struct projection *ip = find_projection(tg->inc_projs, fg);
                remove_projection(tg->inc_projs, ip);
                rnn_free_duplicate_projection(ip);
                /* remove outgoing projection from terminal group */
                struct projection *op = find_projection(fg->out_projs, tg);
                remove_projection(fg->out_projs, op);
                free(op);
                /* remove terminal group */
                // rnn_free_duplicate_groups(fg);
                rnn_free_duplicate_group(fg);
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
                 * need a unique gradient and previous gradient matrix.
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
                flags->recurrent = true; /* <-- flag recurrent */
                /* create projections */
                struct projection *rp = find_projection(rg->out_projs, rg);
                struct projection *op = create_projection(tg, rp->weights,
                        gradients, prev_gradients, rp->prev_deltas,
                        rp->dynamic_params, flags);
                struct projection *ip = create_projection(fg, rp->weights,
                        gradients, prev_gradients, rp->prev_deltas,
                        rp->dynamic_params, flags);
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
                struct projection *op = find_projection(fg->out_projs, tg);
                remove_projection(fg->out_projs, op);
                rnn_free_duplicate_projection(op);
                /* remove incoming projection from 'receiving' group */
                struct projection *ip = find_projection(tg->inc_projs, fg);
                remove_projection(tg->inc_projs, ip);
                free(ip);
        }
}

void rnn_sum_and_reset_gradients(struct rnn_unfolded_network *un)
{
        for (uint32_t i = 1; i < un->stack_size; i++)
                rnn_add_and_reset_gradients(
                        un->stack[0]->output,
                        un->stack[i]->output);
}

void rnn_add_and_reset_gradients(struct group *g, struct group *dg)
{
        /*
         * Provided a group g and g', add gradients for all incoming,
         * projections of g' to those in g. Recursively repeat this for all
         * non-recurrent groups projecting to g.
         */
        for (uint32_t i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *p = g->inc_projs->elements[i];                
                struct projection *dp;
                for (uint32_t j = 0; j < dg->inc_projs->num_elements; j++) {
                        dp = dg->inc_projs->elements[j];
                        if (dp->to->name == p->to->name)
                                break;
                }
                /* skip recurrent projections */
                if (p->flags->recurrent)
                        continue;
                for (uint32_t r = 0; r < p->gradients->rows; r++)
                        for (uint32_t c = 0; c < p->gradients->cols; c++)
                                p->gradients->elements[r][c]
                                        += dp->gradients->elements[r][c];
                copy_matrix(dp->gradients, dp->prev_gradients);
                zero_out_matrix(dp->gradients);
                rnn_add_and_reset_gradients(p->to, dp->to);
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
                struct group *g1 = un->trm_groups->elements[i]; /* terminal */
                struct group *g2 = find_array_element_by_name(  /* stack/0  */
                        un->stack[0]->groups, rg->name);
                struct group *g3 = find_array_element_by_name(  /* stack/1  */
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
                struct projection *g2_to_g1 = /* stack0 -> terminal */
                        find_projection(g2->inc_projs, g1);
                remove_projection(g2->inc_projs, g2_to_g1);
                rnn_free_duplicate_projection(g2_to_g1);
                struct projection *g2_to_g3 = /* stack0 -> stack/1 */
                        find_projection(g2->out_projs, g3);
                remove_projection(g2->out_projs, g2_to_g3);
                free(g2_to_g3);
                copy_vector(g1->vector, g2->vector);

                /* 
                 * Connect the terminal recurrent group to the recurrent
                 * group in stack/1, reusing the gradients that were used
                 * for the projection between stack/0 and stack/1.
                 */
                struct projection *g3_to_g1 = find_projection(g3->inc_projs, g2);
                g3_to_g1->to = g1; /* stack/1 -> terminal */                
                struct projection *g1_to_g3 = find_projection(g1->out_projs, g2);
                g1_to_g3->to = g3; /* terminal -> stack/1 */
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
