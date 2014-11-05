/*
 * rnn_unfold.c
 *
 * Copyright 2012-2014 Harm Brouwer <me@hbrouwer.eu>
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

/**************************************************************************
 * This implements the unfolding of recurrent neural networks (RNNs) for
 * backpropagation through time (BPTT; Rumelhart, Hinton, & Williams, 1986),
 * such that they effectively become feed forward networks (FFNs) that can
 * be trained with standard backpropagation (BP). 
 *
 * Assume an RNN with the following topology:
 *
 * +---------+
 * | output1 |
 * +---------+
 *      |
 * +---------+
 * | hidden1 | <-- recurrent group
 * +---------+
 *      |
 * +---------+
 * | input1  |
 * +---------+
 *
 * where _hidden1_ is a recurrent group. The aim is to unfold this network 
 * in time such that its states at different timesteps are connected through
 * recurrent projections:
 *
 * ...........
 *      |
 * +---------+       +---------+
 * | hidden4 |<--+   | output3 |
 * +---------+   |   +---------+
 *      |        |       |
 * +---------+   |   +---------+       +---------+
 * | input4  |   +-->| hidden3 |<--+   | output2 |
 * +---------+       +---------+   |   +---------+
 *                        |        |       |
 *                   +---------+   |   +---------+       +---------+
 *                   | input3  |   +-->| hidden2 |<--+   | output1 |
 *                   +---------+       +---------+   |   +---------+
 *                                          |        |        |
 *                                     +---------+   |   +---------+
 *                                     | input2  |   +-->| hidden1 |
 *                                     +---------+       +---------+
 *                                                            |
 *                                                       +---------+
 *                                                       | input1  |
 *                                                       +---------+
 *
 * Note: Weight matrices, previous weight delta matrices, and dynamic
 *   learning parameter matrices are shared among recurrent projections.
 *
 * References
 *
 * Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning
 *     internal representations by error propagation. In: D. E. Rumelhart &
 *     J. L. McClelland (Eds.), Parallel distributed processing:
 *     Explorations in the microstructure of cognition, Volume 1:
 *     Foundations, pp. 318-362, Cambridge, MA: MIT Press.
 *************************************************************************/
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
        size_t block_size = un->rcr_groups->num_elements * sizeof(struct matrix *);

        /* array for weight matrices */
        if (!(un->rcr_weights = malloc(block_size)))
                goto error_out;
        memset(un->rcr_weights, 0, block_size);

        /* array for previous weight delta matrices */
        if (!(un->rcr_prev_deltas = malloc(block_size)))
                goto error_out;
        memset(un->rcr_prev_deltas, 0, block_size);

        /* array for dynamic learning parameter matrices */
        if (!(un->rcr_dynamic_pars = malloc(block_size)))
                goto error_out;
        memset(un->rcr_dynamic_pars, 0, block_size);

        /*
         * Fill the constructed arrays with the required matrices,
         * and initialize these appropriately.
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
                un->rcr_dynamic_pars[i] = create_matrix(vz, vz);
                if (n->update_algorithm == bp_update_dbd)
                        fill_matrix_with_value(un->rcr_dynamic_pars[i], n->learning_rate);
                if (n->update_algorithm == bp_update_rprop)
                        fill_matrix_with_value(un->rcr_dynamic_pars[i], n->rp_init_update);
        }

        /* 
         * Allocate a stack for duplicate networks. The size of this stack
         * should be equal to the desired number of back ticks plus one
         * (current timestep plus history).
         */
        un->stack_size = n->back_ticks + 1;
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
                        rnn_attach_recurrent_groups(un, un->stack[i]);
                else
                        rnn_connect_duplicate_networks(un, un->stack[i - 1], un->stack[i]);
        }
        
        return un;

error_out:
        perror("[rnn_init_unfolded_network()]");
        return NULL;
}

/**************************************************************************
 *************************************************************************/
void rnn_dispose_unfolded_network(struct rnn_unfolded_network *un)
{
        /* detach the "terminal" recurrent groups */
        rnn_detach_recurrent_groups(un, un->stack[0]);
        
        /* disconnect each network from the one preceding it */
        for (uint32_t i = 1; i < un->stack_size; i++)
                rnn_disconnect_duplicate_networks(un, un->stack[i - 1], un->stack[i]);

        /* dispose recurrent matrices (and their arrays) */
        for (uint32_t i = 0; i < un->rcr_groups->num_elements; i++) {
                dispose_matrix(un->rcr_weights[i]);
                dispose_matrix(un->rcr_prev_deltas[i]);
                dispose_matrix(un->rcr_dynamic_pars[i]);
        }
        free(un->rcr_weights);
        free(un->rcr_prev_deltas);
        free(un->rcr_dynamic_pars);

        /* dispose the array of recurrent groups */
        dispose_array(un->rcr_groups);

        /* dispose networks and stack */
        for (uint32_t i = 0; i < un->stack_size; i++)
                rnn_dispose_duplicate_network(un->stack[i]);
        free(un->stack);

        free(un);
}

/**************************************************************************
 *************************************************************************/
struct network *rnn_duplicate_network(struct network *n)
{
        /* allocate a duplicate network */
        struct network *dn;
        if (!(dn = malloc(sizeof(struct network))))
                goto error_out;
        memset(dn, 0, sizeof(struct network));
        memcpy(dn, n, sizeof(struct network));

        /* duplicate the network's groups */
        dn->groups = create_array(TYPE_GROUPS);
        rnn_duplicate_groups(n, dn, n->input);

        return dn;

error_out:
        perror("[rnn_duplicate_network()])");
        return NULL;
}

/**************************************************************************
 *************************************************************************/
void rnn_dispose_duplicate_network(struct network *dn)
{
        rnn_dispose_duplicate_groups(dn->output);
        dispose_array(dn->groups);

        free(dn);
}

/**************************************************************************
 *************************************************************************/
struct group *rnn_duplicate_group(struct group *g)
{
        struct group *dg;

        if (!(dg = malloc(sizeof(struct group))))
                goto error_out;
        memset(dg, 0, sizeof(struct group));

        size_t block_size = (strlen(g->name) + 1) * sizeof(char);
        if (!(dg->name = malloc(block_size)))
                goto error_out;
        memset(dg->name, 0, block_size);
        strncpy(dg->name, g->name, strlen(g->name));

        dg->vector = create_vector(g->vector->size);
        dg->error = create_vector(g->vector->size);

        if (!(dg->act_fun = malloc(sizeof(struct act_fun))))
                goto error_out;
        memset(dg->act_fun, 0, sizeof(struct act_fun));
        dg->act_fun->fun = g->act_fun->fun;
        dg->act_fun->deriv = g->act_fun->deriv;

        if (!(dg->err_fun = malloc(sizeof(struct err_fun))))
                goto error_out;
        memset(dg->err_fun, 0, sizeof(struct err_fun));
        dg->err_fun->fun = g->err_fun->fun;
        dg->err_fun->deriv = g->err_fun->deriv;

        dg->inc_projs = create_array(TYPE_PROJS);
        dg->inc_projs->num_elements = g->inc_projs->num_elements;
        dg->out_projs = create_array(TYPE_PROJS);
        dg->out_projs->num_elements = g->out_projs->num_elements;

        dg->bias = g->bias;
        dg->recurrent = g->recurrent;

        return dg;

error_out:
        perror("[rnn_duplicate_group()]");
        return NULL;
}

/**************************************************************************
 *************************************************************************/
struct group *rnn_duplicate_groups(struct network *n, struct network *dn, 
                struct group *g)
{
        /* duplicate groups */
        struct group *dg = rnn_duplicate_group(g);
        add_to_array(dn->groups, dg);

        /* set input and output groups */
        if (n->input == g)
                dn->input = dg;
        if (n->output == g)
                dn->output = dg;

        /* if the current group has a bias group, duplicate it */
        for (uint32_t i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *ip = g->inc_projs->elements[i];
                struct group *bg = ip->to;
                
                /* skip non-bias groups */
                if (!bg->bias)
                        continue;

                /* duplicate bias group */
                struct group *dbg = rnn_duplicate_group(bg);
                add_to_array(dn->groups, dbg);

                /*
                 * Duplicate the projection between the bias group and its
                 * corresponding group.
                 *
                 * Note: We only need a unique gradient and previous
                 *   gradient matrix for this projection.
                 */
                struct matrix *gradients =
                        create_matrix(1, g->vector->size);
                struct matrix *prev_gradients =
                        create_matrix(1, g->vector->size);
                dg->inc_projs->elements[i] =
                        rnn_duplicate_projection(dbg, ip, gradients, prev_gradients);
                dbg->out_projs->elements[0] = 
                        rnn_duplicate_projection(dg, bg->out_projs->elements[0],
                                        gradients, prev_gradients);
        }

        /* recursively duplicate the current group's outgoing projections */
        for (uint32_t i = 0; i < g->out_projs->num_elements; i++) {
                struct projection *op = g->out_projs->elements[i];
                struct group *g2 = op->to;

                /* recursively duplicate groups */
                struct group *rg = rnn_duplicate_groups(n, dn, g2);

                /*
                 * Duplicate the projection between the current group
                 * and the group to which it projects.
                 * 
                 * Note: We only need a unique gradient and previous
                 *   gradient matrix for this projection.
                 */
                struct matrix *gradients =
                        create_matrix(g->vector->size, g2->vector->size);
                struct matrix *prev_gradients =
                        create_matrix(g->vector->size, g2->vector->size);
                dg->out_projs->elements[i] = 
                        rnn_duplicate_projection(rg, op, gradients, prev_gradients);
                
                /*
                 * Add the required incoming projection.
                 *
                 * Note: We want to add this projection in the same array
                 *   location as the original projection.
                 */
                for (uint32_t j = 0; j < g2->inc_projs->num_elements; j++) {
                        struct projection *ip = g2->inc_projs->elements[j];
                        if (ip->to != g)
                                continue;
                        rg->inc_projs->elements[j] =
                                rnn_duplicate_projection(dg, ip, gradients, prev_gradients);
                }
        }

        return dg;
}

/**************************************************************************
 *************************************************************************/
void rnn_dispose_duplicate_groups(struct group *dg)
{
        /* recursively dispose incoming projections */
        for (uint32_t i = 0; i < dg->inc_projs->num_elements; i++) {
                struct projection *ip = dg->inc_projs->elements[i];
                rnn_dispose_duplicate_groups(ip->to);
                rnn_dispose_duplicate_projection(ip);
        }
        dispose_array(dg->inc_projs);

        /* free outgoing projections */
        for (uint32_t i = 0; i < dg->out_projs->num_elements; i++)
                free(dg->out_projs->elements[i]);
        dispose_array(dg->out_projs);

        /* dispose context groups (if allocated) */
        if (dg->ctx_groups)
                dispose_array(dg->ctx_groups);

        dispose_vector(dg->vector);
        dispose_vector(dg->error);

        free(dg->act_fun);
        free(dg->err_fun);

        free(dg->name);
        free(dg);
}

/**************************************************************************
 *************************************************************************/
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

        dp->to = to;
        dp->weights = p->weights;             /* <-- shared */
        dp->gradients = gradients;
        dp->prev_gradients = prev_gradients;
        dp->prev_deltas = p->prev_deltas;     /* <-- shared */
        dp->dynamic_pars = p->dynamic_pars;   /* <-- shared */

        return dp;

error_out:
        perror("[rnn_duplicate_projection()]");
        return NULL;
}

/**************************************************************************
 *************************************************************************/
void rnn_dispose_duplicate_projection(struct projection *dp)
{
        dispose_matrix(dp->gradients);
        dispose_matrix(dp->prev_gradients);

        free(dp);
}

/**************************************************************************
 *************************************************************************/
struct array *rnn_recurrent_groups(struct network *n)
{
        struct array *gs = create_array(TYPE_GROUPS);
        rnn_collect_recurrent_groups(n->input, gs);

        return gs;
}

/**************************************************************************
 *************************************************************************/
void rnn_collect_recurrent_groups(struct group *g, struct array *gs)
{
        if (g->recurrent)
                add_to_array(gs, g);
        for (uint32_t i = 0; i < g->out_projs->num_elements; i++) {
                struct projection *op = g->out_projs->elements[i];
                rnn_collect_recurrent_groups(op->to, gs);
        }
}

/**************************************************************************
 *************************************************************************/
void rnn_attach_recurrent_groups(struct rnn_unfolded_network *un,
                struct network *n)
{
        for (uint32_t i = 0; i < un->rcr_groups->num_elements; i++) {
                /* recurrent group */
                struct group *rg = un->rcr_groups->elements[i];
                struct group *g1 = find_array_element_by_name(n->groups, rg->name);
                uint32_t vz = g1->vector->size;

                /* "terminal" group */
                struct group *g2 = create_group(g1->name, vz, false, true);

                g2->act_fun->fun = g1->act_fun->fun;
                g2->err_fun->fun = g1->err_fun->fun;
                g2->act_fun->deriv = g1->act_fun->deriv;
                g2->err_fun->deriv = g1->err_fun->deriv;

                /*
                 * Create a projection between the recurrent group,
                 * and the "terminal" recurrent group.
                 *
                 * Note: We only need a unique gradient and previous
                 *   gradient matrix for this projection.
                 */
                struct matrix *gradients = create_matrix(vz, vz);
                struct matrix *prev_gradients = create_matrix(vz, vz);
                struct projection *op =
                        create_projection(g1, un->rcr_weights[i], gradients, prev_gradients,
                                        un->rcr_prev_deltas[i], un->rcr_dynamic_pars[i]);
                struct projection *ip =
                        create_projection(g2, un->rcr_weights[i], gradients, prev_gradients,
                                        un->rcr_prev_deltas[i], un->rcr_dynamic_pars[i]);

                op->recurrent = true;
                ip->recurrent = true;

                add_to_array(g2->out_projs, op);
                add_to_array(g1->inc_projs, ip);
        }
}

/**************************************************************************
 *************************************************************************/
void rnn_detach_recurrent_groups(struct rnn_unfolded_network *un,
                struct network *n)
{
        for (uint32_t i = 0; i < un->rcr_groups->num_elements; i++) {
                /* recurrent group */
                struct group *rg = un->rcr_groups->elements[i];
                struct group *g1 = find_array_element_by_name(n->groups, rg->name);

                /* "terminal" group */
                uint32_t z = g1->inc_projs->num_elements - 1;
                struct projection *ip = g1->inc_projs->elements[z];
                struct group *g2 = ip->to;

                /* dispose projection between groups */
                g1->inc_projs->num_elements--;
                g2->out_projs->num_elements--;
                struct projection *p1 = g1->inc_projs->elements[g1->inc_projs->num_elements];
                struct projection *p2 = g2->out_projs->elements[g2->out_projs->num_elements];

                rnn_dispose_duplicate_projection(p1);
                free(p2);

                g1->inc_projs->elements[g1->inc_projs->num_elements] = NULL;
                g2->out_projs->elements[g2->out_projs->num_elements] = NULL;
               
                /* dispose the "terminal" group */
                rnn_dispose_duplicate_groups(g2);
        }
}

/**************************************************************************
 *************************************************************************/
void rnn_connect_duplicate_networks(struct rnn_unfolded_network *un,
                struct network *n1, struct network *n2)
{
        for (uint32_t i = 0; i < un->rcr_groups->num_elements; i++) {
                /* find recurrent groups to connect */
                struct group *rg = un->rcr_groups->elements[i];
                struct group *g1 = find_array_element_by_name(n1->groups, rg->name);
                struct group *g2 = find_array_element_by_name(n2->groups, rg->name);

                /*
                 * Create a projection between the recurrent groups.
                 *
                 * Note: We only need a unique gradient and previous
                 *   gradient matrix for this projection.
                 */
                struct matrix *gradients = 
                        create_matrix(g1->vector->size, g2->vector->size);
                struct matrix *prev_gradients = 
                        create_matrix(g1->vector->size, g2->vector->size);
                struct projection *op =
                        create_projection(g2, un->rcr_weights[i], gradients, prev_gradients,
                                        un->rcr_prev_deltas[i], un->rcr_dynamic_pars[i]);
                struct projection *ip =
                        create_projection(g1, un->rcr_weights[i], gradients, prev_gradients,
                                        un->rcr_prev_deltas[i], un->rcr_dynamic_pars[i]);

                op->recurrent = true;
                ip->recurrent = true;

                add_to_array(g1->out_projs, op);
                add_to_array(g2->inc_projs, ip);
        }
}

/**************************************************************************
 *************************************************************************/
void rnn_disconnect_duplicate_networks(struct rnn_unfolded_network *un,
                struct network *n1, struct network *n2)
{
        for (uint32_t i = 0; i < un->rcr_groups->num_elements; i++) {
                struct group *rg = un->rcr_groups->elements[i];
                struct group *g1 = find_array_element_by_name(n1->groups, rg->name);
                struct group *g2 = find_array_element_by_name(n2->groups, rg->name);

                g1->out_projs->num_elements--;
                g2->inc_projs->num_elements--;
                struct projection *p1 = g1->out_projs->elements[g1->out_projs->num_elements];
                struct projection *p2 = g2->inc_projs->elements[g2->inc_projs->num_elements];

                rnn_dispose_duplicate_projection(p1);
                free(p2);

                g1->out_projs->elements[g1->out_projs->num_elements] = NULL;
                g2->inc_projs->elements[g2->inc_projs->num_elements] = NULL;
        }
}

/**************************************************************************
 *************************************************************************/
void rnn_sum_gradients(struct rnn_unfolded_network *un)
{
        for (uint32_t i = 1; i < un->stack_size; i++)
                rnn_add_gradients(un->stack[0]->output, un->stack[i]->output);
}

/**************************************************************************
 *************************************************************************/
void rnn_add_gradients(struct group *g1, struct group *g2)
{
        for (uint32_t i = 0; i < g1->inc_projs->num_elements; i++) {
                struct projection *p1 = g1->inc_projs->elements[i];
                struct projection *p2 = g2->inc_projs->elements[i];

                for (uint32_t r = 0; r < p1->gradients->rows; r++)
                        for (uint32_t c = 0; c < p1->gradients->cols; c++)
                                p1->gradients->elements[r][c] += p2->gradients->elements[r][c];

                copy_matrix(p2->gradients, p2->prev_gradients);
                zero_out_matrix(p2->gradients);

                if (!p1->recurrent)
                        rnn_add_gradients(p1->to, p2->to);
        }
}

/**************************************************************************
 * Shift the network stack. Assume the following unfolded network:
 *
 *         .   ...........
 *         |        |
 *         |   +---------+       +---------+
 *         +-->| hidden3 |<--+   | output2 |
 *             +---------+   |   +---------+
 *                  |        |        |
 *             +---------+   |   +---------+
 *             | input3  |   +-->| hidden2 |<--+
 *             +---------+       +---------+   |
 *                                    |        |
 *                               +---------+   |   +---------+
 *                               | input2  |   +-->| hidden1 |
 *                               +---------+       +---------+
 *
 * stack/n  ...  stack/1           stack/0          terminal
 *
 * The aim is to completely isolate stack[0], and move it into stack[n].
 *************************************************************************/
void rnn_shift_stack(struct rnn_unfolded_network *un)
{
        /* 
         * Isolate the network in stack/0 by rewiring all of its recurrent
         * terminals to their corresponding groups in stack/1.
         */
        for (uint32_t i = 0; i < un->rcr_groups->num_elements; i++) {
                /* current recurrent group */
                struct group *rg = un->rcr_groups->elements[i];

                /* recurrent groups in stack/0 and stack/1 */
                struct group *g1 = find_array_element_by_name(un->stack[0]->groups, rg->name);
                struct group *g2 = find_array_element_by_name(un->stack[1]->groups, rg->name);

                /* store a reference to the terminal recurrent group */
                uint32_t j = g1->inc_projs->num_elements - 1;
                struct projection *p1 = g1->inc_projs->elements[j];
                struct group *g0 = p1->to;

                /*
                 * Disconnect the current recurrent group in stack/0 from 
                 * its copy in stack/1, and dispose the projection gradient.
                 */
                rnn_dispose_duplicate_projection(p1);
                g1->inc_projs->elements[j] = NULL;
                g1->inc_projs->num_elements--;

                /*
                 * Disconnect the recurrent group in stack/0 from its
                 * copy in stack/1, preserving the projection gradient.
                 */
                j = g1->out_projs->num_elements - 1;
                free(g1->out_projs->elements[j]);
                g1->out_projs->elements[j] = NULL;
                g1->out_projs->num_elements--;

                /* 
                 * Copy the activation vector of the recurrent group in
                 * stack/0 into the terminal group.
                 */
                copy_vector(g0->vector, g1->vector);

                /* 
                 * Connect the terminal recurrent group to the recurrent
                 * group in stack/1, reusing the gradients that were used
                 * for the projection between stack/0 and stack/1.
                 *
                 * First add an incoming projection to the recurrent group
                 * in stack/1:
                 */
                j = g2->inc_projs->num_elements - 1;
                p1 = g2->inc_projs->elements[j];
                p1->to = g0;
                
                /* and, add an outgoing projection to the terminal group */
                j = g0->out_projs->num_elements - 1;
                struct projection *p2 = g0->out_projs->elements[j];
                p2->to = g2;
                p2->gradients = p1->gradients;
                p2->prev_gradients = p1->prev_gradients;
        }

        /*
         * Now all recurrent groups have been rewired, and the
         * actual stack shifting can be done. We store a reference
         * to the network state in stack/0, and shift stack/1 into
         * stack/0, stack/2 into stack/1, until stack/n has been
         * shifted into stack/(n-1), upon which we place stack/0
         * in stack/n
         */
        struct network *n = un->stack[0];
        for (uint32_t i = 0; i < un->stack_size - 1; i++)
                un->stack[i] = un->stack[i + 1];
        un->stack[un->stack_size - 1] = n;

        /* connect stack/(n-1) to stack/n */
        rnn_connect_duplicate_networks(un, un->stack[un->stack_size - 2],
                        un->stack[un->stack_size - 1]);
}
