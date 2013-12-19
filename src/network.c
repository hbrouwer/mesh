/*
 * network.c
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

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "act.h"
#include "bp.h"
#include "defaults.h"
#include "main.h"
#include "math.h"
#include "network.h"
#include "random.h"
#include "sanity.h"
#include "train.h"

/**************************************************************************
 *************************************************************************/
struct network *create_network(char *name, uint32_t type)
{
        struct network *n;
        if (!(n = malloc(sizeof(struct network))))
                goto error_out;
        memset(n, 0, sizeof(struct network));

        size_t block_size = (strlen(name) + 1) * sizeof(char);
        if (!(n->name = malloc(block_size)))
                goto error_out;
        memset(n->name, 0, block_size);
        strncpy(n->name, name, strlen(name));

        n->type = type;

        n->groups = create_array(TYPE_GROUPS);
        n->sets = create_array(TYPE_SETS);

        block_size = sizeof(struct status);
        if (!(n->status = malloc(block_size)))
                goto error_out;
        memset(n->status, 0, block_size);

        set_network_defaults(n);

        return n;

error_out:
        perror("[create_network()]");
        return NULL;
}

/**************************************************************************
 *************************************************************************/
void set_network_defaults(struct network *n)
{
        n->random_algorithm   = DFLT_RANDOM_ALGORITHM;
        n->random_mu          = DFLT_RANDOM_MU;
        n->random_sigma       = DFLT_RANDOM_SIGMA;
        n->random_min         = DFLT_RANDOM_MIN;
        n->random_max         = DFLT_RANDOM_MAX;

        n->learning_rate      = DFLT_LEARNING_RATE;
        n->lr_scale_factor    = DFLT_LR_SCALE_FACTOR;
        n->lr_scale_after     = DFLT_LR_SCALE_AFTER;

        n->momentum           = DFLT_MOMENTUM;
        n->mn_scale_factor    = DFLT_MN_SCALE_FACTOR;
        n->mn_scale_after     = DFLT_MN_SCALE_AFTER;

        n->weight_decay       = DFLT_WEIGHT_DECAY;
        n->wd_scale_factor    = DFLT_WD_SCALE_FACTOR;
        n->wd_scale_after     = DFLT_WD_SCALE_AFTER;

        n->target_radius      = DFLT_TARGET_RADIUS;
        n->zero_error_radius  = DFLT_ZERO_ERROR_RADIUS;

        n->error_threshold    = DFLT_ERROR_THRESHOLD;
        n->max_epochs         = DFLT_MAX_EPOCHS;
        n->report_after       = DFLT_REPORT_AFTER;

        n->rp_init_update     = DFLT_RP_INIT_UPDATE;
        n->rp_eta_plus        = DFLT_RP_ETA_PLUS;
        n->rp_eta_minus       = DFLT_RP_ETA_MINUS;

        n->dbd_rate_increment = DFLT_DBD_RATE_INCREMENT;
        n->dbd_rate_decrement = DFLT_DBD_RATE_DECREMENT;

        n->similarity_metric  = DFLT_SIMILARITY_METRIC;
}

/**************************************************************************
 *************************************************************************/
void init_network(struct network *n)
{
        n->initialized = false;

        if (!verify_network_sanity(n)) {
                eprintf("Cannot initialize network--network is not 'sane'");
                return;
        }

        /* randomize weights */
        srand(n->random_seed);
        randomize_weight_matrices(n->input, n);

        /* initialize dynamic learning parameters */
        initialize_dynamic_pars(n->input, n);

        /* initialize activation function look up table (if required) */
        if (n->act_lookup)
                initialize_act_lookup_vectors(n);

        /* 
         * If batch size is set to 0, set it to the number of items
         * in the active set.
         */
        if (n->batch_size == 0 && n->asp)
                n->batch_size = n->asp->items->num_elements;

        /* unfold RNN (if required) */
        if (n->learning_algorithm == train_network_with_bptt)
                n->unfolded_net = rnn_init_unfolded_network(n);

        /* flag network as initialized */
        n->initialized = true;
}

/**************************************************************************
 *************************************************************************/
void reset_network(struct network *n)
{
        randomize_weight_matrices(n->input, n);
        initialize_dynamic_pars(n->input, n);
}

/**************************************************************************
 *************************************************************************/
void dispose_network(struct network *n)
{
        free(n->name);
        free(n->status);

        if (n->unfolded_net)
                rnn_dispose_unfolded_network(n->unfolded_net);

        dispose_groups(n->groups);
        dispose_array(n->groups);

        dispose_sets(n->sets);
        dispose_array(n->sets);

        free(n);
}

/**************************************************************************
 *************************************************************************/
struct group *create_group(char *name, uint32_t size, bool bias,
                bool recurrent)
{
        struct group *g;
        if (!(g = malloc(sizeof(struct group))))
                goto error_out;
        memset(g, 0, sizeof(struct group));

        size_t block_size = (strlen(name) + 1) * sizeof(char);
        if (!(g->name = malloc(block_size)))
                goto error_out;
        memset(g->name, 0, block_size);
        strncpy(g->name, name, strlen(name));

        g->vector = create_vector(size);
        g->error = create_vector(size);

        if (!(g->act_fun = malloc(sizeof(struct act_fun))))
                goto error_out;
        memset(g->act_fun, 0, sizeof(struct act_fun));
        g->act_fun->fun = act_fun_linear;
        g->act_fun->deriv = act_fun_linear_deriv;

        if (!(g->err_fun = malloc(sizeof(struct err_fun))))
                goto error_out;
        memset(g->err_fun, 0, sizeof(struct err_fun));

        g->inc_projs = create_array(TYPE_PROJS);
        g->out_projs = create_array(TYPE_PROJS);

        g->ctx_groups = create_array(TYPE_GROUPS);

        g->bias = bias;
        g->recurrent = recurrent;

        if(g->bias)
                g->vector->elements[0] = 1.0;

        return g;

error_out:
        perror("[create_group()]");
        return NULL;
}

/**************************************************************************
 *************************************************************************/
struct group *attach_bias_group(struct network *n, struct group *g)
{
        /* create a new "bias" group */
        char *tmp;
        size_t block_size = (strlen(g->name) + 6) * sizeof(char);
        if (!(tmp = malloc(block_size)))
                goto error_out;
        memset(tmp, 0, sizeof(block_size));

        sprintf(tmp, "%s_bias", g->name);
        struct group *bg = create_group(tmp, 1, true, false);

        bg->act_fun->fun = g->act_fun->fun;
        bg->act_fun->deriv = g->act_fun->deriv;
        
        bg->err_fun->fun = g->err_fun->fun;
        bg->err_fun->deriv = g->err_fun->deriv;

        free(tmp);

        /* add "bias" group to the network */
        add_to_array(n->groups, bg);

        /* weight matrix */
        struct matrix *weights = create_matrix(
                        bg->vector->size,
                        g->vector->size);

        /* gradients matrix */
        struct matrix *gradients = create_matrix(
                        bg->vector->size,
                        g->vector->size);

        /* previous gradients matrix */
        struct matrix *prev_gradients = create_matrix(
                        bg->vector->size,
                        g->vector->size);

        /* previous weight deltas matrix */
        struct matrix *prev_deltas = create_matrix(
                        bg->vector->size,
                        g->vector->size);

        /* dynamic learning parameters matrix */
        struct matrix *dynamic_pars = create_matrix(
                        bg->vector->size,
                        g->vector->size);

        /* add projections */
        struct projection *op;
        op = create_projection(g, weights, gradients, prev_gradients,
                                prev_deltas, dynamic_pars, false);
        add_to_array(bg->out_projs, op);
        
        struct projection *ip;
        ip = create_projection(bg, weights, gradients, prev_gradients,
                                prev_deltas, dynamic_pars, false);
        add_to_array(g->inc_projs, ip);

        return bg;

error_out:
        perror("[attach_bias_group()]");
        return NULL;
}

/**************************************************************************
 *************************************************************************/
void dispose_group(struct group *g)
{
        free(g->name);
        dispose_vector(g->vector);
        dispose_vector(g->error);
        if (!g->bias) {
                if (g->act_fun->lookup)
                        dispose_vector(g->act_fun->lookup);
                free(g->act_fun);
                free(g->err_fun);
        }

        for (uint32_t j = 0; j < g->inc_projs->num_elements; j++)
                dispose_projection(g->inc_projs->elements[j]);
        for (uint32_t j = 0; j < g->out_projs->num_elements; j++)
                free(g->out_projs->elements[j]);

        dispose_array(g->inc_projs);
        dispose_array(g->out_projs);

        dispose_array(g->ctx_groups);

        free(g);
}

/**************************************************************************
 *************************************************************************/
void dispose_groups(struct array *gs)
{
        for (uint32_t i = 0; i < gs->num_elements; i++)
                dispose_group(gs->elements[i]);
}

/**************************************************************************
 *************************************************************************/
void shift_context_groups(struct network *n)
{
        for (uint32_t i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];
                for (uint32_t j = 0; j < g->ctx_groups->num_elements; j++) {
                        struct group *cg = g->ctx_groups->elements[j];
                        shift_context_group_chain(cg, g->vector);
                }
        }
}

/**************************************************************************
 * Shifts a context group chain. If group g has a context group c, then the
 * activity vector of g is copied into that of c. However, if c has itself
 * a context group c', then the activity pattern of c is first copied into
 * c', and so forth.
 *************************************************************************/
void shift_context_group_chain(struct group *g,
                struct vector *v)
{
        for (uint32_t i = 0; i < g->ctx_groups->num_elements; i++) {
                struct group *cg = g->ctx_groups->elements[i];
                shift_context_group_chain(cg, g->vector);
        }
        
        copy_vector(g->vector, v);
}

/**************************************************************************
 *************************************************************************/
void reset_context_groups(struct network *n)
{
        for (uint32_t i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];
                for (uint32_t j = 0; j < g->ctx_groups->num_elements; j++) {
                        struct group *cg = g->ctx_groups->elements[j];
                        reset_context_group_chain(cg);
                }
        }
}

/**************************************************************************
 *************************************************************************/
void reset_context_group_chain(struct group *g)
{
        for (uint32_t i = 0; i < g->ctx_groups->num_elements; i++)
                reset_context_group_chain(g->ctx_groups->elements[i]);
        fill_vector_with_value(g->vector, 0.5);
}

/**************************************************************************
 *************************************************************************/
void reset_recurrent_groups(struct network *n)
{
        for (uint32_t i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];
                if (g->recurrent) {
                        for (uint32_t j = 0; j < g->inc_projs->num_elements; j++) {
                                struct projection *p = g->inc_projs->elements[j];
                                if (p->to->recurrent)
                                        fill_vector_with_value(p->to->vector, 0.5);
                        }
                }
        }
}

/**************************************************************************
 *************************************************************************/
void reset_ffn_error_signals(struct network *n)
{
        for (uint32_t i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];
                zero_out_vector(g->error);
        }
}

/**************************************************************************
 *************************************************************************/
void reset_rnn_error_signals(struct network *n)
{
        struct rnn_unfolded_network *un = n->unfolded_net;

        for (uint32_t i = 0; i < un->stack_size; i++) {
                struct network *sn = un->stack[i];
                for (uint32_t j = 0; j < sn->groups->num_elements; j++) {
                        /* reset group's error vector */
                        struct group *g = sn->groups->elements[j];
                        zero_out_vector(g->error);

                        /* 
                         * Reset error vector of "terminal" group, if
                         * required.
                         */
                        if (i > 0 || !g->recurrent)
                                continue;

                        for (uint32_t x = 0; x < g->inc_projs->num_elements; x++) {
                                struct projection *p = g->inc_projs->elements[x];
                                if (p->to->recurrent)
                                        zero_out_vector(p->to->error);
                                
                        }
                }
        }
}

/**************************************************************************
 *************************************************************************/
struct projection *create_projection(
                struct group *to,
                struct matrix *weights,
                struct matrix *gradients,
                struct matrix *prev_gradients,
                struct matrix *prev_deltas,
                struct matrix *dynamic_pars,
                bool recurrent)
{
        struct projection *p;

        if (!(p = malloc(sizeof(struct projection))))
                goto error_out;
        memset(p, 0, sizeof(struct projection));

        p->to = to;
        p->weights = weights;
        p->gradients = gradients;
        p->prev_gradients = prev_gradients;
        p->prev_deltas = prev_deltas;
        p->dynamic_pars = dynamic_pars;
        p->recurrent = recurrent;

        return p;

error_out:
        perror("[create_projection()]");
        return NULL;
}

/**************************************************************************
 *************************************************************************/
void dispose_projection(struct projection *p)
{
        dispose_matrix(p->weights);
        dispose_matrix(p->gradients);
        dispose_matrix(p->prev_gradients);
        dispose_matrix(p->prev_deltas);
        dispose_matrix(p->dynamic_pars);

        free(p);
}

/**************************************************************************
 *************************************************************************/
void dispose_sets(struct array *ss)
{
        for (uint32_t i = 0; i < ss->num_elements; i++)
                dispose_set(ss->elements[i]);
}

/**************************************************************************
 *************************************************************************/
void randomize_weight_matrices(struct group *g, struct network *n)
{
        for (uint32_t i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *ip = g->inc_projs->elements[i];
                n->random_algorithm(ip->weights, n);
        }
        for (uint32_t i = 0; i < g->out_projs->num_elements; i++) {
                struct projection *op = g->out_projs->elements[i];
                randomize_weight_matrices(op->to, n);
        }
}

/**************************************************************************
 *************************************************************************/
void initialize_dynamic_pars(struct group *g, struct network *n)
{
        double v = 0.0;
        if (n->update_algorithm == bp_update_dbd) {
                v = n->learning_rate;
        } else {
                v = n->rp_init_update;
        }

        for (uint32_t i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *ip = g->inc_projs->elements[i];
                fill_matrix_with_value(ip->dynamic_pars, v);
        }

        for (uint32_t i = 0; i < g->out_projs->num_elements; i++) {
                struct projection *op = g->out_projs->elements[i];
                initialize_dynamic_pars(op->to, n);
        }
}

/**************************************************************************
 *************************************************************************/
void initialize_act_lookup_vectors(struct network *n)
{
        for (uint32_t i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];
                if (g == n->input || g->bias)
                        continue;
                if (g->act_fun->lookup)
                        dispose_vector(g->act_fun->lookup);
                g->act_fun->lookup = create_act_lookup_vector(g->act_fun->fun);
        }
}

/**************************************************************************
 *************************************************************************/
bool save_weight_matrices(struct network *n, char *fn)
{
        FILE *fd;
        if (!(fd = fopen(fn, "w")))
                goto error_out;

        if (n->type == TYPE_FFN)
                save_weight_matrix(n->input, fd);
        if (n->type == TYPE_SRN)
                save_weight_matrix(n->input, fd);
        if (n->type == TYPE_RNN)
                save_weight_matrix(n->unfolded_net->stack[0]->input, fd);

        fclose(fd);

        return true;

error_out:
        perror("[save_weight_matrices()]");
        return false;
}

/**************************************************************************
 *************************************************************************/
void save_weight_matrix(struct group *g, FILE *fd)
{
        /* write all incoming projections */
        for (uint32_t i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *ip = g->inc_projs->elements[i];
                
                fprintf(fd, "%s -> %s\n", ip->to->name, g->name);
                
                for (uint32_t r = 0; r < ip->weights->rows; r++) {
                        for (uint32_t c = 0; c < ip->weights->cols; c++) {
                                fprintf(fd, "%f", ip->weights->elements[r][c]);
                                if (c < ip->weights->cols - 1)
                                        fprintf(fd, " ");
                        }
                        fprintf(fd, "\n");
                }

                mprintf("Wrote weights for projection '%s -> %s'",
                                ip->to->name, g->name);
        }

        /* write all outgoing projections */
        for (uint32_t i = 0; i < g->out_projs->num_elements; i++) {
                struct projection *op = g->out_projs->elements[i];
                if (!op->recurrent)
                        save_weight_matrix(op->to, fd);
        }

        return;
}

/**************************************************************************
 *************************************************************************/
bool load_weight_matrices(struct network *n, char *fn)
{
        FILE *fd;
        if (!(fd = fopen(fn, "r")))
                goto error_out;

        struct network *np = NULL;
        if (n->type == TYPE_FFN)
                np = n;
        if (n->type == TYPE_SRN)
                np = n;
        if (n->type == TYPE_RNN)
                np = n->unfolded_net->stack[0];

        char buf[MAX_BUF_SIZE];
        while (fgets(buf, sizeof(buf), fd)) {
                char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];

                if (sscanf(buf, "%s -> %s", tmp1, tmp2) != 2)
                        continue;

                /* find the groups for the projection */
                struct group *g1, *g2;
                if ((g1 = find_array_element_by_name(np->groups, tmp1)) == NULL) {
                        eprintf("No such group '%s'", tmp1);
                        continue;
                }
                if ((g2 = find_array_element_by_name(np->groups, tmp2)) == NULL) {
                        eprintf("No such group '%s'", tmp2);
                        continue;
                }

                /* find the projection */
                struct projection *p = NULL;
                for (uint32_t i = 0; i < g1->out_projs->num_elements; i++) {
                        p = g1->out_projs->elements[i];
                        if (p->to == g2)
                                break;
                }

                /* read the matrix values */
                for (uint32_t r = 0; r < p->weights->rows; r++) {
                        if (!fgets(buf, sizeof(buf), fd))
                                goto error_out;
                        char *tokens = strtok(buf, " ");
                        for (uint32_t c = 0; c < p->weights->cols; c++) {
                                if (!sscanf(tokens, "%lf", &p->weights->elements[r][c]))
                                        goto error_out;
                                tokens = strtok(NULL, " ");
                        }
                }

                mprintf("Read weights for projection '%s -> %s'",
                                tmp1, tmp2);
        }

        fclose(fd);

        return true;

error_out:
        perror("[load_weight_matrices()]");
        return false;
}
