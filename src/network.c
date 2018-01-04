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
#include "train.h"
#include "verify.h"

struct network *create_network(char *name, enum network_type type)
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

        n->type   = type;
        n->groups = create_array(atype_groups);
        n->sets   = create_array(atype_sets);

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

void set_network_defaults(struct network *n)
{
        n->random_algorithm   = DEFAULT_RANDOM_ALGORITHM;
        n->random_mu          = DEFAULT_RANDOM_MU;
        n->random_sigma       = DEFAULT_RANDOM_SIGMA;
        n->random_min         = DEFAULT_RANDOM_MIN;
        n->random_max         = DEFAULT_RANDOM_MAX;
        n->learning_rate      = DEFAULT_LEARNING_RATE;
        n->lr_scale_factor    = DEFAULT_LR_SCALE_FACTOR;
        n->lr_scale_after     = DEFAULT_LR_SCALE_AFTER;
        n->momentum           = DEFAULT_MOMENTUM;
        n->mn_scale_factor    = DEFAULT_MN_SCALE_FACTOR;
        n->mn_scale_after     = DEFAULT_MN_SCALE_AFTER;
        n->weight_decay       = DEFAULT_WEIGHT_DECAY;
        n->wd_scale_factor    = DEFAULT_WD_SCALE_FACTOR;
        n->wd_scale_after     = DEFAULT_WD_SCALE_AFTER;
        n->target_radius      = DEFAULT_TARGET_RADIUS;
        n->zero_error_radius  = DEFAULT_ZERO_ERROR_RADIUS;
        n->error_threshold    = DEFAULT_ERROR_THRESHOLD;
        n->max_epochs         = DEFAULT_MAX_EPOCHS;
        n->report_after       = DEFAULT_REPORT_AFTER;
        n->rp_init_update     = DEFAULT_RP_INIT_UPDATE;
        n->rp_eta_plus        = DEFAULT_RP_ETA_PLUS;
        n->rp_eta_minus       = DEFAULT_RP_ETA_MINUS;
        n->dbd_rate_increment = DEFAULT_DBD_RATE_INCREMENT;
        n->dbd_rate_decrement = DEFAULT_DBD_RATE_DECREMENT;
        n->similarity_metric  = DEFAULT_SIMILARITY_METRIC;
}

void init_network(struct network *n)
{
        n->initialized = false;

        /*
         * Verify network sanity.
         */
        if (!verify_network(n)) {
                eprintf("Cannot initialize network--network cannot be verified\n");
                return;
        }

        /*
         * Randomize weights, and initialize dynamic learning parameters.
         */
        srand(n->random_seed);
        reset_network(n);

        /* 
         * If batch size is zero, set it to the number of items in the
         * active set.
         */
        if (n->batch_size == 0 && n->asp)
                n->batch_size = n->asp->items->num_elements;

        /* 
         * If a recurrent neural network will be trained with
         * backpropagation through time, unfold it.
         */
        if (n->learning_algorithm == train_network_with_bptt)
                n->unfolded_net = rnn_init_unfolded_network(n);

        n->initialized = true;
}

void reset_network(struct network *n)
{
        randomize_weight_matrices(n->input, n);
        initialize_dynamic_params(n->input, n);
}

void free_network(struct network *n)
{
        free(n->name);
        free(n->status);
        if (n->unfolded_net)
                rnn_free_unfolded_network(n->unfolded_net);
        free_groups(n->groups);
        free_array(n->groups);
        free_sets(n->sets);
        free_array(n->sets);
        free(n);
}

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

        /* activation function (default to linear) */
        if (!(g->act_fun = malloc(sizeof(struct act_fun))))
                goto error_out;
        memset(g->act_fun, 0, sizeof(struct act_fun));
        g->act_fun->fun   = act_fun_linear;
        g->act_fun->deriv = act_fun_linear_deriv;

        /* error function (do not set) */
        if (!(g->err_fun = malloc(sizeof(struct err_fun))))
                goto error_out;
        memset(g->err_fun, 0, sizeof(struct err_fun));

        g->vector     = create_vector(size);
        g->error      = create_vector(size);
        g->inc_projs  = create_array(atype_projs);
        g->out_projs  = create_array(atype_projs);
        g->ctx_groups = create_array(atype_groups);
        g->bias       = bias;
        g->recurrent  = recurrent;

        /* bias nodes have activation 1.0 */
        if(g->bias)
                g->vector->elements[0] = 1.0;

        return g;

error_out:
        perror("[create_group()]");
        return NULL;
}

struct group *attach_bias_group(struct network *n, struct group *g)
{
        char *bgn;
        size_t block_size = (strlen(g->name) + 6) * sizeof(char);
        if (!(bgn = malloc(block_size)))
                goto error_out;
        memset(bgn, 0, sizeof(block_size));
        sprintf(bgn, "%s_bias", g->name);
        struct group *bg = create_group(bgn, 1, true, false);
        free(bgn);

        bg->act_fun->fun   = g->act_fun->fun;
        bg->act_fun->deriv = g->act_fun->deriv;
        bg->err_fun->fun   = g->err_fun->fun;
        bg->err_fun->deriv = g->err_fun->deriv;

        /* add bias group to the network */
        add_to_array(n->groups, bg);

        struct matrix *weights = create_matrix(
                bg->vector->size, g->vector->size);
        struct matrix *gradients = create_matrix(
                bg->vector->size, g->vector->size);
        struct matrix *prev_gradients = create_matrix(
                bg->vector->size, g->vector->size);
        struct matrix *prev_deltas = create_matrix(
                bg->vector->size, g->vector->size);
        struct matrix *dynamic_params = create_matrix(
                bg->vector->size, g->vector->size);

        /* add incoming and outgoing projections */
        struct projection *op = create_projection(g, weights,
                gradients, prev_gradients, prev_deltas, dynamic_params);
        op->recurrent = false;
        add_to_array(bg->out_projs, op);
        struct projection *ip = create_projection(bg, weights,
                gradients, prev_gradients, prev_deltas, dynamic_params);
        ip->recurrent = false;
        add_to_array(g->inc_projs, ip);

        return bg;

error_out:
        perror("[attach_bias_group()]");
        return NULL;
}

void free_group(struct group *g)
{
        free(g->name);
        free_vector(g->vector);
        free_vector(g->error);
        if (g->act_fun->lookup)
                free_vector(g->act_fun->lookup);
        free(g->act_fun);
        free(g->err_fun);
        for (uint32_t j = 0; j < g->inc_projs->num_elements; j++)
                free_projection(g->inc_projs->elements[j]);
        free_array(g->inc_projs);
        for (uint32_t j = 0; j < g->out_projs->num_elements; j++)
                free(g->out_projs->elements[j]);
        free_array(g->out_projs);
        free_array(g->ctx_groups);
        free(g);
}

void free_groups(struct array *gs)
{
        for (uint32_t i = 0; i < gs->num_elements; i++)
                free_group(gs->elements[i]);
}

void shift_context_groups(struct network *n)
{
        for (uint32_t i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];
                for (uint32_t j = 0; j < g->ctx_groups->num_elements; j++)
                        shift_context_group_chain(
                                g->ctx_groups->elements[j],
                                g->vector);
        }
}

/*
 * Shifts a context group chain. If group g has a context group c, then the
 * activity vector of g is copied into that of c. However, if c has itself a
 * context group c', then the activity pattern of c is first copied into c',
 * and so forth.
 */
void shift_context_group_chain(struct group *g,
                struct vector *v)
{
        for (uint32_t i = 0; i < g->ctx_groups->num_elements; i++)
                shift_context_group_chain(
                        g->ctx_groups->elements[i],
                        g->vector);
        copy_vector(g->vector, v);
}

/*
 * If the stack pointer of an unfolded net is not yet pointing to stack/n,
 * increment the pointer. Otherwise shift the stack such that stack/n become
 * useable for the next tick.
 */
void shift_pointer_or_stack(struct network *n)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        if (un->sp < un->stack_size - 1)
                un->sp++;
        else
                rnn_shift_stack(un);
}

void reset_context_groups(struct network *n)
{
        for (uint32_t i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];
                for (uint32_t j = 0; j < g->ctx_groups->num_elements; j++)
                        reset_context_group_chain(g->ctx_groups->elements[j]);
        }
}

/*
 * TODO: Condition value on the activation function.
 */
void reset_context_group_chain(struct group *g)
{
        for (uint32_t i = 0; i < g->ctx_groups->num_elements; i++)
                reset_context_group_chain(g->ctx_groups->elements[i]);
        fill_vector_with_value(g->vector, 0.5);
}

/*
 * TODO: Condition value on the activation function.
 */
void reset_recurrent_groups(struct network *n)
{
        for (uint32_t i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];
                if (!g->recurrent)
                        continue;
                for (uint32_t j = 0; j < g->inc_projs->num_elements; j++) {
                        struct projection *p = g->inc_projs->elements[j];
                        if (!p->to->recurrent)
                                continue;
                        fill_vector_with_value(p->to->vector, 0.5);
                }
        }
}

void reset_ffn_error_signals(struct network *n)
{
        for (uint32_t i = 0; i < n->groups->num_elements; i++)
                zero_out_vector(((struct group *)
                        n->groups->elements[i])->error);
}

void reset_rnn_error_signals(struct network *n)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        for (uint32_t i = 0; i < un->stack_size; i++) {
                struct network *sn = un->stack[i];
                for (uint32_t j = 0; j < sn->groups->num_elements; j++) {
                        /* reset group error */
                        struct group *g = sn->groups->elements[j];
                        zero_out_vector(g->error);
                        /* reset error vector of "terminal" group */
                        if (i > 0 || !g->recurrent) continue;
                        for (uint32_t x = 0; x < g->inc_projs->num_elements; x++) {
                                struct projection *p = g->inc_projs->elements[x];
                                if (p->to->recurrent)
                                        zero_out_vector(p->to->error);
                        }
                }
        }
}

struct projection *create_projection(
        struct group *to,
        struct matrix *weights,
        struct matrix *gradients,
        struct matrix *prev_gradients,
        struct matrix *prev_deltas,
        struct matrix *dynamic_params)
{
        struct projection *p;
        if (!(p = malloc(sizeof(struct projection))))
                goto error_out;
        memset(p, 0, sizeof(struct projection));

        p->to             = to;
        p->weights        = weights;
        p->gradients      = gradients;
        p->prev_gradients = prev_gradients;
        p->prev_deltas    = prev_deltas;
        p->dynamic_params = dynamic_params;

        return p;

error_out:
        perror("[create_projection()]");
        return NULL;
}

void free_projection(struct projection *p)
{
        free_matrix(p->weights);
        free_matrix(p->gradients);
        free_matrix(p->prev_gradients);
        free_matrix(p->prev_deltas);
        free_matrix(p->dynamic_params);
        free(p);
}

void free_sets(struct array *sets)
{
        for (uint32_t i = 0; i < sets->num_elements; i++)
                free_set(sets->elements[i]);
}

void randomize_weight_matrices(struct group *g, struct network *n)
{
        /* incoming projections */
        for (uint32_t i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *ip = g->inc_projs->elements[i];
                if (ip->frozen) continue;
                n->random_algorithm(ip->weights, n);
        }
        /* outgoing projections */
        for (uint32_t i = 0; i < g->out_projs->num_elements; i++) {
                struct projection *op = g->out_projs->elements[i];
                randomize_weight_matrices(op->to, n);
        }
}

void initialize_dynamic_params(struct group *g, struct network *n)
{
        double v = 0.0;
        if (n->update_algorithm == bp_update_rprop)
                v = n->rp_init_update;
        if (n->update_algorithm == bp_update_dbd)
                v = n->learning_rate;

        /* incoming projections */
        for (uint32_t i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *ip = g->inc_projs->elements[i];
                fill_matrix_with_value(ip->dynamic_params, v);
        }
        /* outgoing projections */
        for (uint32_t i = 0; i < g->out_projs->num_elements; i++) {
                struct projection *op = g->out_projs->elements[i];
                initialize_dynamic_params(op->to, n);
        }
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Save and load weights. Currently, the format for weights files is:

        from_group -> to_group
        # # # # # # # # #
        # # # # # # # # #
        # # # # # # # # #
        [...]

        from_group -> to_group
        # # # #
        # # # #
        [...]

where each line of '#'s denotes the weights of one unit of the 'from_group'
to each of the units of the 'to_group', and where each '#' is a floating
point weight.

TODO: Adopt a less spartan file format.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool save_weight_matrices(struct network *n, char *filename)
{
        FILE *fd;
        if (!(fd = fopen(filename, "w")))
                goto error_out;

        switch (n->type) {
        case ntype_ffn: /* fall through */
        case ntype_srn:
                save_weight_matrix(n->input, fd);
                break;
        case ntype_rnn:
                save_weight_matrix(n->unfolded_net->stack[0]->input, fd);
                break;
        }

        fclose(fd);

        return true;

error_out:
        perror("[save_weight_matrices()]");
        return false;
}

void save_weight_matrix(struct group *g, FILE *fd)
{
        /* incoming projections */
        for (uint32_t i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *ip = g->inc_projs->elements[i];
                fprintf(fd, "%s -> %s\n",
                        ip->to->name,
                        g->name);
                for (uint32_t r = 0; r < ip->weights->rows; r++) {
                        for (uint32_t c = 0; c < ip->weights->cols; c++) {
                                fprintf(fd, "%f",
                                        ip->weights->elements[r][c]);
                                if (c < ip->weights->cols - 1)
                                        fprintf(fd, " ");
                        }
                        fprintf(fd, "\n");
                }
                mprintf("... wrote weights for projection '%s -> %s'\n",
                        ip->to->name, g->name);
        }
        /* outgoing projections */
        for (uint32_t i = 0; i < g->out_projs->num_elements; i++) {
                struct projection *op = g->out_projs->elements[i];
                if (op->recurrent)
                        continue;
                save_weight_matrix(op->to, fd);
        }

        return;
}

bool load_weight_matrices(struct network *n, char *filename)
{
        FILE *fd;
        if (!(fd = fopen(filename, "r")))
                goto error_out;

        struct network *np = NULL;
        switch (n->type) {
        case ntype_ffn: /* fall through */
        case ntype_srn:
                np = n;
                break;
        case ntype_rnn:
                np = n->unfolded_net->stack[0];
                break;
        }

        char buf[MAX_BUF_SIZE];
        while (fgets(buf, sizeof(buf), fd)) {
                char arg1[MAX_ARG_SIZE]; /* 'from' group name */
                char arg2[MAX_ARG_SIZE]; /* 'to' group name */

                if (sscanf(buf, "%s -> %s", arg1, arg2) != 2)
                        continue;

                /* find 'from' group */
                struct group *fg;
                if ((fg = find_array_element_by_name(np->groups, arg1)) == NULL) {
                        eprintf("Cannot load weights - no such group '%s'\n", arg1);
                        continue;
                }

                /* find 'to' group */
                struct group *tg;
                if ((tg = find_array_element_by_name(np->groups, arg2)) == NULL) {
                        eprintf("Cannot load weights - no such group '%s'\n", arg2);
                        continue;
                }

                /* find outgoing 'from->to' projection */
                struct projection *fg_to_tg = NULL;
                for (uint32_t i = 0; i < fg->out_projs->num_elements; i++) {
                        if (((struct projection *)fg->out_projs->elements[i])->to == tg) {
                                fg_to_tg = (struct projection *)fg->out_projs->elements[i];
                                break;
                        }
                }

                /* projection should exist */
                if (!fg_to_tg) {
                        eprintf("Cannot load weights - no projection between groups '%s' and '%s'\n",
                                arg1, arg2);
                        return true;
                }

                /* read the matrix values */
                for (uint32_t r = 0; r < fg_to_tg->weights->rows; r++) {
                        if (!fgets(buf, sizeof(buf), fd))
                                goto error_out;
                        char *tokens = strtok(buf, " ");
                        for (uint32_t c = 0; c < fg_to_tg->weights->cols; c++) {
                                if (!sscanf(tokens, "%lf", &fg_to_tg->weights->elements[r][c]))
                                        goto error_out;
                                tokens = strtok(NULL, " ");
                        }
                }

                mprintf("... read weights for projection '%s -> %s'\n",
                        arg1, arg2);
        }

        fclose(fd);

        return true;

error_out:
        perror("[load_weight_matrices()]");
        return false;
}
