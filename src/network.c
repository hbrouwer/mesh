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
#include "error.h"
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
        n->reset_contexts     = DEFAULT_RESET_CONTEXTS;
        n->init_context_units = DEFAULT_INIT_CONTEXT_UNITS;
        n->random_algorithm   = DEFAULT_RANDOM_ALGORITHM;
        n->random_mu          = DEFAULT_RANDOM_MU;
        n->random_sigma       = DEFAULT_RANDOM_SIGMA;
        n->random_min         = DEFAULT_RANDOM_MIN;
        n->random_max         = DEFAULT_RANDOM_MAX;
        n->learning_algorithm = DEFAULT_LEARNING_ALGORITHM;
        n->update_algorithm   = DEFAULT_UPDATE_ALGORITHM;
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
        reset_context_groups(n);
        reset_recurrent_groups(n);
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

void inspect_network(struct network *n)
{
                /*****************
                 **** general ****
                 *****************/

        /* name */
        cprintf("| Name: \t\t\t %s\n", n->name);
        cprintf("| Type: \t\t\t ");
        if (n->type == ntype_ffn)
                cprintf("ffn");
        if (n->type == ntype_srn)
                cprintf("ffn");
        if (n->type == ntype_rnn)
                cprintf("ffn");
        cprintf("\n");
        cprintf("| Initialized: \t\t\t ");
        n->initialized ? cprintf("true\n") : cprintf("false\n");
        cprintf("| Unfolded: \t\t\t ");
        n->unfolded_net ? cprintf("true\n") : cprintf("false\n");
        cprintf("| Groups: \t\t\t ");
        for (uint32_t i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];
                if (i > 0) cprintf(", ");
                cprintf("%s (%d)", g->name, g->vector->size);
        }
        cprintf("\n");
        cprintf("| Input: \t\t\t ");
        n->input != NULL
                ? cprintf("%s (%d)\n",
                        n->input->name, n->input->vector->size)
                : cprintf("\n");
        cprintf("| Output: \t\t\t ");
        n->output != NULL
                ? cprintf("%s (%d)\n",
                        n->output->name, n->output->vector->size)
                : cprintf("\n");
        cprintf("| Sets: \t\t\t ");
        for (uint32_t i = 0; i < n->sets->num_elements; i++) {
                struct set *set = n->sets->elements[i];
                if (i > 0) cprintf(", ");
                cprintf("%s (%d)", set->name, set->items->num_elements);
        }
        cprintf("\n");

                /******************
                 **** contexts ****
                 ******************/

        cprintf("|\n");
        cprintf("| Reset contexts: \t\t ");
        n->reset_contexts ? cprintf("true\n") : cprintf("false\n");
        cprintf("| Init context units: \t\t %f\n", n->init_context_units);

                /******************
                 **** training ****
                 ******************/

        cprintf("|\n");
        cprintf("| Learning algorithm: \t\t ");
        if (n->learning_algorithm == train_network_with_bp)
                cprintf("bp");
        if (n->learning_algorithm == train_network_with_bptt) {
                cprintf("bptt");
        }
        cprintf("\n");
        cprintf("| Back ticks: \t\t\t %d\n", n->back_ticks);
        cprintf("| Update algorithm: \t\t ");
        if (n->update_algorithm == bp_update_sd
                && n->sd_type == SD_DEFAULT)
                cprintf("steepest");
        if (n->update_algorithm == bp_update_sd
                && n->sd_type == SD_BOUNDED)
                cprintf("bounded");
        if (n->update_algorithm == bp_update_rprop
                && n->rp_type == RPROP_PLUS)
                cprintf("rprop+");
        if (n->update_algorithm == bp_update_rprop
                && n->rp_type == RPROP_MINUS)
                cprintf("rprop-");
        if (n->update_algorithm == bp_update_rprop
                && n->rp_type == IRPROP_PLUS)
                cprintf("irprop+");
        if (n->update_algorithm == bp_update_rprop
                && n->rp_type == IRPROP_MINUS)
                cprintf("irprop+");
        if (n->update_algorithm == bp_update_qprop)
                cprintf("qprop");
        if (n->update_algorithm == bp_update_dbd)
                cprintf("dbd");
        cprintf("\n");
        cprintf("|\n");
        cprintf("| Learning rate (LR): \t\t %f\n",      n->learning_rate);
        cprintf("| LR scale factor: \t\t %f\n",         n->lr_scale_factor);
        cprintf("| LR scale after (%%epochs): \t %f\n", n->lr_scale_after);
        cprintf("|\n");
        cprintf("| Momentum (MN): \t\t %f\n",           n->momentum);
        cprintf("| MN scale factor: \t\t %f\n",         n->mn_scale_factor);
        cprintf("| MN scale after (%%epochs): \t %f\n", n->mn_scale_after);
        cprintf("|\n");
        cprintf("| Rprop init update: \t\t %f\n",       n->rp_init_update);
        cprintf("| Rprop Eta-: \t\t\t %f\n",            n->rp_eta_minus);
        cprintf("| Rprop Eta+: \t\t\t %f\n",            n->rp_eta_plus);
        cprintf("|\n");
        cprintf("| DBD rate increment: \t\t %f\n",      n->rp_init_update);
        cprintf("| DBD rate decrement: \t\t %f\n",      n->rp_eta_minus);
        cprintf("|\n");
        cprintf("| Weight decay (WD): \t\t %f\n",       n->weight_decay);
        cprintf("| WD scale factor: \t\t %f\n",         n->wd_scale_factor);
        cprintf("| WD scale after (%%epochs): \t %f\n", n->wd_scale_after);
        cprintf("|\n");
        cprintf("| Target radius: \t\t %f\n",           n->target_radius);
        cprintf("| Zero error radius: \t\t %f\n",       n->zero_error_radius);
        cprintf("| Error threshold: \t\t %f\n",         n->error_threshold);
        cprintf("|\n");
        cprintf("| Training order: \t\t ");
        if (n->training_order == train_ordered)
                cprintf("ordered");
        if (n->training_order == train_permuted)
                cprintf("permuted");
        if (n->training_order == train_randomized)
                cprintf("randomized");
        cprintf("\n");
        cprintf("| Batch size: \t\t\t %d\n",            n->batch_size);
        cprintf("| Maximum #epochs: \t\t %d\n",         n->max_epochs);
        cprintf("| Report after #epochs \t\t %d\n",     n->report_after);
        cprintf("|\n");
        cprintf("| Multi-stage input: \t\t ");
        n->ms_input
                ? cprintf("%s (%d)\n",
                        n->ms_input->name, n->ms_input->vector->size)
                : cprintf("\n");
        cprintf("| Multi-stage set: \t\t ");
        n->ms_set
                ? cprintf("%s (%d)\n",
                        n->ms_set->name, n->ms_set->items->num_elements)
                : cprintf("\n");

                /***********************
                 **** randomization ****
                 ***********************/

        cprintf("|\n");
        cprintf("| Random algorithm: \t\t ");
        if (n->random_algorithm == randomize_gaussian)
                cprintf("gaussian");
        if (n->random_algorithm == randomize_range)
                cprintf("range");
        if (n->random_algorithm == randomize_nguyen_widrow)
                cprintf("nguyen_widrow");
        if (n->random_algorithm == randomize_fan_in)
                cprintf("fan_in");
        if (n->random_algorithm == randomize_binary)
                cprintf("binary");
        cprintf("\n");
        cprintf("| Random Seed: \t\t\t %d\n", n->random_seed);
        cprintf("| Random Mu: \t\t\t %f\n",   n->random_mu);
        cprintf("| Random Sigma: \t\t %f\n",  n->random_sigma);
        cprintf("| Random Min: \t\t\t %f\n",  n->random_min);
        cprintf("| Random Max: \t\t\t %f\n",  n->random_max);

                /***************
                 **** other ****
                 ***************/

        cprintf("|\n");
        cprintf("| Similarity metric: \t\t ");
        if (n->similarity_metric == inner_product)
                cprintf("inner_product");
        if (n->similarity_metric == harmonic_mean)
                cprintf("harmonic_mean");
        if (n->similarity_metric == cosine)
                cprintf("cosine");
        if (n->similarity_metric == tanimoto)
                cprintf("tanimoto");
        if (n->similarity_metric == dice)
                cprintf("dice");
        if (n->similarity_metric == pearson_correlation)
                cprintf("pearson_correlation");
        cprintf("\n");

                /****************
                 **** status ****
                 ****************/

        /*
        cprintf("|\n");
        cprintf("| Epoch: \t\t\t %d\n",
                n->status->epoch);
        cprintf("| Error: \t\t\t %f\n",
                n->status->error);
        cprintf("| Previous error: \t\t %f\n",
                n->status->prev_error);
        cprintf("| Gradient linearity: \t\t %f\n",
                n->status->gradient_linearity);
        cprintf("| Last deltas length: \t\t %f\n",
                n->status->last_deltas_length);
        cprintf("| Gradient length: \t\t %f\n",
                n->status->gradients_length);
        */
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

        g->vector        = create_vector(size);
        g->error         = create_vector(size);
        g->inc_projs     = create_array(atype_projs);
        g->out_projs     = create_array(atype_projs);
        g->ctx_groups    = create_array(atype_groups);
        g->bias          = bias;
        g->recurrent     = recurrent;

        g->relu_alpha    = DEFAULT_RELU_ALPHA;
        g->logistic_fsc  = DEFAULT_LOGISTIC_FSC;
        g->logistic_gain = DEFAULT_LOGISTIC_GAIN;

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
        if (find_array_element_by_name(n->groups, bgn))
                return NULL;
        struct group *bg = create_group(bgn, 1, true, false);
        free(bgn);

        bg->act_fun->fun   = g->act_fun->fun;
        bg->act_fun->deriv = g->act_fun->deriv;
        bg->err_fun->fun   = g->err_fun->fun;
        bg->err_fun->deriv = g->err_fun->deriv;

        /* add bias group to the network */
        add_group(n, bg);

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
        add_projection(bg->out_projs, op);
        struct projection *ip = create_projection(bg, weights,
                gradients, prev_gradients, prev_deltas, dynamic_params);
        ip->recurrent = false;
        add_projection(g->inc_projs, ip);

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
        free(g->act_fun);
        free(g->err_fun);
        for (uint32_t i = 0; i < g->inc_projs->num_elements; i++)
                free_projection(g->inc_projs->elements[i]);
        free_array(g->inc_projs);
        for (uint32_t i = 0; i < g->out_projs->num_elements; i++)
                free(g->out_projs->elements[i]);
        free_array(g->out_projs);
        free_array(g->ctx_groups);
        free(g);
}

void free_groups(struct array *gs)
{
        for (uint32_t i = 0; i < gs->num_elements; i++)
                free_group(gs->elements[i]);
}

void add_group(struct network *n, struct group *g)
{
        add_to_array(n->groups, g);
}

void remove_group(struct network *n, struct group *g)
{
        /* remove outgoing projections from a group g' to group g */
        for (uint32_t i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *p = g->inc_projs->elements[i];
                struct group *fg = p->to;
                for (uint32_t j = 0; j < fg->out_projs->num_elements; j++) {
                        struct projection *op = fg->out_projs->elements[j];
                        if (op->to == g) {
                                remove_projection(fg->out_projs, op);
                                break;
                        }
                }
        }

        /* remove incoming projections to group g from a group g' */
        for (uint32_t i = 0; i < g->out_projs->num_elements; i++) {
                struct projection *p = g->out_projs->elements[i];
                struct group *tg = p->to;
                for (uint32_t j = 0; j < tg->inc_projs->num_elements; j++) {
                        struct projection *ip = tg->inc_projs->elements[j];
                        if (ip->to == g) {
                                remove_projection(tg->inc_projs, ip);
                                break;
                        }
                }
        }

        /* remove Elman projections from a group g' to group g */
        for (uint32_t i = 0; i < n->groups->num_elements; i++) {
                struct group *fg = n->groups->elements[i];
                for (uint32_t j = 0; j < fg->ctx_groups->num_elements; j++) {
                        if (fg->ctx_groups->elements[j] == g) {
                                remove_elman_projection(fg, g);
                                break;
                        }
                }
        }

        /* remove group */
        remove_from_array(n->groups, g);
        free_group(g);
}

void print_groups(struct network *n)
{
        if (n->groups->num_elements == 0) {
                cprintf("(no groups)\n");
                return;
        }
        for (uint32_t i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];

                /* name and size */
                cprintf("* %d: %s :: %d", i + 1, g->name, g->vector->size);

                /* activation function */
                if (g->act_fun->fun == act_fun_logistic)
                        cprintf(" :: logistic (fsc = %f; gain = %f)",
                                g->logistic_fsc, g->logistic_gain);
                if (g->act_fun->fun == act_fun_bipolar_sigmoid)
                        cprintf(" :: bipolar_sigmoid");
                if (g->act_fun->fun == act_fun_softmax)
                        cprintf(" :: softmax");
                if (g->act_fun->fun == act_fun_tanh)
                        cprintf(" :: tanh");
                if (g->act_fun->fun == act_fun_linear)
                        cprintf(" :: linear");
                if (g->act_fun->fun == act_fun_softplus)
                        cprintf(" :: softplus");
                if (g->act_fun->fun == act_fun_relu)
                        cprintf(" :: relu");
                if (g->act_fun->fun == act_fun_binary_relu)
                        cprintf(" :: binary_relu");
                if (g->act_fun->fun == act_fun_leaky_relu)
                        cprintf(" :: leaky_relu (alpha = %f)",
                                g->relu_alpha);
                if (g->act_fun->fun == act_fun_elu)
                        cprintf(" :: elu (alpha = %f)",
                                g->relu_alpha);

                /* error function */
                if (g->err_fun->fun == err_fun_sum_of_squares)
                        cprintf(" :: sum_of_squares");
                if (g->err_fun->fun == err_fun_cross_entropy)
                        cprintf(" :: cross_entropy");
                if (g->err_fun->fun == err_fun_divergence)
                        cprintf(" :: divergence");

                /* input/output group */
                if (g == n->input)
                        cprintf(" :: input group\n");
                else if (g == n->output)
                        cprintf(" :: output group\n");
                else
                        cprintf("\n");
        }       
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

void reset_stack_pointer(struct network *n)
{
        /*
         * If context groups should not be reset, we want to keep the cycle
         * running, so we do not reset the stack pointer.
         * 
         * TODO: Validate this logic!
         */
        if (n->initialized && !n->reset_contexts)
                return;
        n->unfolded_net->sp = 0;
}

void reset_context_groups(struct network *n)
{
        /*
         * If context groups should not be reset, shift the context groups.
         */
        if (n->initialized && !n->reset_contexts) {
                shift_context_groups(n);
                return;
        }
        for (uint32_t i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];
                for (uint32_t j = 0; j < g->ctx_groups->num_elements; j++)
                        reset_context_group_chain(n, g->ctx_groups->elements[j]);
        }
}

void reset_context_group_chain(struct network *n, struct group *g)
{
        for (uint32_t i = 0; i < g->ctx_groups->num_elements; i++)
                reset_context_group_chain(n, g->ctx_groups->elements[i]);
        fill_vector_with_value(g->vector, n->init_context_units);
}

void reset_recurrent_groups(struct network *n)
{
        /*
         * If context groups should not be reset, shift the pointer or the
         * stack.
         *
         * TODO: Validate this logic!
         */
        if (n->initialized && !n->reset_contexts) {
                shift_pointer_or_stack(n);
                return;
        }
        for (uint32_t i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];
                if (!g->recurrent)
                        continue;
                for (uint32_t j = 0; j < g->inc_projs->num_elements; j++) {
                        struct projection *p = g->inc_projs->elements[j];
                        if (!p->to->recurrent)
                                continue;
                        fill_vector_with_value(p->to->vector,
                                n->init_context_units);
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

void add_projection(struct array *projs, struct projection *p)
{
        add_to_array(projs, p);
}

void add_bidirectional_projection(struct group *fg, struct group *tg)
{
        if (fg == tg) {
                fg->recurrent = true;
                return;
        }
        /* weight matrix */
        struct matrix *weights = create_matrix(
                fg->vector->size, tg->vector->size);
        /* gradients matrix */
        struct matrix *gradients = create_matrix(
                fg->vector->size, tg->vector->size);
        /* previous gradients matrix */
        struct matrix *prev_gradients = create_matrix(
                fg->vector->size, tg->vector->size);
        /* previous weight deltas matrix */
        struct matrix *prev_deltas = create_matrix(
                fg->vector->size, tg->vector->size);
        /* dynamic learning parameters matrix */
        struct matrix *dynamic_params = create_matrix(
                fg->vector->size, tg->vector->size);
        /* add projections */
        struct projection *op = create_projection(tg, weights,
                gradients, prev_gradients, prev_deltas, dynamic_params);
        struct projection *ip = create_projection(fg, weights,
                gradients, prev_gradients, prev_deltas, dynamic_params);
        add_projection(fg->out_projs, op);
        add_projection(tg->inc_projs, ip);
}

void remove_projection(struct array *projs, struct projection *p)
{
        remove_from_array(projs, p);
}

void remove_bidirectional_projection(
        struct group *fg,
        struct projection *fg_to_tg,
        struct group *tg,
        struct projection *tg_to_fg)
{
        remove_projection(fg->out_projs, fg_to_tg);
        remove_projection(tg->inc_projs, tg_to_fg);
        free_projection(fg_to_tg);
        free(tg_to_fg);
}

struct projection *find_projection(struct array *projs, struct group *g)
{
        for (uint32_t i = 0; i < projs->num_elements; i++) {
                struct projection *p = projs->elements[i];
                if (p->to == g)
                        return p;
        }
        return NULL;
}

void add_elman_projection(struct group *fg, struct group *tg)
{
        add_to_array(fg->ctx_groups, tg);
}

void remove_elman_projection(struct group *fg, struct group *tg)
{
        remove_from_array(fg->ctx_groups, tg);
}

bool find_elman_projection(struct group *fg, struct group *tg)
{
        for (uint32_t i = 0; i < fg->ctx_groups->num_elements; i++)
                if (fg->ctx_groups->elements[i] == tg)
                        return true;
        return false;
}

void print_projections(struct network *n)
{
        if (n->groups->num_elements == 0) {
                cprintf("(no groups)\n");
                return;
        }
        /*
         * List incoming, recurrent, and outgoing projections for each
         * group.
         */
        for (uint32_t i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];

                /* incoming projections */
                cprintf("* %d: ", i + 1);
                for (uint32_t j = 0; j < g->inc_projs->num_elements; j++) {
                        struct projection *p = g->inc_projs->elements[j];
                        struct group *fg = p->to;
                        if (j > 0 && j < g->inc_projs->num_elements)
                                cprintf(", ");
                        cprintf("%s (%dx%d)", fg->name,
                                p->weights->rows, p->weights->cols);
                }
                
                /* recurrent incoming projection */
                if (g->recurrent) {
                        if (g->inc_projs->num_elements > 0)
                                cprintf(", ");
                        cprintf("%s (%d x %d)", g->name,
                                g->vector->size, g->vector->size);
                }

                /* current group */
                if (g->recurrent || g->inc_projs->num_elements > 0)
                        cprintf(" -> ", g->name);
                cprintf("[%s]", g->name);
                if (g->recurrent || g->out_projs->num_elements > 0)
                        cprintf(" -> ", g->name);

                /* outgoing projections */
                for (uint32_t j = 0; j < g->out_projs->num_elements; j++) {
                        struct projection *p = g->out_projs->elements[j];
                        struct group *tg = p->to;
                        if (j > 0 && j < g->out_projs->num_elements)
                                cprintf(", ");
                        cprintf("%s (%dx%d)", tg->name,
                                p->weights->rows, p->weights->cols);
                }

                /* recurrent outgoing projection */
                if (g->recurrent) {
                        if (g->out_projs->num_elements > 0)
                                cprintf(", ");
                        cprintf("%s", g->name);
                }

                cprintf("\n");

                /* context (Elman) groups */
                if (g->ctx_groups->num_elements > 0) {
                        cprintf("* %d: [%s] => ", i + 1, g->name);
                        for (uint32_t j = 0;
                                j < g->ctx_groups->num_elements;
                                j++) {
                                struct group *cg = g->ctx_groups->elements[j];
                                if (j > 0 && j < g->out_projs->num_elements)
                                        cprintf(", ");
                                cprintf("%s (copy)", cg->name);
                        }
                        cprintf("\n");
                }
        }     
}


void freeze_projection(struct projection *fg_to_tg,
        struct projection *tg_to_fg)
{
        fg_to_tg->frozen = true;
        tg_to_fg->frozen = true;
}

void unfreeze_projection(struct projection *fg_to_tg,
        struct projection *tg_to_fg)
{
        fg_to_tg->frozen = false;
        tg_to_fg->frozen = false;
}

void free_sets(struct array *sets)
{
        for (uint32_t i = 0; i < sets->num_elements; i++)
                free_set(sets->elements[i]);
}

void add_set(struct network *n, struct set *set)
{
        add_to_array(n->sets, set);
        n->asp = set;
}

void remove_set(struct network *n, struct set *set)
{
        /*
         * If the set to be removed is the active set, try finding another
         * active set.
         */
        if (set == n->asp) {
                n->asp = NULL;
                for (uint32_t i = 0; i < n->sets->num_elements; i++)
                        if (n->sets->elements[i] != NULL
                                && n->sets->elements[i] != set)
                                n->asp = n->sets->elements[i];
        }

        /* remove set */
        remove_from_array(n->sets, set);
        free_set(set);
}

void print_sets(struct network *n)
{
        if (n->sets->num_elements == 0) {
                cprintf("(no sets)\n");
                return;
        }
        for (uint32_t i = 0; i < n->sets->num_elements; i++) {
                struct set *set = n->sets->elements[i];
                cprintf("* %d: %s (%d)", i + 1, set->name, set->items->num_elements);
                if (set == n->asp)
                        cprintf(" :: active set\n");
                else
                        cprintf("\n");
        }     
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
Save and load weights. The format for weights files is:

        Projection from_group to_group
        [Dimensions F T]
        # # # # # # # # #
        # # # # # # # # #
        # # # # # # # # #
        [...]

        Projection from_group to_group
        [Dimensions F T]
        # # # #
        # # # #
        [...]

where each line of '#'s denotes the weights of one unit of the 'from_group'
to each of the units of the 'to_group', and where each '#' is a floating
point weight. The Dimensions F G statement is an optional specification of
the size of the 'from_group' and the 'to_group', respectively.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool save_weight_matrices(struct network *n, char *filename)
{
        FILE *fd;
        if (!(fd = fopen(filename, "w")))
                goto error_file;

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

error_file:
        eprintf("Cannot save weights - unable to write file '%s'\n", filename);
        return false;
}

void save_weight_matrix(struct group *g, FILE *fd)
{
        /* incoming projections */
        for (uint32_t i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *ip = g->inc_projs->elements[i];
                fprintf(fd, "Projection %s %s\n", ip->to->name, g->name);
                fprintf(fd, "Dimensions %d %d\n",
                        ip->to->vector->size, g->vector->size);  
                for (uint32_t r = 0; r < ip->weights->rows; r++) {
                        for (uint32_t c = 0; c < ip->weights->cols; c++) {
                                fprintf(fd, "%f", ip->weights->elements[r][c]);
                                if (c < ip->weights->cols - 1)
                                        fprintf(fd, " ");
                        }
                        fprintf(fd, "\n");
                }
                fprintf(fd, "\n");
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
                goto error_file;

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
                buf[strlen(buf) - 1] = '\0';
                /* comment or blank line */
                switch (buf[0]) {
                case '%':       /* verbose comment */
                        cprintf("\x1b[1m\x1b[36m%s\x1b[0m\n", buf);
                        continue;
                case '#':       /* silent comment */
                case '\0':      /* blank line */
                        continue;
                }
                /* 
                 * Read projection specification, which we are expecting at
                 * this point. If it is not there, we ran into a
                 * dimensionality mismatch problem.
                 */
                char arg1[MAX_ARG_SIZE]; /* 'from' group name */
                char arg2[MAX_ARG_SIZE]; /* 'to' group name */
                if (sscanf(buf, "Projection %s %s", arg1, arg2) != 2)
                        /*
                         * XXX: Legacy format ...
                         */
                        if (sscanf(buf, "%s -> %s", arg1, arg2) != 2)
                                /* error: expected no more rows */
                                goto error_projecting_group;
                /* find 'from' group */
                struct group *fg;
                if ((fg = find_array_element_by_name(np->groups, arg1)) == NULL) {
                        eprintf("Cannot load weights - no such group '%s'\n", arg1);
                        return false;
                }
                /* find 'to' group */
                struct group *tg;
                if ((tg = find_array_element_by_name(np->groups, arg2)) == NULL) {
                        eprintf("Cannot load weights - no such group '%s'\n", arg2);
                        return false;
                }
                /* projection should exist */
                struct projection *fg_to_tg = find_projection(fg->out_projs, tg);
                if (!fg_to_tg) {
                        eprintf("Cannot load weights - no projection between groups '%s' and '%s'\n", arg1, arg2);
                        return false;
                }
                /*
                 * Read the next line, which may be an optional dimensions
                 * specification, or the first row of weights.
                */
                if (!fgets(buf, sizeof(buf), fd))
                        goto error_format;
                uint32_t arg3; /* 'from' group size */
                uint32_t arg4; /* 'to' group size */
                /* 
                 * Check for dimension specification, and in case it is
                 * present, verify the dimensionality.
                 */
                if (sscanf(buf, "Dimensions %d %d", &arg3, &arg4) == 2) {
                        /* error: projecting group of incorrect size */
                        if (fg->vector->size != arg3) 
                                goto error_projecting_group;
                        /* error: receiving group of incorrect size */
                        if (tg->vector->size != arg4)
                                goto error_receiving_group;
                        /* read first row of weights */
                        if (!fgets(buf, sizeof(buf), fd))
                                goto error_format;
                }
                /* read the matrix values */
                for (uint32_t r = 0; r < fg_to_tg->weights->rows; r++) {
                        char *tokens = strtok(buf, " ");
                        for (uint32_t c = 0; c < fg_to_tg->weights->cols; c++) {
                                /* error: expected another column */
                                if (!tokens)
                                        goto error_receiving_group;
                                /* error: non-numeric input */
                                if (sscanf(tokens, "%lf", &fg_to_tg->weights->elements[r][c]) != 1)
                                        goto error_format;
                                tokens = strtok(NULL, " ");
                                /* error: expected no more columns */
                                if (c == fg_to_tg->weights->cols - 1 && tokens)
                                        goto error_receiving_group;
                        }
                        /* error: expected another row */
                        if (r < fg_to_tg->weights->rows - 1 && !fgets(buf, sizeof(buf), fd))
                                goto error_projecting_group;
                }
                mprintf("... read weights for projection '%s -> %s'\n", arg1, arg2);
        }
        fclose(fd);
        return true;

error_file:
        eprintf("Cannot load weights - no such file '%s'\n", filename);
        return false;
error_format:
        eprintf("Cannot load weights - file has incorrect format\n");
        return false;
error_projecting_group:
        eprintf("Cannot load weights - projecting group of incorrect size\n");
        return false;
error_receiving_group:
        eprintf("Cannot load weights - receiving group of incorrect size\n");
        return false;
}
