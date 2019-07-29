/*
 * Copyright 2012-2019 Harm Brouwer <me@hbrouwer.eu>
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

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */
#include <math.h>
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
#include "rnn_unfold.h"
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
        strcpy(n->name, name);

        if (!(n->flags = malloc(sizeof(struct network_flags))))
                goto error_out;
        memset(n->flags, 0, sizeof(struct network_flags));
        if (!(n->pars = malloc(sizeof(struct network_params))))
                goto error_out;
        memset(n->pars, 0, sizeof(struct network_params));

        n->flags->type = type;
        n->groups      = create_array(atype_groups);
        n->sets        = create_array(atype_sets);

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
        n->flags->reset_contexts    = DEFAULT_RESET_CONTEXTS;
        n->pars->init_context_units = DEFAULT_INIT_CONTEXT_UNITS;
        n->random_algorithm         = DEFAULT_RANDOM_ALGORITHM;
        n->pars->random_mu          = DEFAULT_RANDOM_MU;
        n->pars->random_sigma       = DEFAULT_RANDOM_SIGMA;
        n->pars->random_min         = DEFAULT_RANDOM_MIN;
        n->pars->random_max         = DEFAULT_RANDOM_MAX;
        n->learning_algorithm       = DEFAULT_LEARNING_ALGORITHM;
        n->update_algorithm         = DEFAULT_UPDATE_ALGORITHM;
        n->pars->learning_rate      = DEFAULT_LEARNING_RATE;
        n->pars->lr_scale_factor    = DEFAULT_LR_SCALE_FACTOR;
        n->pars->lr_scale_after     = DEFAULT_LR_SCALE_AFTER;
        n->pars->momentum           = DEFAULT_MOMENTUM;
        n->pars->mn_scale_factor    = DEFAULT_MN_SCALE_FACTOR;
        n->pars->mn_scale_after     = DEFAULT_MN_SCALE_AFTER;
        n->pars->weight_decay       = DEFAULT_WEIGHT_DECAY;
        n->pars->wd_scale_factor    = DEFAULT_WD_SCALE_FACTOR;
        n->pars->wd_scale_after     = DEFAULT_WD_SCALE_AFTER;
        n->pars->target_radius      = DEFAULT_TARGET_RADIUS;
        n->pars->zero_error_radius  = DEFAULT_ZERO_ERROR_RADIUS;
        n->pars->error_threshold    = DEFAULT_ERROR_THRESHOLD;
        n->pars->max_epochs         = DEFAULT_MAX_EPOCHS;
        n->pars->report_after       = DEFAULT_REPORT_AFTER;
        n->pars->rp_init_update     = DEFAULT_RP_INIT_UPDATE;
        n->pars->rp_eta_plus        = DEFAULT_RP_ETA_PLUS;
        n->pars->rp_eta_minus       = DEFAULT_RP_ETA_MINUS;
        n->pars->dbd_rate_increment = DEFAULT_DBD_RATE_INCREMENT;
        n->pars->dbd_rate_decrement = DEFAULT_DBD_RATE_DECREMENT;
        n->similarity_metric        = DEFAULT_SIMILARITY_METRIC;
}

void init_network(struct network *n)
{
        n->flags->initialized = false;

        /*
         * Verify network sanity.
         */
        if (!verify_network(n))
                return;

        /*
         * Randomize weights, and initialize dynamic learning parameters.
         */
        srand(n->pars->random_seed);
        reset_network(n);

        /* 
         * If batch size is zero, set it to the number of items in the
         * active set.
         */
        if (n->pars->batch_size == 0 && n->asp)
                n->pars->batch_size = n->asp->items->num_elements;

        /*
         * If a recurrent neural network will be trained with
         * backpropagation through time, unfold it.
         */
        if (n->learning_algorithm == train_network_with_bptt) {
                if (n->unfolded_net)
                        rnn_free_unfolded_network(n->unfolded_net);
                n->unfolded_net = rnn_unfold_network(n);
        }

        n->flags->initialized = true;
}

void reset_network(struct network *n)
{
        reset_groups(n);
        reset_ffn_error_signals(n);
        reset_projection_matrices(n->input, n);
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
        free(n->flags);
        free(n->pars);
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
        if (n->flags->type == ntype_ffn)
                cprintf("ffn");
        if (n->flags->type == ntype_srn)
                cprintf("srn");
        if (n->flags->type == ntype_rnn)
                cprintf("rnn");
        cprintf("\n");
        cprintf("| Initialized: \t\t\t ");
        n->flags->initialized ? cprintf("true\n") : cprintf("false\n");
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
        n->flags->reset_contexts ? cprintf("true\n") : cprintf("false\n");
        cprintf("| Init context units: \t\t %f\n", n->pars->init_context_units);

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
        cprintf("| Back ticks: \t\t\t %d\n", n->pars->back_ticks);
        cprintf("| Update algorithm: \t\t ");
        if (n->update_algorithm == bp_update_sd
                && n->flags->sd_type == SD_DEFAULT)
                cprintf("steepest");
        if (n->update_algorithm == bp_update_sd
                && n->flags->sd_type == SD_BOUNDED)
                cprintf("bounded");
        if (n->update_algorithm == bp_update_rprop
                && n->flags->rp_type == RPROP_PLUS)
                cprintf("rprop+");
        if (n->update_algorithm == bp_update_rprop
                && n->flags->rp_type == RPROP_MINUS)
                cprintf("rprop-");
        if (n->update_algorithm == bp_update_rprop
                && n->flags->rp_type == IRPROP_PLUS)
                cprintf("irprop+");
        if (n->update_algorithm == bp_update_rprop
                && n->flags->rp_type == IRPROP_MINUS)
                cprintf("irprop+");
        if (n->update_algorithm == bp_update_qprop)
                cprintf("qprop");
        if (n->update_algorithm == bp_update_dbd)
                cprintf("dbd");
        cprintf("\n");
        cprintf("|\n");
        cprintf("| Learning rate (LR): \t\t %f\n",      n->pars->learning_rate);
        cprintf("| LR scale factor: \t\t %f\n",         n->pars->lr_scale_factor);
        cprintf("| LR scale after (%%epochs): \t %f\n", n->pars->lr_scale_after);
        cprintf("|\n");
        cprintf("| Momentum (MN): \t\t %f\n",           n->pars->momentum);
        cprintf("| MN scale factor: \t\t %f\n",         n->pars->mn_scale_factor);
        cprintf("| MN scale after (%%epochs): \t %f\n", n->pars->mn_scale_after);
        cprintf("|\n");
        cprintf("| Rprop init update: \t\t %f\n",       n->pars->rp_init_update);
        cprintf("| Rprop Eta-: \t\t\t %f\n",            n->pars->rp_eta_minus);
        cprintf("| Rprop Eta+: \t\t\t %f\n",            n->pars->rp_eta_plus);
        cprintf("|\n");
        cprintf("| DBD rate increment: \t\t %f\n",      n->pars->rp_init_update);
        cprintf("| DBD rate decrement: \t\t %f\n",      n->pars->rp_eta_minus);
        cprintf("|\n");
        cprintf("| Weight decay (WD): \t\t %f\n",       n->pars->weight_decay);
        cprintf("| WD scale factor: \t\t %f\n",         n->pars->wd_scale_factor);
        cprintf("| WD scale after (%%epochs): \t %f\n", n->pars->wd_scale_after);
        cprintf("|\n");
        cprintf("| Target radius: \t\t %f\n",           n->pars->target_radius);
        cprintf("| Zero error radius: \t\t %f\n",       n->pars->zero_error_radius);
        cprintf("| Error threshold: \t\t %f\n",         n->pars->error_threshold);
        cprintf("|\n");
        cprintf("| Training order: \t\t ");
        if (n->flags->training_order == train_ordered)
                cprintf("ordered");
        if (n->flags->training_order == train_permuted)
                cprintf("permuted");
        if (n->flags->training_order == train_randomized)
                cprintf("randomized");
        cprintf("\n");
        cprintf("| Batch size: \t\t\t %d\n",            n->pars->batch_size);
        cprintf("| Maximum #epochs: \t\t %d\n",         n->pars->max_epochs);
        cprintf("| Report after #epochs \t\t %d\n",     n->pars->report_after);
        if (n->ts_fw_group) {
                cprintf("|\n");
                cprintf("| Two-stage forward: \t\t %s (%d) :: %s (%d)\n", 
                        n->ts_fw_group->name, n->ts_fw_group->vector->size,
                        n->ts_fw_set->name, n->ts_fw_set->items->num_elements);
        }
        if (n->ts_bw_group) {
                if (!n->ts_fw_group)
                        cprintf("|\n");
                cprintf("| Two-stage backward: \t\t %s (%d) :: %s (%d)\n", 
                        n->ts_bw_group->name, n->ts_bw_group->vector->size,
                        n->ts_bw_set->name, n->ts_bw_set->items->num_elements);
        }

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
        cprintf("| Random Seed: \t\t\t %d\n", n->pars->random_seed);
        cprintf("| Random Mu: \t\t\t %f\n",   n->pars->random_mu);
        cprintf("| Random Sigma: \t\t %f\n",  n->pars->random_sigma);
        cprintf("| Random Min: \t\t\t %f\n",  n->pars->random_min);
        cprintf("| Random Max: \t\t\t %f\n",  n->pars->random_max);

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
                 **** OpenMP ****
                 ****************/
                
#ifdef _OPENMP
        cprintf("|\n");
        cprintf("| Multithreading enabled: \t ");
        n->flags->omp_mthreaded ? cprintf("true\n") : cprintf("false\n");
        cprintf("| Processor(s) available: \t %d\n",  omp_get_num_procs());
        cprintf("| Maximum #threads: \t\t %d\n",      omp_get_max_threads());
        cprintf("| Schedule: \t\t\t ");
        omp_sched_t k;
        int m;
        omp_get_schedule(&k, &m);
        switch (k) {
        case omp_sched_static:
                cprintf("static\n");
                break;
        case omp_sched_dynamic:
                cprintf("dynamic\n");
                break;
        case omp_sched_guided:
                cprintf("guided\n");
                break;
        case omp_sched_auto:
                cprintf("auto\n");
                break;
        default: 
                /* to handle omp_sched_monotonic */
                break;
        }
        cprintf("| Chunk size \t\t\t %d\n", m);
        cprintf("\n");
#endif /* _OPENMP */


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
        strcpy(g->name, name);

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

        if(!(g->flags = malloc(sizeof(struct group_flags))))
                goto error_out;
        memset(g->flags, 0, sizeof(struct  group_flags));
        if(!(g->pars = malloc(sizeof(struct group_params))))
                goto error_out;
        memset(g->pars, 0, sizeof(struct  group_params));

        g->vector     = create_vector(size);
        g->error      = create_vector(size);
        g->inc_projs  = create_array(atype_projs);
        g->out_projs  = create_array(atype_projs);
        g->ctx_groups = create_array(atype_groups);

        g->flags->bias = bias;

        g->pars->relu_alpha    = DEFAULT_RELU_ALPHA;
        g->pars->relu_max      = DEFAULT_RELU_MAX;
        g->pars->logistic_fsc  = DEFAULT_LOGISTIC_FSC;
        g->pars->logistic_gain = DEFAULT_LOGISTIC_GAIN;

        /* bias nodes have activation 1.0 */
        if(g->flags->bias)
                g->vector->elements[0] = 1.0;

        return g;

error_out:
        perror("[create_group()]");
        return NULL;
}

struct group *create_bias_group(char *name)
{
        return create_group(name, 1, true, false);
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

        add_group(n, bg);
        add_bidirectional_projection(bg, g);

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
        free(g->flags);
        free(g->pars);
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
                struct projection *ip = g->inc_projs->elements[i];
                struct group *fg = ip->to;
                struct projection *op = find_projection(fg->out_projs, g);
                remove_projection(fg->out_projs, op);
        }
        /* remove incoming projections to a group g' from g */
        for (uint32_t i = 0; i < g->out_projs->num_elements; i++) {
                struct projection *op = g->out_projs->elements[i];
                struct group *tg = op->to;
                struct projection *ip = find_projection(tg->inc_projs, g);
                remove_projection(tg->inc_projs, ip);
        }
        /* remove Elman projections from a group g' to group g */
        for (uint32_t i = 0; i < n->groups->num_elements; i++) {
                struct group *fg = n->groups->elements[i];
                if (find_elman_projection(fg, g))
                        remove_elman_projection(fg, g);
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
                                g->pars->logistic_fsc,
                                g->pars->logistic_gain);
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
                        cprintf(" :: relu (max = %f)",
                                g->pars->relu_max);
                if (g->act_fun->fun == act_fun_leaky_relu)
                        cprintf(" :: leaky_relu (alpha = %f; max = %f)",
                                g->pars->relu_alpha,
                                g->pars->relu_max);
                if (g->act_fun->fun == act_fun_elu)
                        cprintf(" :: elu (alpha = %f; max = %f)",
                                g->pars->relu_alpha,
                                g->pars->relu_max);

                /* error function */
                if (g->err_fun->fun == err_fun_sum_of_squares)
                        cprintf(" :: sum_of_squares");
                if (g->err_fun->fun == err_fun_cross_entropy)
                        cprintf(" :: cross_entropy");
                if (g->err_fun->fun == err_fun_divergence)
                        cprintf(" :: divergence");

                /* bias */
                if (g->flags->bias)
                        cprintf(" :: bias group");

                /* input/output group */
                if (g == n->input)
                        cprintf(" :: input group\n");
                else if (g == n->output)
                        cprintf(" :: output group\n");
                else
                        cprintf("\n");                        
        }       
}

/*
 * Resets the units of all non-bias groups to zero. Groups that have context
 * groups get their units set to the initial context unit value. This
 * assures that when context resetting is disabled, the initial context unit
 * values get shifted into the context groups at the first tick after
 * initialization.
 */
void reset_groups(struct network *n)
{
        for (uint32_t i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];
                if (g->flags->bias)
                        continue;
                if (g->ctx_groups->num_elements > 0) {
                        fill_vector_with_value(
                                g->vector,
                                n->pars->init_context_units);
                } else {
                        zero_out_vector(g->vector);
                }
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
        copy_vector(v, g->vector);
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
        if (n->flags->initialized && !n->flags->reset_contexts)
                return;
        n->unfolded_net->sp = 0;
}

void reset_context_groups(struct network *n)
{
        /*
         * If context groups should not be reset, shift the context groups.
         */
        if (n->flags->initialized && !n->flags->reset_contexts) {
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
        fill_vector_with_value(g->vector, n->pars->init_context_units);
}

void reset_recurrent_groups(struct network *n)
{
        /*
         * If context groups should not be reset, shift the pointer or the
         * stack.
         *
         * TODO: Validate this logic!
         */
        if (n->flags->initialized && !n->flags->reset_contexts) {
                shift_pointer_or_stack(n);
                return;
        }
        for (uint32_t i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];
                for (uint32_t j = 0; j < g->inc_projs->num_elements; j++) {
                        struct projection *p = g->inc_projs->elements[j];
                        if (!p->flags->recurrent)
                                continue;
                        fill_vector_with_value(p->to->vector,
                                n->pars->init_context_units);
                }
        }
}

void reset_ffn_error_signals(struct network *n)
{
        for (uint32_t i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];
                zero_out_vector(g->error);
        }
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
                        /*
                        if (i > 0)
                                continue;
                        for (uint32_t x = 0; x < g->inc_projs->num_elements; x++) {
                                struct projection *p = g->inc_projs->elements[x];
                                if (p->flags->recurrent) {
                                        print_vector(p->to->error);
                                        zero_out_vector(p->to->error);
                                }
                        }
                        */
                }
        }
}

struct projection *create_projection(
        struct group *to,
        struct matrix *weights,
        struct matrix *gradients,
        struct matrix *prev_gradients,
        struct matrix *prev_deltas,
        struct matrix *dynamic_params,
        struct projection_flags *flags)
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
        p->flags          = flags;

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
        free(p->flags);
        free(p);
}

void add_projection(struct array *projs, struct projection *p)
{
        add_to_array(projs, p);
}

void add_bidirectional_projection(struct group *fg, struct group *tg)
{
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
        /* flags */
        struct projection_flags *flags;
        if (!(flags = malloc(sizeof(struct projection_flags))))
                goto error_out;
        memset(flags, 0, sizeof(struct projection_flags));
        
        /* 
         * Flag recurrent projection, if 'from' and 'to' group
         * are the same.
         */
        if (fg == tg)
                flags->recurrent = true;

        /* add projections */
        struct projection *op = create_projection(tg, weights, gradients,
                prev_gradients, prev_deltas, dynamic_params, flags);
        struct projection *ip = create_projection(fg, weights, gradients,
                prev_gradients, prev_deltas, dynamic_params, flags);
        add_projection(fg->out_projs, op);
        add_projection(tg->inc_projs, ip);

        return;

error_out:
        perror("[add_bidirectional_projection()]");
        return;
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
                /* current group */
                if (g->inc_projs->num_elements > 0)
                        cprintf(" -> ", g->name);
                cprintf("[%s]", g->name);
                if (g->out_projs->num_elements > 0)
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
                cprintf("\n");
                /* context (Elman) groups */
                if (g->ctx_groups->num_elements > 0) {
                        cprintf("* %d: [%s] => ", i + 1, g->name);
                        for (uint32_t j = 0;
                                j < g->ctx_groups->num_elements; j++) {
                                struct group *cg = g->ctx_groups->elements[j];
                                if (j > 0 && j < g->out_projs->num_elements)
                                        cprintf(", ");
                                cprintf("%s (copy)", cg->name);
                        }
                        cprintf("\n");
                }
        }     
}


void freeze_projection(struct projection *p)
{
        p->flags->frozen = true;
}

void unfreeze_projection(struct projection *p)
{
        p->flags->frozen = false;
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

void reset_projection_matrices(struct group *g, struct network *n)
{
        /* incoming projections */
        for (uint32_t i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *ip = g->inc_projs->elements[i];
                if (ip->flags->frozen)
                        continue;
                zero_out_matrix(ip->weights);
                zero_out_matrix(ip->gradients);
                zero_out_matrix(ip->prev_deltas);
                zero_out_matrix(ip->prev_gradients);
        }
        /* outgoing projections */
        for (uint32_t i = 0; i < g->out_projs->num_elements; i++) {
                struct projection *op = g->out_projs->elements[i];
                if (op->flags->recurrent)
                        continue;
                reset_projection_matrices(op->to, n);
        }
}

void randomize_weight_matrices(struct group *g, struct network *n)
{
        /* incoming projections */
        for (uint32_t i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *ip = g->inc_projs->elements[i];
                if (ip->flags->frozen)
                        continue;
                n->random_algorithm(ip->weights, n);
        }
        /* outgoing projections */
        for (uint32_t i = 0; i < g->out_projs->num_elements; i++) {
                struct projection *op = g->out_projs->elements[i];
                if (op->flags->recurrent)
                        continue;
                randomize_weight_matrices(op->to, n);
        }
}

void initialize_dynamic_params(struct group *g, struct network *n)
{
        double v = 0.0;
        if (n->update_algorithm == bp_update_rprop)
                v = n->pars->rp_init_update;
        if (n->update_algorithm == bp_update_dbd)
                v = n->pars->learning_rate;
        /* incoming projections */
        for (uint32_t i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *ip = g->inc_projs->elements[i];
                fill_matrix_with_value(ip->dynamic_params, v);
        }
        /* outgoing projections */
        for (uint32_t i = 0; i < g->out_projs->num_elements; i++) {
                struct projection *op = g->out_projs->elements[i];
                if (op->flags->recurrent)
                        continue;
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
        save_weight_matrix(n->input, fd);
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
                if (op->flags->recurrent)
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
                 * this point. If it is not there, we ran into a file format
                 * or a dimensionality mismatch problem.
                 */
                char arg1[MAX_ARG_SIZE]; /* 'from' group name */
                char arg2[MAX_ARG_SIZE]; /* 'to' group name */
                if (sscanf(buf, "Projection %s %s", arg1, arg2) != 2)
                        /*
                         * XXX: Legacy format ...
                         */
                        if (sscanf(buf, "%s -> %s", arg1, arg2) != 2)
                                /* error: expected no more rows */
                                goto error_format;
                /* find 'from' group */
                struct group *fg;
                if ((fg = find_array_element_by_name(n->groups, arg1)) == NULL) {
                        eprintf("Cannot load weights - no such group '%s'\n", arg1);
                        return false;
                }
                /* find 'to' group */
                struct group *tg;
                if ((tg = find_array_element_by_name(n->groups, arg2)) == NULL) {
                        eprintf("Cannot load weights - no such group '%s'\n", arg2);
                        return false;
                }
                /* projection should exist */
                struct projection *fg_to_tg = find_projection(fg->out_projs, tg);
                if (!fg_to_tg) {
                        eprintf("Cannot load weights - no projection between groups '%s' and '%s'\n", arg1, arg2);
                        return false;
                }
                /* read weight matrix */
                if (load_weight_matrix(fd, fg_to_tg->weights))
                        mprintf("... read weights for projection '%s -> %s'\n", arg1, arg2);
                else
                        return false;
        }
        fclose(fd);
        return true;

error_file:
        eprintf("Cannot load weights - no such file '%s'\n", filename);
        return false;
error_format:
        eprintf("Cannot load weights - file has incorrect format\n");
        return false;
}

bool load_weight_matrix(FILE *fd, struct matrix *weights)
{
        char buf[MAX_BUF_SIZE];
        /*
         * Read the next line, which may be an optional dimensions
         * specification, or the first row of weights.
         */
        if (!fgets(buf, sizeof(buf), fd))
                goto error_format;
        uint32_t arg1; /* 'from' group size */
        uint32_t arg2; /* 'to' group size */
        /* 
         * Check for dimension specification, and in case it is
         * present, verify the dimensionality.
         */
        if (sscanf(buf, "Dimensions %d %d", &arg1, &arg2) == 2) {
                /* error: projecting group of incorrect size */
                if (weights->rows != arg1) 
                        goto error_projecting_group;
                /* error: receiving group of incorrect size */
                if (weights->cols != arg2)
                        goto error_receiving_group;
                /* read first row of weights */
                if (!fgets(buf, sizeof(buf), fd))
                        goto error_format;
        }
        /* read the matrix values */
        for (uint32_t r = 0; r < weights->rows; r++) {
                char *tokens = strtok(buf, " ");
                for (uint32_t c = 0; c < weights->cols; c++) {
                        /* error: expected another column */
                        if (!tokens)
                                goto error_receiving_group;
                        /* error: non-numeric input */
                        if (sscanf(tokens, "%lf", &weights->elements[r][c]) != 1)
                                goto error_format;
                        tokens = strtok(NULL, " ");
                        /* error: expected no more columns */
                        if (c == weights->cols - 1 && tokens)
                                goto error_receiving_group;
                }
                /* error: expected another row */
                if (r < weights->rows - 1 && !fgets(buf, sizeof(buf), fd))
                        goto error_projecting_group;
        }
        return true;

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
