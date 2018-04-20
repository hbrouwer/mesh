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
#include "classify.h"
#include "cmd.h"
#include "error.h"
#include "help.h"
#include "main.h"
#include "math.h"
#include "matrix.h"
#include "network.h"
#include "pprint.h"
#include "random.h"
#include "record.h"
#include "set.h"
#include "stats.h"
#include "similarity.h"
#include "test.h"
#include "train.h"
#include "modules/dss.h"
#include "modules/erp.h"

                /***************************
                 **** command processor ****
                 ***************************/

/*
 * Match an incoming command against the base of a command in the command
 * list, and process it if possible. Within the command list, two commands
 * are special `createNetwork` and `init`:
 *
 * - All commands following `createNetwork` require an active network to be
 *   present in the current session;
 * 
 * - And all commands following `init` require an initialized network to be
 *   prsent in the current session;
 */
void process_command(char *cmd, struct session *s)
{
        /* comment or blank line */
        switch (cmd[0]) {
        case '%':       /* verbose comment */
                cprintf("\x1b[1m\x1b[36m%s\x1b[0m\n", cmd);
                goto out;
        case '#':       /* silent comment */
        case '\0':      /* blank line */
                goto out;
        }

        char fmt[MAX_FMT_SIZE];       
        bool req_anp  = false; /* require active network */
        bool req_init = false; /* require intialized network */
        bool req_asp  = false; /* require active set */
        for (uint32_t i = 0; cmds[i].cmd_base != NULL; i++) {
                /* 
                 * Skip commands that require an active network if
                 * necessary.
                 */
                if (req_anp && !s->anp) {
                        eprintf("Cannot process command: `%s`\n", cmd);
                        eprintf("(no active network - see `help networks`)\n");
                        goto out;
                }
                /* 
                 * Skip commands that require an initialized network if
                 * necessary.
                 */
                if (req_init && !s->anp->initialized) {
                        eprintf("Cannot process command: `%s`\n", cmd);
                        eprintf("(uninitialized network - use `init` command to initialize)\n");
                        goto out;
                }
                /*
                 * Skip commands that require an active example set if
                 * necessary.
                 */
                if (req_asp && !s->anp->asp) {
                        eprintf("Cannot process command: `%s`\n", cmd);
                        eprintf("(no active set - see `help sets`)\n");
                        goto out;
                }
                /*
                 * If a command has arguments, we pass its processor its
                 * base and its arguments. Otherwise, we just pass its base.
                 * 
                 * Each command processor returns true if the command passed
                 * to it could be parsed and executed either successfully or
                 * unsuccessfully. It returns false, by contrast, if a
                 * command could not be parsed.
                 */
                if (strncmp(cmd, cmds[i].cmd_base,
                        strlen(cmds[i].cmd_base)) == 0) {
                        bool success;
                        if (cmds[i].cmd_args != NULL) {
                                size_t block_size = (strlen(cmds[i].cmd_base) + 1
                                         + strlen(cmds[i].cmd_args) + 1) * sizeof(char);
                                memset(fmt, 0, MAX_FMT_SIZE);
                                snprintf(fmt, block_size, "%s %s",
                                        cmds[i].cmd_base, cmds[i].cmd_args);
                                success = cmds[i].cmd_proc(cmd, fmt, s);
                        } else {                   
                                success = cmds[i].cmd_proc(
                                        cmd,cmds[i].cmd_base, s);
                        }
                        if (success)
                                goto out;
                }
                /* 
                 * All commands following `createNetwork` require an active
                 * network.
                 */
                else if (strcmp("createNetwork", cmds[i].cmd_base) == 0) {
                        req_anp = true;
                }
                /* 
                 * All commands following `init` require an initialized
                 * network, and an active example set.
                 */
                else if (strcmp("init", cmds[i].cmd_base) == 0) {
                        req_init = true;
                        req_asp  = true;
                }
        }

        /* invalid command */
        if (strlen(cmd) > 1) {
                eprintf("No such command: `%s`\n", cmd); 
                eprintf("(type `help` for help)\n");
        }

out:
        return;
}

                /******************
                 **** commands ****
                 ******************/

bool cmd_exit(char *cmd, char *fmt, struct session *s)
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        cprintf("Goodbye.\n");
        free_session(s);
        exit(EXIT_SUCCESS);

        return true;
}

bool cmd_about(char *cmd, char *fmt, struct session *s)
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        help("about");

        return true;
}

bool cmd_help(char *cmd, char *fmt, struct session *s)
{
        char arg[MAX_ARG_SIZE]; /* help topic */
        
        bool help_on_topic = false;
        if (strcmp(cmd, fmt) != 0) {
                if (sscanf(cmd, fmt, arg) != 1) {
                        return false;
                } else {
                        help_on_topic = true;
                }
        }

        if (help_on_topic)
                help(arg);
        else
                help("general");
        
        return true;
}

bool cmd_load_file(char *cmd, char *fmt, struct session *s)
{
        char arg[MAX_ARG_SIZE]; /* filename */
        if (sscanf(cmd, fmt, arg) != 1)
                return false;

        FILE *fd;
        if (!(fd = fopen(arg, "r"))) {
                eprintf("cannot open file '%s'\n", arg);
                return true;
        }
        char buf[MAX_BUF_SIZE];
        while (fgets(buf, sizeof(buf), fd)) {
                buf[strlen(buf) - 1] = '\0';
                process_command(buf, s);
        }
        fclose(fd);

        mprintf("Loaded file \t\t\t [ %s ]\n", arg);

        return true;
}

bool cmd_create_network(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* network name */
        char arg2[MAX_ARG_SIZE]; /* network type */
        if (sscanf(cmd, fmt, arg1, arg2) != 2)
                return false;

        enum network_type type = 0;
        /* feed forward network */
        if (strcmp(arg2, "ffn") == 0)
                type = ntype_ffn;
         /* simple recurrent network */
        else if (strcmp(arg2, "srn") == 0)
                type = ntype_srn;
        /* recurrent network */
        else if (strcmp(arg2, "rnn") == 0)
                type = ntype_rnn;
        else {
                eprintf("Cannot create network - invalid network type: '%s'\n",
                        arg2);
                return true;
        }

        /* network should not already exist */
        if (find_array_element_by_name(s->networks, arg1)) {
                eprintf("Cannot create network - network '%s' already exists\n",
                        arg1);
                return true;
        }
        
        /* create network, and set as active */
        struct network *n = create_network(arg1, type);
        add_to_array(s->networks, n);
        s->anp = n;

        mprintf("Created network \t\t [ %s :: %s ]\n", arg1, arg2);

        return true;
}

bool cmd_remove_network(char *cmd, char *fmt, struct session *s)
{
        char arg[MAX_ARG_SIZE]; /* network name */
        if (sscanf(cmd, fmt, arg) != 1)
                return false;

        /* find network */
        struct network *n = find_array_element_by_name(s->networks, arg);
        if (!n) {
                eprintf("Cannot remove network - no such network '%s'\n", arg);
                return true;
        }

        /*
         * If the network to be removed is the active network, try finding
         * another active network.
         */
        if (n == s->anp) {
                s->anp = NULL;
                for (uint32_t i = 0; i < s->networks->num_elements; i++)
                        if (s->networks->elements[i] != NULL
                                && s->networks->elements[i] != n)
                                s->anp = s->networks->elements[i];
        }

        /* remove network */
        remove_from_array(s->networks, n);
        free_network(n);

        mprintf("Removed network \t\t [ %s ]\n", arg);

        return true;
}

bool cmd_networks(char *cmd, char *fmt, struct session *s)
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        cprintf("Networks:\n");
        if (s->networks->num_elements == 0) {
                cprintf("(no networks)\n");
        } else {
                for (uint32_t i = 0; i < s->networks->num_elements; i++) {
                        struct network *n = s->networks->elements[i];
                        cprintf("* %d: %s", i + 1, n->name);
                        if (n == s->anp)
                                cprintf(" :: active network\n");
                        else
                                cprintf("\n");
                }
        }

        return true;
}

bool cmd_change_network(char *cmd, char *fmt, struct session *s)
{
        char arg[MAX_ARG_SIZE]; /* network name */
        if (sscanf(cmd, fmt, arg) != 1)
                return false;

        /* find network */
        struct network *n = find_array_element_by_name(s->networks, arg);
        if (!n) {
                eprintf("Cannot change to network - no such network '%s'\n", arg);
                return true;
        }
        s->anp = n;

        mprintf("Changed to network \t [ %s ]\n", arg);

        return true;
}

bool cmd_inspect(char *cmd, char *fmt, struct session *s)
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        struct network *n = s->anp;

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

        return true;
}

bool cmd_create_group(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* group name */
        uint32_t arg2;
        if (sscanf(cmd, fmt, arg1, &arg2) != 2)
                return false;

        /* group should not already exist */
        if (find_array_element_by_name(s->anp->groups, arg1)) {
                eprintf("Cannot create group - group '%s' already exists in network '%s'\n",
                        arg1, s->anp->name);
                return true;
        }

        /* group size should be positive */
        if (!(arg2 > 0)) {
                eprintf("Cannot create group - group size should be positive\n");
                return true;
        }

        struct group *g = create_group(arg1, arg2, false, false);
        add_to_array(s->anp->groups, g);

        mprintf("Created group \t\t [ %s :: %d ]\n", arg1, arg2);

        return true;
}

bool cmd_remove_group(char *cmd, char *fmt, struct session *s)
{
        char arg[MAX_ARG_SIZE]; /* group name */
        if (sscanf(cmd, fmt, arg) != 1)
                return false;

        /* find group */
        struct group *g = find_array_element_by_name(s->anp->groups, arg);
        if (g == NULL) {
                eprintf("Cannot remove group - no such group '%s'\n", arg);
                return true;
        }

        /* remove outgoing projections from a group g' to group g */
        for (uint32_t i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *p = g->inc_projs->elements[i];
                struct group *fg = p->to;
                for (uint32_t j = 0; j < fg->out_projs->num_elements; j++) {
                        struct projection *op = fg->out_projs->elements[j];
                        if (op->to == g) {
                                remove_from_array(fg->out_projs, op);
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
                                remove_from_array(tg->inc_projs, ip);
                                break;
                        }
                }
        }

        /* remove Elman projections from a group g' to group g */
        for (uint32_t i = 0; i < s->anp->groups->num_elements; i++) {
                struct group *fg = s->anp->groups->elements[i];
                for (uint32_t j = 0; j < fg->ctx_groups->num_elements; j++) {
                        if (fg->ctx_groups->elements[j] == g) {
                                remove_from_array(fg->ctx_groups, g);
                                break;
                        }
                }
        }

        /* remove group */
        remove_from_array(s->anp->groups, g);
        free_group(g);

        mprintf("Removed group \t\t [ %s ]\n", arg);

        return true;
}

bool cmd_groups(char *cmd, char *fmt, struct session *s)
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        cprintf("Groups in network '%s':\n", s->anp->name);
        if (s->anp->groups->num_elements == 0) {
                cprintf("(no groups)\n");
        } else {
                for (uint32_t i = 0; i < s->anp->groups->num_elements; i++) {
                        struct group *g = s->anp->groups->elements[i];

                        /* name and size */
                        cprintf("* %d: %s :: %d", i + 1, g->name, g->vector->size);

                        /* activation function */
                        if (g->act_fun->fun == act_fun_logistic)
                                cprintf(" :: logistic");
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
                        if (g->err_fun->fun == error_sum_of_squares)
                                cprintf(" :: sum_of_squares");
                        if (g->err_fun->fun == error_cross_entropy)
                                cprintf(" :: cross_entropy");
                        if (g->err_fun->fun == error_divergence)
                                cprintf(" :: divergence");

                        /* input/output group */
                        if (g == s->anp->input)
                                cprintf(" :: input group\n");
                        else if (g == s->anp->output)
                                cprintf(" :: output group\n");
                        else
                                cprintf("\n");
                }
        }

        return true;
}

bool cmd_attach_bias(char *cmd, char *fmt, struct session *s)
{
        char arg[MAX_ARG_SIZE]; /* group name */
        if (sscanf(cmd, fmt, arg) != 1)
                return false;

        /* find group */
        struct group *g = find_array_element_by_name(s->anp->groups, arg);
        if (g == NULL) {
                eprintf("Cannot attach bias group - no such group '%s'\n", arg);
                return true;
        }

        /* bias group should not already exists */
        char bias_suffix[] = "_bias";
        char *arg_bias;
        size_t block_size = (strlen(arg) + strlen(bias_suffix) + 1) * sizeof(char);
        if (!(arg_bias = malloc(block_size)))
                goto error_out;
        memset(arg_bias, 0, block_size);
        snprintf(arg_bias, block_size, "%s%s", arg, bias_suffix);
        if (find_array_element_by_name(s->anp->groups, arg_bias)) {
                eprintf("Cannot attach bias group - group '%s' already exists in network '%s'\n",
                        arg_bias, s->anp->name);
                return true;
        }
        free(arg_bias);

        /* attach bias */
        struct group *bg = attach_bias_group(s->anp, g);

        mprintf("Attached bias to group \t [ %s -> %s ]\n", bg->name, g->name);

        return true;

error_out:
        perror("[cmd_attach_bias()]");
        return true;
}

bool cmd_set_input_group(char *cmd, char *fmt, struct session *s)
{
        char arg[MAX_ARG_SIZE]; /* group name */
        if (sscanf(cmd, fmt, arg) != 1)
                return false;

        /* find group */
        struct group *g = find_array_element_by_name(s->anp->groups, arg);
        if (g == NULL) {
                eprintf("Cannot set input group - no such group '%s'\n", arg);
                return true;
        }

        s->anp->input = g;
        
        mprintf("Set input group \t\t [ %s ]\n", arg);

        return true;
}

bool cmd_set_output_group(char *cmd, char *fmt, struct session *s)
{
        char arg[MAX_ARG_SIZE]; /* group name */
        if (sscanf(cmd, fmt, arg) != 1)
                return false;

        /* find group */
        struct group *g = find_array_element_by_name(s->anp->groups, arg);
        if (g == NULL) {
                eprintf("Cannot set output group - no such group '%s'\n", arg);
                return true;
        }

        s->anp->output = g;
        
        mprintf("Set output group \t\t [ %s ]\n", arg);

        return true;
}

bool cmd_set_act_func(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* group name */
        char arg2[MAX_ARG_SIZE]; /* activation function */
        if (sscanf(cmd, fmt, arg1, arg2) != 2)
                return false;

        /* find group */
        struct group *g = find_array_element_by_name(s->anp->groups, arg1);
        if (g == NULL) {
                eprintf("Cannot set activation function - no such group '%s'\n", arg1);
                return true;
        }

        /* logistic function */
        if (strcmp(arg2, "logistic") == 0) {
                g->act_fun->fun   = act_fun_logistic;
                g->act_fun->deriv = act_fun_logistic_deriv;
        }
        /* binary sigmoid [= logistic] function (for legacy models) */
        else if (strcmp(arg2, "binary_sigmoid") == 0) {
                g->act_fun->fun   = act_fun_logistic;
                g->act_fun->deriv = act_fun_logistic_deriv;
        }        
        /* bipolar sigmoid function */
        else if (strcmp(arg2, "bipolar_sigmoid") == 0) {
                g->act_fun->fun   = act_fun_bipolar_sigmoid;
                g->act_fun->deriv = act_fun_bipolar_sigmoid_deriv;
        }
        /* softmax activation function */
        else if (strcmp(arg2, "softmax") == 0) {
                g->act_fun->fun   = act_fun_softmax;
                g->act_fun->deriv = act_fun_softmax_deriv;
        }
        /* hyperbolic tangent function */
        else if (strcmp(arg2, "tanh") == 0) {
                g->act_fun->fun   = act_fun_tanh;
                g->act_fun->deriv = act_fun_tanh_deriv;
        }
        /* linear function */
        else if (strcmp(arg2, "linear") == 0) {
                g->act_fun->fun   = act_fun_linear;
                g->act_fun->deriv = act_fun_linear_deriv;
        }
        /* softplus activation function */
        else if (strcmp(arg2, "softplus") == 0) {
                g->act_fun->fun   = act_fun_softplus;
                g->act_fun->deriv = act_fun_softplus_deriv;
        }
        /* relu activation function */
        else if (strcmp(arg2, "relu") == 0) {
                g->act_fun->fun   = act_fun_relu;
                g->act_fun->deriv = act_fun_relu_deriv;
        }
        /* binary relu activation function */
        else if (strcmp(arg2, "binary_relu") == 0) {
                g->act_fun->fun   = act_fun_binary_relu;
                g->act_fun->deriv = act_fun_binary_relu_deriv;
        }        
        /* leaky relu activation function */
        else if (strcmp(arg2, "leaky_relu") == 0) {
                g->act_fun->fun   = act_fun_leaky_relu;
                g->act_fun->deriv = act_fun_leaky_relu_deriv;
        }
        /* elu activation function */
        else if (strcmp(arg2, "elu") == 0) {
                g->act_fun->fun   = act_fun_elu;
                g->act_fun->deriv = act_fun_elu_deriv;
        } else {
                eprintf("Cannot set activation function - no such activation function '%s'\n", arg2);
                return true;
        }

        mprintf("Set activation function \t [ %s :: %s ]\n", arg1, arg2);

        return true;
}

bool cmd_set_err_func(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* group name */
        char arg2[MAX_ARG_SIZE]; /* error function */
        if (sscanf(cmd, fmt, arg1, arg2) != 2)
                return false;

        /* find group */
        struct group *g = find_array_element_by_name(s->anp->groups, arg1);
        if (g == NULL) {
                eprintf("Cannot set error function - no such group '%s'\n", arg1);
                return true;
        }

        /* sum of squares */
        if (strcmp(arg2, "sum_of_squares") == 0) {
                g->err_fun->fun   = error_sum_of_squares;
                g->err_fun->deriv = error_sum_of_squares_deriv;
        }
        /* sum of squares */
        else if (strcmp(arg2, "sum_squares") == 0) {
                g->err_fun->fun   = error_sum_of_squares;
                g->err_fun->deriv = error_sum_of_squares_deriv;
        }        
        /* cross-entropy */
        else if (strcmp(arg2, "cross_entropy") == 0) {
                g->err_fun->fun   = error_cross_entropy;
                g->err_fun->deriv = error_cross_entropy_deriv;
        }
        /* divergence */
        else if (strcmp(arg2, "divergence") == 0) {
                g->err_fun->fun   = error_divergence;
                g->err_fun->deriv = error_divergence_deriv;
        } else {
                eprintf("Cannot set error function - no such error function '%s'\n", arg2);
                return true;
        }

        mprintf("Set error function \t\t [ %s :: %s ]\n", arg1, arg2);

        return true;
}

bool cmd_create_projection(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* 'from' group name */
        char arg2[MAX_ARG_SIZE]; /* 'to' group name */
        if (sscanf(cmd, fmt, arg1, arg2) != 2)
                return false;

        /* find 'from' group */
        struct group *fg = find_array_element_by_name(s->anp->groups, arg1);
        if (fg == NULL) {
                eprintf("Cannot set projection - no such group '%s'\n", arg1);
                return true;
        }
        /* find 'to' group */
        struct group *tg = find_array_element_by_name(s->anp->groups, arg2);
        if (tg == NULL) {
                eprintf("Cannot set projection - no such group '%s'\n", arg2);
                return true;
        }

        /* projection should not already exist */
        bool exists = false;
        if (fg->recurrent)
                exists = true;
        for (uint32_t i = 0; i < fg->out_projs->num_elements; i++)
                if (((struct projection *)fg->out_projs->elements[i])->to == tg) {
                        exists = true;
                        break;
                }
        if (exists) {
                eprintf("Cannot set projection - projection '%s -> %s' already exists\n",
                        arg1, arg2);
                return true;
        }

        /* create projection */
        if (fg == tg)
                fg->recurrent = true;
        else {
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

                add_to_array(fg->out_projs, op);
                add_to_array(tg->inc_projs, ip);
        }

        mprintf("Created projection \t\t [ %s -> %s ]\n", arg1, arg2);

        return true;
}

bool cmd_remove_projection(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* 'from' group name */
        char arg2[MAX_ARG_SIZE]; /* 'to' group name */
        if (sscanf(cmd, fmt, arg1, arg2) != 2)
                return false;

        /* find 'from' group */
        struct group *fg = find_array_element_by_name(s->anp->groups, arg1);
        if (fg == NULL) {
                eprintf("Cannot remove projection - no such group '%s'\n", arg1);
                return  true;
        }

        /* find 'to' group */
        struct group *tg = find_array_element_by_name(s->anp->groups, arg2);
        if (tg == NULL) {
                eprintf("Cannot remove projection - no such group '%s'\n", arg2);
                return true;
        }

        /* find outgoing 'from->to' projection */
        struct projection *fg_to_tg = NULL;
        for (uint32_t i = 0; i < fg->out_projs->num_elements; i++)
                if (((struct projection *)
                        fg->out_projs->elements[i])->to == tg) {
                        fg_to_tg = fg->out_projs->elements[i];
                        break;
                }
        
        /* find incoming 'to->from' projection */
        struct projection *tg_to_fg = NULL;
        for (uint32_t i = 0; i < tg->inc_projs->num_elements; i++)
                if (((struct projection *)
                        tg->inc_projs->elements[i])->to == fg) {
                        tg_to_fg = tg->inc_projs->elements[i];
                        break;
                }
        
        /* remove projection, if it exists */
        if (fg_to_tg && tg_to_fg) {
                remove_from_array(fg->out_projs, fg_to_tg);
                remove_from_array(tg->inc_projs, tg_to_fg);
                free_projection(fg_to_tg);
                free(tg_to_fg);
        } else {
                eprintf("Cannot remove projection - no projection between groups '%s' and '%s')\n",
                        arg1, arg2);
                return true;
        }

        mprintf("Removed projection \t [ %s -> %s ]\n", arg1, arg2);

        return true;
}

bool cmd_create_elman_projection(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* 'from' group name */
        char arg2[MAX_ARG_SIZE]; /* 'to' group name */
        if (sscanf(cmd, fmt, arg1, arg2) != 2)
                return false;

        /* find 'from' group */
        struct group *fg = find_array_element_by_name(s->anp->groups, arg1);
        if (fg == NULL) {
                eprintf("Cannot set Elman-projection - no such group '%s'\n",
                        arg1);
                return true;
        }
        
        /* find 'to' group */
        struct group *tg = find_array_element_by_name(s->anp->groups, arg2);
        if (tg == NULL) {
                eprintf("Cannot set Elman-projection - no such group '%s'\n",
                        arg2);
                return true;
        }

        /* projection should be recurrent */ 
        if (fg == tg) {
                eprintf("Cannot set Elman-projection - projection is recurrent for group '%s'\n",
                        fg->name);
                return true;
        }

        /* groups should not have unequal vector size */
        if (fg->vector->size != tg->vector->size) {
                eprintf("Cannot set Elman-projection - groups '%s' and '%s' have unequal vector sizes (%d and %d)\n",
                        fg->name, tg->name, fg->vector->size, tg->vector->size);
                return true;
        }

        /* Elman projection should not already exist */
        for (uint32_t i = 0; i < fg->ctx_groups->num_elements; i++) {
                if (fg->ctx_groups->elements[i] == tg) {
                        eprintf("Cannot set Elman-projection - Elman-projection '%s -> %s' already exists\n",
                                arg1, arg2);
                        return true;
                }
        }

        /* add Elman projection */
        add_to_array(fg->ctx_groups, tg);

        mprintf("Created Elman projection \t [ %s -> %s ]\n", arg1, arg2);

        return true;
}

bool cmd_remove_elman_projection(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* 'from' group name */
        char arg2[MAX_ARG_SIZE]; /* 'to' group name */
        if (sscanf(cmd, fmt, arg1, arg2) != 2)
                return false;

        /* find 'from' group */
        struct group *fg = find_array_element_by_name(s->anp->groups, arg1);
        if (fg == NULL) {
                eprintf("Cannot remove Elman-projection - no such group '%s'\n", arg1);
                return true;
        }

        /* find 'to' group */
        struct group *tg = find_array_element_by_name(s->anp->groups, arg2);
        if (tg == NULL) {
                eprintf("Cannot remove Elman-projection - no such group '%s'\n", arg2);
                return true;
        }

        /* remove Elman projection, if it exists */
        bool removed = false;
        for (uint32_t i = 0; i < fg->ctx_groups->num_elements; i++) {
                if (fg->ctx_groups->elements[i] == tg) {
                        remove_from_array(fg->ctx_groups, tg);
                        removed = true;
                        break;
                }
        }
        if (!removed) {
                eprintf("Cannot remove Elman-projection - no Elman projection from group '%s' to '%s'\n",
                        arg1, arg2);
                return true;
        }

        mprintf("Removed Elman projection \t [ %s -> %s ]\n", arg1, arg2);
        
        return true;

}

bool cmd_projections(char *cmd, char *fmt, struct session *s)
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        /*
         * List incoming, recurrent, and outgoing projections for each
         * group.
         */
        cprintf("Projections (by group) in network '%s':\n", s->anp->name);
        for (uint32_t i = 0; i < s->anp->groups->num_elements; i++) {
                struct group *g = s->anp->groups->elements[i];

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
        
        return true;
}

bool cmd_freeze_projection(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* 'from' group name */
        char arg2[MAX_ARG_SIZE]; /* 'to' group name */
        if (sscanf(cmd, fmt, arg1, arg2) != 2)
                return false;

        /* find 'from' group */
        struct group *fg = find_array_element_by_name(s->anp->groups, arg1);
        if (fg == NULL) {
                eprintf("Cannot freeze projection - no such group '%s'\n", arg1);
                return true;
        }

        /* find 'to' group */
        struct group *tg = find_array_element_by_name(s->anp->groups, arg2);
        if (tg == NULL) {
                eprintf("Cannot freeze projection - no such group '%s'\n", arg2);
                return true;
        }

        /* find outgoing 'from->to' projection */
        struct projection *fg_to_tg = NULL;
        for (uint32_t i = 0; i < fg->out_projs->num_elements; i++)
                if (((struct projection *)
                        fg->out_projs->elements[i])->to == tg) {
                        fg_to_tg = fg->out_projs->elements[i];
                        break;
                }

        /* find incoming 'to->from' projection */
        struct projection *tg_to_fg = NULL;
        for (uint32_t i = 0; i < tg->inc_projs->num_elements; i++)
                if (((struct projection *)
                        tg->inc_projs->elements[i])->to == fg) {
                        tg_to_fg = tg->inc_projs->elements[i];
                        break;
                }
        
        /* freeze projection, if it exists */
        if (fg_to_tg && tg_to_fg) {
                fg_to_tg->frozen = true;
                tg_to_fg->frozen = true;
        } else {
                eprintf("Cannot freeze projection - no projection between groups '%s' and '%s')\n",
                        arg1, arg2);
                return true;
        }

        mprintf("Froze projection \t\t [ %s -> %s ]\n", arg1, arg2);

        return true;
}

bool cmd_unfreeze_projection(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* 'from' group name */
        char arg2[MAX_ARG_SIZE]; /* 'to' group name */
        if (sscanf(cmd, fmt, arg1, arg2) != 2)
                return false;

        /* find 'from' group */
        struct group *fg = find_array_element_by_name(s->anp->groups, arg1);
        if (fg == NULL) {
                eprintf("Cannot unfreeze projection - no such group '%s'\n", arg1);
                return true;
        }

        /* find 'to' group */
        struct group *tg = find_array_element_by_name(s->anp->groups, arg2);
        if (tg == NULL) {
                eprintf("Cannot unfreeze projection - no such group '%s'\n", arg2);
                return true;
        }

        /* find outgoing 'from->to' projection */
        struct projection *fg_to_tg = NULL;
        for (uint32_t i = 0; i < fg->out_projs->num_elements; i++)
                if (((struct projection *)
                        fg->out_projs->elements[i])->to == tg) {
                        fg_to_tg = fg->out_projs->elements[i];
                        break;
                }

        /* find incoming 'to->from' projection */
        struct projection *tg_to_fg = NULL;
        for (uint32_t i = 0; i < tg->inc_projs->num_elements; i++)
                if (((struct projection *)
                        tg->inc_projs->elements[i])->to == fg) {
                        tg_to_fg = tg->inc_projs->elements[i];
                        break;
                }
        
        /* unfreeze projection, if it exists */
        if (fg_to_tg && tg_to_fg) {
                fg_to_tg->frozen = false;
                tg_to_fg->frozen = false;
        } else {
                eprintf("Cannot unfreeze projection - no projection between groups '%s' and '%s')\n",
                        arg1, arg2);
                return true;
        }

        mprintf("Unfroze projection \t\t [ %s -> %s ]\n", arg1, arg2);

        return true;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
This implements machinery for the "tunneling" of a subset of units of a
layer, allowing for the segmentation of a single input vector into multiple
ones:

        +---------+    +---------+    +---------+
        | output1 |    | output2 |    | output3 |
        +---------+    +---------+    +---------+
                 \          |           /
             +---------+---------+---------+
             |         : input0  :         |
             +---------+---------+---------+

and for the merging of several output vectors into a single vector:

             +---------+---------+---------+
             |         : output0 :         |
             +---------+---------+---------+
                 /          |           \
        +---------+    +---------+    +---------+
        | output1 |    | output2 |    | output3 |
        +---------+    +---------+    +---------+
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

/*
 * TODO: 
 * -- Check tunneling logic;
 * -- Remove tunnel projection.
 */
bool cmd_create_tunnel_projection(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* 'from' group name */
        uint32_t arg2;           /* 'from' group start unit */
        uint32_t arg3;           /* 'from' group end unit */
        char arg4[MAX_ARG_SIZE]; /* 'to' group name */
        uint32_t arg5;           /* 'to' group start unit */
        uint32_t arg6;           /* 'to' group start unit */
        if (sscanf(cmd, fmt,
                arg1, &arg2, &arg3,
                arg4, &arg5, &arg6) != 6)
                return false;

        /* find 'from' group */
        struct group *fg = find_array_element_by_name(s->anp->groups, arg1);
        if (fg == NULL) {
                eprintf("Cannot set tunnel projection - no such group '%s'\n", arg1);
                return true;
        }
        
        /* find 'to' group */
        struct group *tg = find_array_element_by_name(s->anp->groups, arg4);
        if (tg == NULL) {
                eprintf("Cannot set tunnel projection - no such group '%s'\n", arg4);
                return true;
        }

        /* 'from' and 'to' should be the same group */
        if (fg == tg) {
                eprintf("Cannot set recurrent tunnel projection\n");
                return true;
        }

        /*
         * The from group should not be a recurrent group, and there should
         * not already be a projection between the 'from' and 'to' group.
         */
        bool exists = false;
        if (fg->recurrent)
                exists = true;
        for (uint32_t i = 0; i < fg->out_projs->num_elements; i++)
                if (((struct projection *)
                        fg->out_projs->elements[i])->to == tg) {
                        exists = true;
                        break;
                }
        if (exists) {
                eprintf("Cannot set tunnel projection - projection '%s -> %s' already exists\n",
                        arg1, arg4);
                return true;
        }

        /* 'from' and 'to' ranges should not mismatch */
        if (arg3 - arg2 != arg6 - arg5) {
                eprintf("Cannot set tunnel projection - indices [%d:%d] and [%d:%d] cover differ ranges\n",
                        arg2, arg3, arg5, arg6);
                return true;
        }

        /* tunnel should be within 'from' group bounds */
        if (arg2 > fg->vector->size || arg3 > fg->vector->size || arg3 < arg2)
        {
                eprintf("Cannot set tunnel projection - indices [%d:%d] out of bounds\n",
                        arg2, arg3);
                return true;
        }

        /* tunnel should be within 'to' group bounds */
        if (arg5 > tg->vector->size || arg6 > tg->vector->size || arg6 < arg5)
        {
                eprintf("Cannot set tunnel projection - indices [%d:%d] out of bounds\n",
                        arg5, arg6);
                return true;
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
        
        /* freeze projections */
        op->frozen = true;
        ip->frozen = true;

        /* add projections */
        add_to_array(fg->out_projs, op);
        add_to_array(tg->inc_projs, ip);

        /* setup weights for tunneling */
        for (uint32_t r = arg2 - 1, c = arg5 - 1;
                r < arg3 && c < arg6;
                r++, c++) 
                weights->elements[r][c] = 1.0;

        mprintf("Created tunnel projection \t [ %s [%d:%d] -> %s [%d:%d] ]\n",
                arg1, arg2, arg3, arg4, arg5, arg6);

        return true;
}

bool cmd_toggle_reset_contexts(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        s->anp->reset_contexts = !s->anp->reset_contexts;

        if (s->anp->reset_contexts)
                mprintf("Toggled reset contexts \t [ on ]\n");
        else
                mprintf("Toggled reset contexts \t [ off ]\n");

        return true;
}

bool cmd_toggle_pretty_printing(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        s->pprint = !s->pprint;
        
        if (s->pprint)
                mprintf("Toggled pretty printing \t [ on ]\n");
        else
                mprintf("Toggled pretty printing \t [ off ]\n");

        return true;
}

bool cmd_set_color_scheme(char *cmd, char *fmt, struct session *s)
{
        char arg[MAX_ARG_SIZE]; /* color scheme */
        if (sscanf(cmd, fmt, arg) != 1)
                return false;

        /* blue and red */
        if (strcmp(arg, "blue_red") == 0)
                s->scheme = scheme_blue_red;
        /* blue and yellow */
        else if (strcmp(arg, "blue_yellow") == 0)
                s->scheme = scheme_blue_yellow;
        /* grayscale */
        else if (strcmp(arg, "grayscale") == 0)
                s->scheme = scheme_grayscale;
        /* spacepigs */
        else if (strcmp(arg, "spacepigs") == 0)
                s->scheme = scheme_spacepigs;
        /* moody blues */
        else if (strcmp(arg, "moody_blues") == 0)
                s->scheme = scheme_moody_blues;
        /* for John */
        else if (strcmp(arg, "for_john") == 0)
                s->scheme = scheme_for_john;
        /* gray and orange */
        else if (strcmp(arg, "gray_orange") == 0)
                s->scheme = scheme_gray_orange;
        else {
                eprintf("Cannot set color scheme - no such scheme '%s'\n", arg);
                return true;
        }

        mprintf("Set color scheme \t\t [ %s ]\n", arg);

        return true;
}

bool cmd_set_int_parameter(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* parameter */
        int32_t arg2;            /* value */
        if (sscanf(cmd, fmt, arg1, &arg2) != 2)
                return false;

        /* batch size */
        if (strcmp(arg1, "BatchSize") == 0) {
                s->anp->batch_size = arg2;
                mprintf("Set batch size \t\t [ %d ]\n",
                        s->anp->batch_size);
        /* max number of epochs */
        } else if (strcmp(arg1, "MaxEpochs") == 0) {
                s->anp->max_epochs = arg2;
                mprintf("Set maximum #epochs \t\t [ %d ]\n",
                        s->anp->max_epochs);
        /* report after */
        } else if (strcmp(arg1, "ReportAfter") == 0) {
                s->anp->report_after = arg2;
                mprintf("Set report after (#epochs) \t [ %d ]\n",
                        s->anp->report_after);
        /* random seed */
        } else if (strcmp(arg1, "RandomSeed") == 0) {
                s->anp->random_seed = arg2;
                mprintf("Set random seed \t\t [ %d ]\n",
                        s->anp->random_seed);
        /* number of back ticks */
        } else if (strcmp(arg1, "BackTicks") == 0) {
                s->anp->back_ticks = arg2;
                mprintf("Set BPTT back ticks \t\t [ %d ]\n",
                        s->anp->back_ticks);
        /* error: no matching variable */                        
        } else {
                return false;
        }

        return true;
}

bool cmd_set_double_parameter(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* parameter */
        double arg2;             /* value */
        if (sscanf(cmd, fmt, arg1, &arg2) != 2)
                return false;

        /* initial context activation */
        if (strcmp(arg1, "InitContextUnits") == 0) {
                s->anp->init_context_units = arg2;
                mprintf("Set init context units \t [ %lf ]\n",
                        s->anp->init_context_units);
        /* random mu */
        } else if (strcmp(arg1, "RandomMu") == 0) {
                s->anp->random_mu = arg2;
                mprintf("Set random Mu \t\t [ %lf ]\n",
                        s->anp->random_mu);
        /* random sigma */
        } else if (strcmp(arg1, "RandomSigma") == 0) {
                s->anp->random_sigma = arg2;
                mprintf("Set random Sigma \t\t [ %lf ]\n",
                        s->anp->random_sigma);
        /* random minimum */
        } else if (strcmp(arg1, "RandomMin") == 0) {
                s->anp->random_min = arg2;
                mprintf("Set random minimum \t\t [ %lf ]\n",
                        s->anp->random_min);
        /* random maximum */
        } else if (strcmp(arg1, "RandomMax") == 0) {
                s->anp->random_max = arg2;
                mprintf("Set random maximum \t\t [ %lf ]\n",
                        s->anp->random_max);
        /* learning rate */
        } else if (strcmp(arg1, "LearningRate") == 0) {
                s->anp->learning_rate = arg2;
                mprintf("Set learning rate \t\t [ %lf ]\n",
                        s->anp->learning_rate);
        /* learning rate scale factor */
        } else if (strcmp(arg1, "LRScaleFactor") == 0) {
                s->anp->lr_scale_factor = arg2;
                mprintf("Set LR scale factor \t\t [ %lf ]\n",
                        s->anp->lr_scale_factor);
        /* learning rate scale after */
        } else if (strcmp(arg1, "LRScaleAfter") == 0) {
                s->anp->lr_scale_after = arg2;
                mprintf("Set LR scale after (%%epochs) \t [ %lf ]\n",
                        s->anp->lr_scale_after);
        /* momentum */
        } else if (strcmp(arg1, "Momentum") == 0) {
                s->anp->momentum = arg2;
                mprintf("Set momentum \t\t\t [ %lf ]\n",
                        s->anp->momentum);
        /* momentum scale factor */
        } else if (strcmp(arg1, "MNScaleFactor") == 0) {
                s->anp->mn_scale_factor = arg2;
                mprintf("Set MN scale factor \t [ %lf ]\n",
                        s->anp->mn_scale_factor);
        /* momentum scale after */
        } else if (strcmp(arg1, "MNScaleAfter") == 0) {
                s->anp->mn_scale_after = arg2;
                mprintf("Set MN scale after (%%epochs) [ %lf ]\n",
                        s->anp->mn_scale_after);
        /* weight decay */
        } else if (strcmp(arg1, "WeightDecay") == 0) {
                s->anp->weight_decay = arg2;
                mprintf("Set weight decay \t\t [ %lf ]\n",
                        s->anp->weight_decay);
        /* weight decay scale factor */
        } else if (strcmp(arg1, "WDScaleFactor") == 0) {
                s->anp->wd_scale_factor = arg2;
                mprintf("Set WD scale factor \t [ %lf ]\n",
                        s->anp->wd_scale_factor);
        /* weight decay scale after */
        } else if (strcmp(arg1, "WDScaleAfter") == 0) {
                s->anp->wd_scale_after = arg2;
                mprintf("Set WD scale after (%%epochs) [ %lf ]\n",
                        s->anp->wd_scale_after);
        /* error threshold */
        } else if (strcmp(arg1, "ErrorThreshold") == 0) {
                s->anp->error_threshold = arg2;
                mprintf("Set error threshold \t\t [ %lf ]\n",
                        s->anp->error_threshold);
        /* target radius */
        } else if (strcmp(arg1, "TargetRadius") == 0) {
                s->anp->target_radius = arg2;
                mprintf("Set target radius \t\t [ %lf ]\n",
                        s->anp->target_radius);
        /* zero error radius */
        } else if (strcmp(arg1, "ZeroErrorRadius") == 0) {
                s->anp->zero_error_radius = arg2;
                mprintf("Set zero-error radius \t [ %lf ]\n",
                        s->anp->zero_error_radius);
        /* rprop initial update value */
        } else if (strcmp(arg1, "RpropInitUpdate") == 0) {
                s->anp->rp_init_update = arg2;
                mprintf("Set init update (for Rprop)  [ %lf ]\n",
                        s->anp->rp_init_update);
        /* rprop eta plus */
        } else if (strcmp(arg1, "RpropEtaPlus") == 0) {
                s->anp->rp_eta_plus = arg2;
                mprintf("Set Eta+ (for Rprop) \t [ %lf ]\n",
                        s->anp->rp_eta_plus);
        /* rprop eta minus */
        } else if (strcmp(arg1, "RpropEtaMinus") == 0) {
                s->anp->rp_eta_minus = arg2;
                mprintf("Set Eta- (for Rprop) \t [ %lf ]\n",
                        s->anp->rp_eta_minus);
        /* delta-bar-delta increment rate */
        } else if (strcmp(arg1, "DBDRateIncrement") == 0) {
                s->anp->dbd_rate_increment = arg2;
                mprintf("Set increment rate (for DBD) \t [ %lf ]\n",
                        s->anp->dbd_rate_increment);
        /* delta-bar-delta decrement rate */
        } else if (strcmp(arg1, "DBDRateDecrement") == 0) {
                s->anp->dbd_rate_decrement = arg2;
                mprintf("Set decrement rate (for DBD) \t [ %lf ]\n",
                        s->anp->dbd_rate_decrement);
        /* error: no matching variable */                        
        } else {
                return false;
        }        

        return true;
}

bool cmd_set_group_double_parameter(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* parameter */
        char arg2[MAX_ARG_SIZE]; /* group */
        double arg3;             /* value */
        if (sscanf(cmd, fmt, arg1, arg2, &arg3) != 3)
                return false;

        /* find group */
        struct group *g = find_array_element_by_name(s->anp->groups, arg2);
        if (g == NULL) {
                eprintf("Cannot set parameter '%s' - no such group '%s'\n", arg1, arg2);
                return true;
        }

        /* ReLU Alpha */
        if (strcmp(arg1, "ReLUAlpha") == 0) {
                g->relu_alpha = arg3;
                mprintf("Set ReLU alpha \t [ %s :: %lf ]\n",
                        arg2, g->relu_alpha);
        /* error: no matching variable */
        } else {
                return false;
        }  


        return true;
}

bool cmd_set_random_algorithm(char *cmd, char *fmt, struct session *s)
{
        char arg[MAX_ARG_SIZE]; /* training order */
        if (sscanf(cmd, fmt, arg) != 1)
                return false;

        /* gaussian randomization */
        if (strcmp(arg, "gaussian") == 0)
                s->anp->random_algorithm = randomize_gaussian;
        /* range randomization */
        else if (strcmp(arg, "range") == 0)
                s->anp->random_algorithm = randomize_range;
        /* Nguyen-Widrow randomization */
        else if (strcmp(arg, "nguyen_widrow") == 0)
                s->anp->random_algorithm = randomize_nguyen_widrow;
        /* fan-in method */
        else if (strcmp(arg, "fan_in") == 0)
                s->anp->random_algorithm = randomize_fan_in;
        /* binary randomization */
        else if (strcmp(arg, "binary") == 0)
                s->anp->random_algorithm = randomize_binary;
        else {
                eprintf("Invalid randomization algorithm '%s'\n", arg);
                return true;
        }

        mprintf("Set random algorithm \t\t [ %s ]\n", arg);

        return true;
}

bool cmd_set_learning_algorithm(char *cmd, char *fmt, struct session *s)
{
        char arg[MAX_ARG_SIZE]; /* learning algorithm */
        if (sscanf(cmd, fmt, arg) != 1)
                return false;

        /* backpropagation */
        if (strlen(arg) == 2 && strcmp(arg, "bp") == 0)
                s->anp->learning_algorithm = train_network_with_bp;
        /* backpropgation through time */
        else if (strlen(arg) == 4 && strcmp(arg, "bptt") == 0)
                s->anp->learning_algorithm = train_network_with_bptt;
        else {
                eprintf("Invalid learning algorithm '%s'\n", arg);
                return true;
        }
        
        mprintf("Set learning algorithm \t [ %s ]\n", arg);

        return true;
}

bool cmd_set_update_algorithm(char *cmd, char *fmt, struct session *s)
{
        char arg[MAX_ARG_SIZE]; /* update algorithm */
        if (sscanf(cmd, fmt, arg) != 1)
                return false;

        /* steepest descent */
        if (strcmp(arg, "steepest") == 0) {
                s->anp->update_algorithm = bp_update_sd;
                s->anp->sd_type = SD_DEFAULT;
        }
        /* gradient [= steepest] descent */
        else if (strcmp(arg, "gradient") == 0) {
                s->anp->update_algorithm = bp_update_sd;
                s->anp->sd_type = SD_DEFAULT;
        }        
        /* bounded steepest descent */
        else if (strcmp(arg, "bounded") == 0) {
                s->anp->update_algorithm = bp_update_sd;
                s->anp->sd_type = SD_BOUNDED;
        }
        /* resilient propagation plus */
        else if (strcmp(arg, "rprop+") == 0) {
                s->anp->update_algorithm = bp_update_rprop;
                s->anp->rp_type = RPROP_PLUS;
        }
        /* resilient propagation minus */
        else if (strcmp(arg, "rprop-") == 0) {
                s->anp->update_algorithm = bp_update_rprop;
                s->anp->rp_type = RPROP_MINUS;
        }
        /* modified resilient propagation plus */
        else if (strcmp(arg, "irprop+") == 0) {
                s->anp->update_algorithm = bp_update_rprop;
                s->anp->rp_type = IRPROP_PLUS;
        }
        /* modified resilient propagation minus */
        else if (strcmp(arg, "irprop-") == 0) {
                s->anp->update_algorithm = bp_update_rprop;
                s->anp->rp_type = IRPROP_MINUS;
        }
        /* quickprop */
        else if (strcmp(arg, "qprop") == 0)
                s->anp->update_algorithm = bp_update_qprop;
        /* delta-bar-delta */
        else if (strcmp(arg, "dbd") == 0)
                s->anp->update_algorithm = bp_update_dbd;
        else {
                eprintf("Invalid update algorithm '%s'\n", arg);
                return true;
        }

        mprintf("Set update algorithm \t\t [ %s ]\n", arg);

        return true;
}

bool cmd_set_similarity_metric(char *cmd, char *fmt, struct session *s)
{
        char arg[MAX_ARG_SIZE]; /* similarity metric */
        if (sscanf(cmd, fmt, arg) != 1)
                return false;

        /* inner product */
        if (strcmp(arg, "inner_product") == 0)
                s->anp->similarity_metric = inner_product;
        /* harmonic mean */
        else if (strcmp(arg, "harmonic_mean") == 0)
                s->anp->similarity_metric = harmonic_mean;
        /* cosine similarity */
        else if (strcmp(arg, "cosine") == 0)
                s->anp->similarity_metric = cosine;
        /* tanimoto */
        else if (strcmp(arg, "tanimoto") == 0)
                s->anp->similarity_metric = tanimoto;
        /* dice */
        else if (strcmp(arg, "dice") == 0)
                s->anp->similarity_metric = dice;
        /* pearson correlation */
        else if (strcmp(arg, "pearson_correlation") == 0)
                s->anp->similarity_metric = pearson_correlation;
        else {
                eprintf("Invalid similarity metric '%s'\n", arg);
                return true;
        }

        mprintf("Set similarity metric \t [ %s ]\n", arg);

        return true;
}

bool cmd_set_training_order(char *cmd, char *fmt, struct session *s)
{
        char arg[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, arg) != 1)
                return false;

        /* ordered */
        if (strcmp(arg, "ordered") == 0)
                s->anp->training_order = train_ordered;
        /* permuted */
        else if (strcmp(arg, "permuted") == 0)    
                s->anp->training_order = train_permuted;
        /* randomized */
        else if (strcmp(arg, "randomized") == 0)
                s->anp->training_order = train_randomized;
        else {
                eprintf("Invalid training order '%s'\n", arg);
                return true;
        }

        mprintf("Set training order \t\t [ %s ]\n", arg);

        return true;
}

bool cmd_weight_stats(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        struct weight_stats *ws = create_weight_statistics(s->anp);
        print_weight_statistics(s->anp, ws);
        free_weight_statistics(ws);

        return true;
}

bool cmd_save_weights(char *cmd, char *fmt, struct session *s)
{
        char arg[MAX_ARG_SIZE]; /* filename */
        if (sscanf(cmd, fmt, arg) != 1)
                return false;

        if (save_weight_matrices(s->anp, arg))
                mprintf("Saved weights \t\t [ %s ]\n", arg);
        else
                eprintf("Cannot save weights to file '%s'\n", arg);

        return true;
}

bool cmd_load_weights(char *cmd, char *fmt, struct session *s)
{
        char arg[MAX_ARG_SIZE]; /* filename */
        if (sscanf(cmd, fmt, arg) != 1)
                return false;

        if (load_weight_matrices(s->anp, arg))
                mprintf("Loaded weights \t\t [ %s ]\n", arg);
        else
                eprintf("Cannot load weights from file '%s'\n", arg);

        return true;
}

bool cmd_show_vector(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* vector type */
        char arg2[MAX_ARG_SIZE]; /* group name */
        if (sscanf(cmd, fmt, arg1, arg2) != 2)
                return false;

        /* vector types */
        enum vector_type
        {
                vtype_units,
                vtype_error
        } type;
        if (strcmp(arg1, "units") == 0) {               /* units */
                type = vtype_units;
        } else if (strcmp(arg1, "error") == 0) {        /* error */
                type = vtype_error;
        } else {
                eprintf("Cannot show vector - no such vector type '%s'\n",
                        arg1);
                return true;
        }

        /* find group */
        struct group *g = find_array_element_by_name(s->anp->groups, arg2);
        if (g == NULL) {
                eprintf("Cannot show vector - no such group '%s'\n", arg2);
                return true;
        }

        cprintf("\n");
        switch (type) {
        case vtype_units:
                cprintf("Unit vector for '%s':\n\n", arg2);
                s->pprint ? pprint_vector(g->vector, s->scheme)
                          : print_vector(g->vector);
                break;
        case vtype_error:
                cprintf("Error vector for '%s':\n\n", arg2);
                s->pprint ? pprint_vector(g->error, s->scheme)
                          : print_vector(g->error);
                break;
        }
        cprintf("\n");

        return true;
}

bool cmd_show_matrix(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* matrix type */
        char arg2[MAX_ARG_SIZE]; /* 'from' group name */
        char arg3[MAX_ARG_SIZE]; /* 'to' group name */
        if (sscanf(cmd, fmt, arg1, arg2, arg3) != 3)
                return false;

        /* matrix type */
        enum matrix_type
        {
                mtype_weights,
                mtype_gradients,
                mtype_dynamic_params
        } type;
        if (strcmp(arg1, "weights") == 0)               /* weights */
                type = mtype_weights;
        else if (strcmp(arg1, "gradients") == 0)        /* gradients */
                type = mtype_gradients;
        else if (strcmp(arg1, "dynamics") == 0)         /* dynamics */
                type = mtype_dynamic_params;
        else {
                eprintf("Cannot show matrix - no such matrix type '%s'\n",
                        arg1);
                return false;
        }

        /* find 'from' group */
        struct group *fg = find_array_element_by_name(s->anp->groups, arg2);
        if (fg == NULL) {
                eprintf("Cannot show matrix - no such group '%s'\n", arg2);
                return true;
        }

        /* find 'to' group */
        struct group *tg = find_array_element_by_name(s->anp->groups, arg3);
        if (tg == NULL) {
                eprintf("Cannot show matrix - no such group '%s'\n", arg3);
                return true;
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
                eprintf("Cannot show matrix - no projection between groups '%s' and '%s'\n",
                        arg2, arg3);
                return true;
        }

        cprintf("\n");
        switch(type) {
        case mtype_weights:
                cprintf("Weight matrix for projection '%s -> %s':\n\n",
                        arg2, arg3);
                s->pprint ? pprint_matrix(fg_to_tg->weights, s->scheme)
                          : print_matrix(fg_to_tg->weights);
                break;
        case mtype_gradients:
                cprintf("Gradient matrix for projection '%s -> %s':\n\n",
                        arg2, arg3);
                s->pprint ? pprint_matrix(fg_to_tg->gradients, s->scheme)
                          : print_matrix(fg_to_tg->gradients);
                break;
        case mtype_dynamic_params:
                cprintf("Dynamic learning parameters for projection '%s -> %s':\n\n",
                        arg2, arg3);
                s->pprint ? pprint_matrix(fg_to_tg->dynamic_params, s->scheme)
                          : print_matrix(fg_to_tg->dynamic_params);
                break;
        }
        cprintf("\n");

        return true;
}

bool cmd_load_set(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* set name */
        char arg2[MAX_ARG_SIZE]; /* set filename */
        if (sscanf(cmd, fmt, arg1, arg2) != 2)
                return false;

        /* input group size should be known */
        if (!s->anp->input) {
                eprintf("Cannot load set - input group size unknown\n");
                return true;
        }
        
        /* output group size should be known */
        if (!s->anp->output) {
                eprintf("Cannot load set - output group size unknown\n");
                return true;
        }

        /* set should not already exist */
        if (find_array_element_by_name(s->anp->sets, arg1)) {
                eprintf("Cannot load set - set '%s' already exists\n", arg1);
                return true;
        }

        /* load set, if it exists */
        struct set *set = load_set(arg1, arg2,
                s->anp->input->vector->size,
                s->anp->output->vector->size);
        if (!set) {
                eprintf("Cannot load set - no such file '%s'\n", arg2);
                return true;
        }
        
        /* add set to active network */
        add_to_array(s->anp->sets, set);
        s->anp->asp = set;

        mprintf("Loaded set \t\t\t [ %s => %s (%d) ]\n", arg2, set->name,
                set->items->num_elements);

        return true;
}

bool cmd_remove_set(char *cmd, char *fmt, struct session *s)
{
        char arg[MAX_ARG_SIZE]; /* set name */
        if (sscanf(cmd, fmt, arg) != 1)
                return false;

        /* find set */
        struct set *set = find_array_element_by_name(s->anp->sets, arg);
        if (!set) {
                eprintf("Cannot change to set - no such set '%s'\n", arg);
                return true;
        }

        /*
         * If the set to be removed is the active set, try finding another
         * active set.
         */
        if (set == s->anp->asp) {
                s->anp->asp = NULL;
                for (uint32_t i = 0; i < s->anp->sets->num_elements; i++)
                        if (s->anp->sets->elements[i] != NULL
                                && s->anp->sets->elements[i] != set)
                                s->anp->asp = s->anp->sets->elements[i];
        }

        /* remove set */
        remove_from_array(s->anp->sets, set);
        free_set(set);

        mprintf("Removed set \t\t [ %s ]\n", arg);

        return true;
}

bool cmd_sets(char *cmd, char *fmt, struct session *s)
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        cprintf("Sets in network '%s':\n", s->anp->name);
        if (s->anp->sets->num_elements == 0) {
                cprintf("(no sets)\n");
        } else {
                for (uint32_t i = 0; i < s->anp->sets->num_elements; i++) {
                        struct set *set = s->anp->sets->elements[i];
                        cprintf("* %d: %s (%d)", i + 1, set->name, set->items->num_elements);
                        if (set == s->anp->asp)
                                cprintf(" :: active set\n");
                        else
                                cprintf("\n");
                }
        }

        return true;
}

bool cmd_change_set(char *cmd, char *fmt, struct session *s)
{
        char arg[MAX_ARG_SIZE]; /* set name */
        if (sscanf(cmd, fmt, arg) != 1)
                return false;

        /* find set */
        struct set *set = find_array_element_by_name(s->anp->sets, arg);
        if (!set) {
                eprintf("Cannot change to set - no such set '%s'\n", arg);
                return true;
        }
        s->anp->asp = set;

        mprintf("Changed to set \t\t [ %s ]\n", arg);

        return true;
}

bool cmd_init(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        init_network(s->anp);

        if (s->anp->initialized)
                mprintf("Initialized network \t\t [ %s ]\n", s->anp->name);

        return true;
}

bool cmd_reset(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        reset_network(s->anp);

        mprintf("Reset network '%s'\n", s->anp->name);

        return true;
}

bool cmd_train(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf("Training network '%s'\n", s->anp->name);

        train_network(s->anp);

        return true;
}

bool cmd_test_item(char *cmd, char *fmt, struct session *s)
{
        char arg[MAX_ARG_SIZE]; /* item name */
        if (sscanf(cmd, fmt, arg) != 1)
                return false;

        /* find item */
        struct item *item = find_array_element_by_name(s->anp->asp->items, arg);
        if (!item) {
                eprintf("Cannot test network - no such item '%s'\n", arg);
                return true;
        }

        mprintf("Testing network '%s' with item '%s'\n", s->anp->name, arg);

        test_network_with_item(s->anp, item, s->pprint, s->scheme);

        return true;
}

bool cmd_test_item_no(char *cmd, char *fmt, struct session *s)
{
        uint32_t arg; /* item number */
        if (sscanf(cmd, fmt, &arg) != 1)
                return false;

        /* find item */
        if (arg == 0 || arg > s->anp->asp->items->num_elements) {
                eprintf("Cannot test network - no such item number '%d'\n", arg);
                return true;
        }
        struct item *item = s->anp->asp->items->elements[arg - 1];
        
        mprintf("Testing network '%s' with item '%s'\n", s->anp->name, item->name);

        test_network_with_item(s->anp, item, s->pprint, s->scheme);

        return true;
}

bool cmd_test(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf("Testing network '%s'\n", s->anp->name);

        test_network(s->anp, false);

        return true;
}

bool cmd_test_verbose(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf("Testing network '%s'\n", s->anp->name);

        test_network(s->anp, true);

        return true;
}

bool cmd_items(char *cmd, char *fmt, struct session *s)
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        /* there should be an active set */
        if (!s->anp->asp) {
                eprintf("Cannot list items - no active set\n");
                return true;
        }

        cprintf("Items in set '%s' of network '%s':\n",
                s->anp->asp->name, s->anp->name);
        for (uint32_t i = 0; i < s->anp->asp->items->num_elements; i++) {
                struct item *item = s->anp->asp->items->elements[i];
                cprintf("* %d: \"%s\" %d \"%s\"\n", i + 1,
                        item->name, item->num_events, item->meta);
        }

        return true;
}

bool cmd_show_item(char *cmd, char *fmt, struct session *s)
{
        char arg[MAX_ARG_SIZE]; /* item name */
        if (sscanf(cmd, fmt, arg) != 1)
                return false;

        /* find item */
        struct item *item = find_array_element_by_name(s->anp->asp->items, arg);
        if (!item) {
                eprintf("Cannot show item - no such item '%s'\n", arg);
                return true;
        }

        cprintf("\n");
        cprintf("Name:   \"%s\"\n", item->name);
        cprintf("Meta:   \"%s\"\n", item->meta);
        cprintf("Events: %d\n", item->num_events);
        cprintf("\n");
        cprintf("(E: Event; I: Input; T: Target)\n");
        for (uint32_t i = 0; i < item->num_events; i++) {
                cprintf("\n");
                cprintf("E: %d\n", i + 1);
                cprintf("I: ");
                s->pprint ? pprint_vector(item->inputs[i], s->scheme)
                          : print_vector(item->inputs[i]);
                if (item->targets[i]) {
                        cprintf("T: ");
                        s->pprint ? pprint_vector(item->targets[i], s->scheme)
                                  : print_vector(item->targets[i]);
                }
        }
        cprintf("\n");

        return true;
}
 
bool cmd_record_units(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* group name */
        char arg2[MAX_ARG_SIZE]; /* filename */
        if (sscanf(cmd, fmt, arg1, arg2) != 2)
                return false;

        /* find group */
        struct group *g = find_array_element_by_name(s->anp->groups, arg1);
        if (g == NULL) {
                eprintf("Cannot record units - no such group '%s'\n", arg1);
                return true;
        }

        mprintf("Recording units of group '%s' in '%s'\n", g->name, s->anp->name);

        record_units(s->anp, g, arg2);

        mprintf("Written activation vectors \t [ %s ]\n", arg2);

        return true;
}

/*
 * TODO: Document this.
 */
bool cmd_set_multi_stage(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* multi-stage input group name */
        char arg2[MAX_ARG_SIZE]; /* multi-stage input set name */
        if (sscanf(cmd, fmt, arg1, arg2) != 2)
                return false;

        /* find multi-stage input group */
        struct group *g = find_array_element_by_name(s->anp->groups, arg1);
        if (g == NULL) {
                eprintf("Cannot set multi-stage training-no such group '%s'\n",
                        arg1);
                return true;
        }

        /* find multi-stage input set */
        struct set *set = find_array_element_by_name(s->anp->sets, arg2);
        if (!set) {
                eprintf("Cannot set multi-stage training - no such set '%s'\n",
                        arg2);
                return true;
        }

        s->anp->ms_input = g;
        s->anp->ms_set   = set;

        mprintf("Set multi-stage training \t [ %s --> %s :: %s ==> %s ]\n", 
                s->anp->input->name, s->anp->ms_input->name,
                s->anp->ms_set->name, s->anp->output->name);

        return true;
}

bool cmd_set_single_stage(char *cmd, char *fmt, struct session *s)
{
        s->anp->ms_input = NULL;
        s->anp->ms_set   = NULL;

        mprintf("Set single-stage training \t [ %s --> %s ]\n", 
                s->anp->input->name, s->anp->output->name);
        
        return true;
}

bool cmd_similarity_matrix(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf("Computing similarity matrix for network '%s'\n", s->anp->name);

        struct matrix *sm = similarity_matrix(s->anp);
        print_sm_summary(s->anp, sm, true, s->pprint, s->scheme);
        free_matrix(sm);

        return true;
}

bool cmd_similarity_stats(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf("Computing similarity matrix for network '%s'\n", s->anp->name);

        struct matrix *sm = similarity_matrix(s->anp);
        print_sm_summary(s->anp, sm, false, s->pprint, s->scheme);
        free_matrix(sm);

        return true;
}

bool cmd_confusion_matrix(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf("Computing confusion matrix for network '%s'\n", s->anp->name);

        struct matrix *cm = confusion_matrix(s->anp);
        print_cm_summary(cm, true, s->pprint, s->scheme);
        free_matrix(cm);

        return true;
}

bool cmd_confusion_stats(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf("Computing confusion matrix for network '%s'\n", s->anp->name);

        struct matrix *cm = confusion_matrix(s->anp);
        print_cm_summary(cm, false,  s->pprint, s->scheme);
        free_matrix(cm);

        return true;
}

                /********************************************
                 **** distributed-situation state spaces ****
                 ********************************************/

bool cmd_dss_test(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf("Testing network '%s':\n", s->anp->name);

        dss_test(s->anp);

        return true;
}

bool cmd_dss_scores(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* set name */
        char arg2[MAX_ARG_SIZE]; /* item name */
        if (sscanf(cmd, fmt, arg1, arg2) != 2)
                return false;

        /* find set */
        struct set *set = find_array_element_by_name(s->anp->sets, arg1);
        if (!set) {
                eprintf("Cannot compute scores - no such set '%s'\n", arg1);
                return true;
        }

        /* find item */
        struct item *item = find_array_element_by_name(s->anp->asp->items,
                arg2);
        if (!item) {
                eprintf("Cannot compute scores - no such item '%s'\n", arg2);
                return true;
        }

        dss_scores(s->anp, set, item);

        return true;
}

bool cmd_dss_inferences(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* set name */
        char arg2[MAX_ARG_SIZE]; /* item name */
        double arg3;             /* inference score threshold */
        if (sscanf(cmd, fmt, arg1, arg2, &arg3) != 3)
                return false;

        /* find set */
        struct set *set = find_array_element_by_name(s->anp->sets, arg1);
        if (!set) {
                eprintf("Cannot compute inferences - no such set '%s'\n",
                        arg1);
                return true;
        }

        /* find item */
        struct item *item = find_array_element_by_name(s->anp->asp->items,
                arg2);
        if (!item) {
                eprintf("Cannot compute inferences - no such item '%s'\n",
                        arg2);
                return true;
        }

        /* require a proper threshold */
        if (arg3 < -1.0 || arg3 > 1.0) {
                eprintf("Cannot compute inferences - invalid score threshold '%lf'\n",
                        arg3);
                return true;

        }

        dss_inferences(s->anp, set, item, arg3);

        return true;
}

bool cmd_dss_word_info(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* set name */
        char arg2[MAX_ARG_SIZE]; /* item name */
        if (sscanf(cmd, fmt, arg1, arg2) != 2)
                return false;

        /* find set */
        struct set *set = find_array_element_by_name(s->anp->sets, arg1);
        if (!set) {
                eprintf("Cannot compute informativity metrics - no such set '%s'\n",
                        arg1);
                return true;
        }
        
        /* find item */
        struct item *item = find_array_element_by_name(s->anp->asp->items, arg2);
        if (!item) {
                eprintf("Cannot compute informativity metrics - no such item '%s'\n",
                        arg2);
                return true;
        }
        
        mprintf("Testing network '%s' with item '%s':\n", s->anp->name, arg2);

        dss_word_info(s->anp, set, item);

        return true;
}

bool cmd_dss_write_word_info(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* set name */
        char arg2[MAX_ARG_SIZE]; /* filename */
        if (sscanf(cmd, fmt, arg1, arg2) != 2)
                return false;

        /* find set */
        struct set *set = find_array_element_by_name(s->anp->sets, arg1);
        if (!set) {
                eprintf("Cannot compute informativity metrics - no such set '%s'\n",
                        arg1);
                return true;
        }

        mprintf("Computing word informativity metrics \t [ %s :: %s ]\n",
                s->anp->asp->name, arg1);

        dss_write_word_info(s->anp, set, arg2);

        mprintf("Written word informativity metrics \t [ %s ]\n", arg2);

        return true;
}

                /**********************************
                 **** event-related potentials ****
                 **********************************/

bool cmd_erp_contrast(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* group name */
        char arg2[MAX_ARG_SIZE]; /* 'control' item name */
        char arg3[MAX_ARG_SIZE]; /* 'target' item name */
        if (sscanf(cmd, fmt, arg1, arg2, arg3) != 3)
                return false;
        
        /* find group */
        struct group *gen = find_array_element_by_name(s->anp->groups,
                        arg1);
        if (gen == NULL) {
                eprintf("Cannot compute ERP correlates - no such group '%s'\n",
                        arg1);
                return true;
        }

        /* find 'control' item */
        struct item *item1 = find_array_element_by_name(s->anp->asp->items,
                arg2);
        if (!item1) {
                eprintf("Cannot compute ERP correlates - no such item '%s'\n",
                        arg2);
                return true;
        }

        /* find 'target' item */
        struct item *item2 = find_array_element_by_name(s->anp->asp->items,
                arg3);
        if (!item2) {
                eprintf("Cannot compute ERP correlates - no such item '%s'\n",
                        arg3);
                return true;
        }

        erp_contrast(s->anp, gen, item1, item2);

        return true;
}

bool cmd_erp_write_values(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* N400 generator group name */
        char arg2[MAX_ARG_SIZE]; /* P600 generator group name */
        char arg3[MAX_ARG_SIZE]; /* filename */
        if (sscanf(cmd, fmt, arg1, arg2, arg3) != 3)
                return false;

        /* find 'N400 generator' group */
        struct group *N400_gen = find_array_element_by_name(s->anp->groups,
                arg1);
        if (N400_gen == NULL) {
                eprintf("Cannot compute ERP correlates - no such group '%s'\n",
                        arg1);
                return true;
        }

        /* find 'P600 generator' group */
        struct group *P600_gen = find_array_element_by_name(s->anp->groups,
                arg2);
        if (P600_gen == NULL) {
                eprintf("Cannot compute ERP correlates - no such group '%s'\n",
                        arg2);
                return true;
        }

        mprintf("Computing ERP estimates \t [ N400 :: %s | P600 :: %s ]\n",
                arg1, arg2);

        erp_write_values(s->anp, N400_gen, P600_gen, arg3);

        mprintf("Written ERP estimates \t [ %s ]\n", arg3);

        return true;
}
