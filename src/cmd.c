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
#include "engine.h"
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
                        eprintf("No active network - see `help networks`\n");
                        goto out;
                }
                /* 
                 * Skip commands that require an initialized network if
                 * necessary.
                 */
                if (req_init && !s->anp->flags->initialized) {
                        eprintf("Cannot process command: `%s`\n", cmd);
                        eprintf("Uninitialized network - use `init` command to initialize\n");
                        goto out;
                }
                /*
                 * Skip commands that require an active example set if
                 * necessary. Also, an active example set needs to have the
                 * same dimensionality as the network.
                 */
                if (req_asp) {
                        if (!s->anp->asp) {
                                eprintf("Cannot process command: `%s`\n", cmd);
                                eprintf("No active set - see `help sets`\n");
                                goto out;
                        }
                        struct item *item = s->anp->asp->items->elements[0];
                        if (s->anp->input->vector->size != item->inputs[0]->size) {
                                eprintf("Cannot process command: `%s`\n", cmd);
                                eprintf("Input dimensionality mismatch: model (%d) != set (%d)\n",
                                        s->anp->input->vector->size, item->inputs[0]->size);
                                goto out;
                        }
                        if (s->anp->output->vector->size != item->targets[0]->size) {
                                eprintf("Cannot process command: `%s`\n", cmd);
                                eprintf("Output dimensionality mismatch: model (%d) != set (%d)\n",
                                        s->anp->input->vector->size, item->inputs[0]->size);
                                goto out;
                        }
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
                eprintf("Type `help` for help\n");
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
                if (sscanf(cmd, fmt, arg) != 1)
                        return false;
                help_on_topic = true;
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
        if      (strcmp(arg2, "ffn") == 0) type = ntype_ffn;
         /* simple recurrent network */
        else if (strcmp(arg2, "srn") == 0) type = ntype_srn;
        /* recurrent network */
        else if (strcmp(arg2, "rnn") == 0) type = ntype_rnn;
        else {
                eprintf("Cannot create network - invalid network type: '%s'\n", arg2);
                return true;
        }
        /* network should not already exist */
        if (find_array_element_by_name(s->networks, arg1)) {
                eprintf("Cannot create network - network '%s' already exists\n", arg1);
                return true;
        }
        /* create network, and set as active */
        struct network *n = create_network(arg1, type);
        add_network(s, n);
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
        remove_network(s, n);
        mprintf("Removed network \t\t [ %s ]\n", arg);
        return true;
}

bool cmd_networks(char *cmd, char *fmt, struct session *s)
{
        if (strcmp(cmd, fmt) != 0)
                return false;
        print_networks(s);
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
        inspect_network(s->anp);
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
                eprintf("Cannot create group - group '%s' already exists\n", arg1);
                return true;
        }
        /* group size should be positive */
        if (!(arg2 > 0)) {
                eprintf("Cannot create group - group size should be positive\n");
                return true;
        }
        struct group *g = create_group(arg1, arg2, false, false);
        add_group(s->anp, g);
        mprintf("Created group \t\t [ %s :: %d ]\n", arg1, arg2);
        return true;
}

bool cmd_create_bias_group(char *cmd, char *fmt, struct session *s)
{
        char arg[MAX_ARG_SIZE]; /* group name */
        if (sscanf(cmd, fmt, arg) != 1)
                return false;
        /* group should not already exist */
        if (find_array_element_by_name(s->anp->groups, arg)) {
                eprintf("Cannot create group - group '%s' already exists\n", arg);
                return true;
        }
        struct group *bg = create_bias_group(arg);
        add_group(s->anp, bg);
        mprintf("Created bias group \t\t [ %s ]\n", arg);
        return true;        
}

bool cmd_create_dcs_group(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* DCS group name */
        char arg2[MAX_ARG_SIZE]; /* DCS set name */
        if (sscanf(cmd, fmt, arg1, arg2) != 2)
                return false;
        /* find DCS context set */
        struct set *set = find_array_element_by_name(s->anp->sets, arg2);
        if (!set) {
                eprintf("Cannot create DCS group - no such set '%s'\n", arg2);
                return true;
        }
        /* create DCS context group */
        struct group *g = create_group(arg1, set->items->num_elements, false, false);
        g->pars->dcs_set = set;
        add_group(s->anp, g);
        /* enable DCS */
        s->anp->flags->dcs = true;
        mprintf("Created DCS group \t\t [ (%s :: %s) <-- %s ]\n",
                g->name, set->name, s->anp->output->name);                
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
        remove_group(s->anp, g);
        mprintf("Removed group \t\t [ %s ]\n", arg);
        return true;
}

bool cmd_groups(char *cmd, char *fmt, struct session *s)
{
        if (strcmp(cmd, fmt) != 0)
                return false;
        print_groups(s->anp);
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
        struct group *bg = attach_bias_group(s->anp, g);
        bg ? mprintf("Attached bias to group \t [ %s -> %s ]\n", bg->name, g->name)
           : eprintf("Cannot attach bias group - bias already exists\n");
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
        if      (strcmp(arg2, "logistic") == 0) {
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
        if      (strcmp(arg2, "sum_of_squares") == 0) {
                g->err_fun->fun   = err_fun_sum_of_squares;
                g->err_fun->deriv = err_fun_sum_of_squares_deriv;
        }
        /* sum of squares */
        else if (strcmp(arg2, "sum_squares") == 0) {
                g->err_fun->fun   = err_fun_sum_of_squares;
                g->err_fun->deriv = err_fun_sum_of_squares_deriv;
        }        
        /* cross-entropy */
        else if (strcmp(arg2, "cross_entropy") == 0) {
                g->err_fun->fun   = err_fun_cross_entropy;
                g->err_fun->deriv = err_fun_cross_entropy_deriv;
        }
        /* divergence */
        else if (strcmp(arg2, "divergence") == 0) {
                g->err_fun->fun   = err_fun_divergence;
                g->err_fun->deriv = err_fun_divergence_deriv;
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
                eprintf("Cannot create projection - no such group '%s'\n", arg1);
                return true;
        }
        /* find 'to' group */
        struct group *tg = find_array_element_by_name(s->anp->groups, arg2);
        if (tg == NULL) {
                eprintf("Cannot create projection - no such group '%s'\n", arg2);
                return true;
        }
        /* projection should not already exist */
        if (find_projection(fg->out_projs, tg)) {
        
                eprintf("Cannot create projection - projection '%s -> %s' already exists\n", arg1, arg2);
                return true;
        }
        add_bidirectional_projection(fg, tg);
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
        /* find projections */
        struct projection *fg_to_tg = find_projection(fg->out_projs, tg);
        struct projection *tg_to_fg = find_projection(tg->inc_projs, fg);
        if (!fg_to_tg || !tg_to_fg) {
                eprintf("Cannot remove projection - no projection between groups '%s' and '%s')\n",
                        arg1, arg2);
                return true;
        }
        remove_bidirectional_projection(fg, fg_to_tg, tg, tg_to_fg);
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
                eprintf("Cannot set Elman-projection - no such group '%s'\n", arg1);
                return true;
        }
        /* find 'to' group */
        struct group *tg = find_array_element_by_name(s->anp->groups, arg2);
        if (tg == NULL) {
                eprintf("Cannot set Elman-projection - no such group '%s'\n", arg2);
                return true;
        }
        /* projection should be recurrent */ 
        if (fg == tg) {
                eprintf("Cannot set Elman-projection - projection is recurrent for group '%s'\n", fg->name);
                return true;
        }
        /* groups should not have unequal vector size */
        if (fg->vector->size != tg->vector->size) {
                eprintf("Cannot set Elman-projection - groups '%s' and '%s' have unequal vector sizes (%d and %d)\n",
                        fg->name, tg->name, fg->vector->size, tg->vector->size);
                return true;
        }
        /* Elman projection should not already exist */
        if (find_elman_projection(fg, tg)) {
                eprintf("Cannot set Elman-projection - Elman-projection '%s -> %s' already exists\n", arg1, arg2);
                return true;
        }
        /* add Elman projection */
        add_elman_projection(fg, tg);
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
        if (find_elman_projection(fg, tg)) {
                remove_elman_projection(fg, tg);
        } else {
                eprintf("Cannot remove Elman-projection - no Elman projection from group '%s' to '%s'\n", arg1, arg2);
                return true;
        }
        mprintf("Removed Elman projection \t [ %s -> %s ]\n", arg1, arg2);
        return true;
}

bool cmd_projections(char *cmd, char *fmt, struct session *s)
{
        if (strcmp(cmd, fmt) != 0)
                return false;
        print_projections(s->anp);
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
        /* freeze projection, if it exists */
        struct projection *fg_to_tg = find_projection(fg->out_projs, tg);
        if (!fg_to_tg) {
                eprintf("Cannot freeze projection - no projection between groups '%s' and '%s')\n",
                        arg1, arg2);
                return true;
        }
        freeze_projection(fg_to_tg);
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
        /* unfreeze projection, if it exists */
        struct projection *fg_to_tg = find_projection(fg->out_projs, tg);
        if (!fg_to_tg) {
                eprintf("Cannot unfreeze projection - no projection between groups '%s' and '%s')\n",
                        arg1, arg2);
                return true;
        }
        unfreeze_projection(fg_to_tg);
        mprintf("Unfroze projection \t\t [ %s -> %s ]\n", arg1, arg2);
        return true;
}

bool cmd_toggle_reset_contexts(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;
        s->anp->flags->reset_contexts = !s->anp->flags->reset_contexts;
        if (s->anp->flags->reset_contexts)
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
        if      (strcmp(arg, "blue_red") == 0)
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
        if        (strcmp(arg1, "BatchSize") == 0) {
                s->anp->pars->batch_size = arg2;
                mprintf("Set batch size \t\t [ %d ]\n",
                        s->anp->pars->batch_size);
        /* max number of epochs */
        } else if (strcmp(arg1, "MaxEpochs") == 0) {
                s->anp->pars->max_epochs = arg2;
                mprintf("Set maximum #epochs \t\t [ %d ]\n",
                        s->anp->pars->max_epochs);
        /* report after */
        } else if (strcmp(arg1, "ReportAfter") == 0) {
                s->anp->pars->report_after = arg2;
                mprintf("Set report after (#epochs) \t [ %d ]\n",
                        s->anp->pars->report_after);
        /* random seed */
        } else if (strcmp(arg1, "RandomSeed") == 0) {
                s->anp->pars->random_seed = arg2;
                mprintf("Set random seed \t\t [ %d ]\n",
                        s->anp->pars->random_seed);
        /* number of back ticks */
        } else if (strcmp(arg1, "BackTicks") == 0) {
                s->anp->pars->back_ticks = arg2;
                mprintf("Set BPTT back ticks \t\t [ %d ]\n",
                        s->anp->pars->back_ticks);
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
        if        (strcmp(arg1, "InitContextUnits") == 0) {
                s->anp->pars->init_context_units = arg2;
                mprintf("Set init context units \t [ %lf ]\n",
                        s->anp->pars->init_context_units);
        /* random mu */
        } else if (strcmp(arg1, "RandomMu") == 0) {
                s->anp->pars->random_mu = arg2;
                mprintf("Set random Mu \t\t [ %lf ]\n",
                        s->anp->pars->random_mu);
        /* random sigma */
        } else if (strcmp(arg1, "RandomSigma") == 0) {
                s->anp->pars->random_sigma = arg2;
                mprintf("Set random Sigma \t\t [ %lf ]\n",
                        s->anp->pars->random_sigma);
        /* random minimum */
        } else if (strcmp(arg1, "RandomMin") == 0) {
                s->anp->pars->random_min = arg2;
                mprintf("Set random minimum \t\t [ %lf ]\n",
                        s->anp->pars->random_min);
        /* random maximum */
        } else if (strcmp(arg1, "RandomMax") == 0) {
                s->anp->pars->random_max = arg2;
                mprintf("Set random maximum \t\t [ %lf ]\n",
                        s->anp->pars->random_max);
        /* learning rate */
        } else if (strcmp(arg1, "LearningRate") == 0) {
                s->anp->pars->learning_rate = arg2;
                mprintf("Set learning rate \t\t [ %lf ]\n",
                        s->anp->pars->learning_rate);
        /* learning rate scale factor */
        } else if (strcmp(arg1, "LRScaleFactor") == 0) {
                s->anp->pars->lr_scale_factor = arg2;
                mprintf("Set LR scale factor \t\t [ %lf ]\n",
                        s->anp->pars->lr_scale_factor);
        /* learning rate scale after */
        } else if (strcmp(arg1, "LRScaleAfter") == 0) {
                s->anp->pars->lr_scale_after = arg2;
                mprintf("Set LR scale after (%%epochs) \t [ %lf ]\n",
                        s->anp->pars->lr_scale_after);
        /* momentum */
        } else if (strcmp(arg1, "Momentum") == 0) {
                s->anp->pars->momentum = arg2;
                mprintf("Set momentum \t\t\t [ %lf ]\n",
                        s->anp->pars->momentum);
        /* momentum scale factor */
        } else if (strcmp(arg1, "MNScaleFactor") == 0) {
                s->anp->pars->mn_scale_factor = arg2;
                mprintf("Set MN scale factor \t [ %lf ]\n",
                        s->anp->pars->mn_scale_factor);
        /* momentum scale after */
        } else if (strcmp(arg1, "MNScaleAfter") == 0) {
                s->anp->pars->mn_scale_after = arg2;
                mprintf("Set MN scale after (%%epochs) [ %lf ]\n",
                        s->anp->pars->mn_scale_after);
        /* weight decay */
        } else if (strcmp(arg1, "WeightDecay") == 0) {
                s->anp->pars->weight_decay = arg2;
                mprintf("Set weight decay \t\t [ %lf ]\n",
                        s->anp->pars->weight_decay);
        /* weight decay scale factor */
        } else if (strcmp(arg1, "WDScaleFactor") == 0) {
                s->anp->pars->wd_scale_factor = arg2;
                mprintf("Set WD scale factor \t [ %lf ]\n",
                        s->anp->pars->wd_scale_factor);
        /* weight decay scale after */
        } else if (strcmp(arg1, "WDScaleAfter") == 0) {
                s->anp->pars->wd_scale_after = arg2;
                mprintf("Set WD scale after (%%epochs) [ %lf ]\n",
                        s->anp->pars->wd_scale_after);
        /* error threshold */
        } else if (strcmp(arg1, "ErrorThreshold") == 0) {
                s->anp->pars->error_threshold = arg2;
                mprintf("Set error threshold \t\t [ %lf ]\n",
                        s->anp->pars->error_threshold);
        /* target radius */
        } else if (strcmp(arg1, "TargetRadius") == 0) {
                s->anp->pars->target_radius = arg2;
                mprintf("Set target radius \t\t [ %lf ]\n",
                        s->anp->pars->target_radius);
        /* zero error radius */
        } else if (strcmp(arg1, "ZeroErrorRadius") == 0) {
                s->anp->pars->zero_error_radius = arg2;
                mprintf("Set zero-error radius \t [ %lf ]\n",
                        s->anp->pars->zero_error_radius);
        /* rprop initial update value */
        } else if (strcmp(arg1, "RpropInitUpdate") == 0) {
                s->anp->pars->rp_init_update = arg2;
                mprintf("Set init update (for Rprop)  [ %lf ]\n",
                        s->anp->pars->rp_init_update);
        /* rprop eta plus */
        } else if (strcmp(arg1, "RpropEtaPlus") == 0) {
                s->anp->pars->rp_eta_plus = arg2;
                mprintf("Set Eta+ (for Rprop) \t [ %lf ]\n",
                        s->anp->pars->rp_eta_plus);
        /* rprop eta minus */
        } else if (strcmp(arg1, "RpropEtaMinus") == 0) {
                s->anp->pars->rp_eta_minus = arg2;
                mprintf("Set Eta- (for Rprop) \t [ %lf ]\n",
                        s->anp->pars->rp_eta_minus);
        /* delta-bar-delta increment rate */
        } else if (strcmp(arg1, "DBDRateIncrement") == 0) {
                s->anp->pars->dbd_rate_increment = arg2;
                mprintf("Set increment rate (for DBD) \t [ %lf ]\n",
                        s->anp->pars->dbd_rate_increment);
        /* delta-bar-delta decrement rate */
        } else if (strcmp(arg1, "DBDRateDecrement") == 0) {
                s->anp->pars->dbd_rate_decrement = arg2;
                mprintf("Set decrement rate (for DBD) \t [ %lf ]\n",
                        s->anp->pars->dbd_rate_decrement);
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
        /* ReLU alpha */
        if        (strcmp(arg1, "ReLUAlpha") == 0) {
                g->pars->relu_alpha = arg3;
                mprintf("Set ReLU alpha \t\t [ %s :: %lf ]\n",
                        arg2, g->pars->relu_alpha);
        /* ReLU max value */
        } else if (strcmp(arg1, "ReLUMax") == 0) {
                g->pars->relu_max = arg3;
                mprintf("Set ReLU max \t\t\t [ %s :: %lf ]\n",
                        arg2, g->pars->relu_max);
        /* logistic FSC (Flat Spot Correction) */
        } else if (strcmp(arg1, "LogisticFSC") == 0) {
                g->pars->logistic_fsc = arg3;
                mprintf("Set Logistic FSC \t\t [ %s :: %lf ]\n",
                        arg2, g->pars->logistic_fsc);
        /* logistic gain */
        } else if (strcmp(arg1, "LogisticGain") == 0) {
                g->pars->logistic_gain = arg3;
                mprintf("Set Logistic gain \t\t [ %s :: %lf ]\n",
                        arg2, g->pars->logistic_gain);                        
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
        if      (strcmp(arg, "gaussian") == 0)
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
        if      (strlen(arg) == 2 && strcmp(arg, "bp") == 0)
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
        if      (strcmp(arg, "steepest") == 0) {
                s->anp->update_algorithm = bp_update_sd;
                s->anp->flags->sd_type = SD_DEFAULT;
        }
        /* gradient [= steepest] descent */
        else if (strcmp(arg, "gradient") == 0) {
                s->anp->update_algorithm = bp_update_sd;
                s->anp->flags->sd_type = SD_DEFAULT;
        }        
        /* bounded steepest descent */
        else if (strcmp(arg, "bounded") == 0) {
                s->anp->update_algorithm = bp_update_sd;
                s->anp->flags->sd_type = SD_BOUNDED;
        }
        /* resilient propagation plus */
        else if (strcmp(arg, "rprop+") == 0) {
                s->anp->update_algorithm = bp_update_rprop;
                s->anp->flags->rp_type = RPROP_PLUS;
        }
        /* resilient propagation minus */
        else if (strcmp(arg, "rprop-") == 0) {
                s->anp->update_algorithm = bp_update_rprop;
                s->anp->flags->rp_type = RPROP_MINUS;
        }
        /* modified resilient propagation plus */
        else if (strcmp(arg, "irprop+") == 0) {
                s->anp->update_algorithm = bp_update_rprop;
                s->anp->flags->rp_type = IRPROP_PLUS;
        }
        /* modified resilient propagation minus */
        else if (strcmp(arg, "irprop-") == 0) {
                s->anp->update_algorithm = bp_update_rprop;
                s->anp->flags->rp_type = IRPROP_MINUS;
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
        if      (strcmp(arg, "inner_product") == 0)
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
        if      (strcmp(arg, "ordered") == 0)
                s->anp->flags->training_order = train_ordered;
        /* permuted */
        else if (strcmp(arg, "permuted") == 0)    
                s->anp->flags->training_order = train_permuted;
        /* randomized */
        else if (strcmp(arg, "randomized") == 0)
                s->anp->flags->training_order = train_randomized;
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
        print_weight_statistics(s->anp);
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
        /* units */
        if        (strcmp(arg1, "units") == 0) {
                type = vtype_units;
        /* error */
        } else if (strcmp(arg1, "error") == 0) {
                type = vtype_error;
        } else {
                eprintf("Cannot show vector - no such vector type '%s'\n", arg1);
                return true;
        }
        /* find group */
        struct group *g = find_network_group_by_name(s->anp, arg2);
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
        /* weights */
        if      (strcmp(arg1, "weights") == 0)
                type = mtype_weights;
        /* gradients */
        else if (strcmp(arg1, "gradients") == 0)
                type = mtype_gradients;
         /* dynamics */
        else if (strcmp(arg1, "dynamics") == 0)
                type = mtype_dynamic_params;
        else {
                eprintf("Cannot show matrix - no such matrix type '%s'\n", arg1);
                return false;
        }
        /* find 'from' group */
        struct group *fg = find_network_group_by_name(s->anp, arg2);
        if (fg == NULL) {
                eprintf("Cannot show matrix - no such group '%s'\n", arg2);
                return true;
        }
        /* find 'to' group */
        struct group *tg = find_network_group_by_name(s->anp, arg3);
        if (tg == NULL) {
                eprintf("Cannot show matrix - no such group '%s'\n", arg3);
                return true;
        }
        /* find projection */
        struct projection *fg_to_tg = find_projection(fg->out_projs, tg);
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

bool cmd_load_legacy_set(char *cmd, char *fmt, struct session *s)
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
        struct set *set = load_legacy_set(arg1, arg2,
                s->anp->input->vector->size,
                s->anp->output->vector->size);
        if (!set) {
                eprintf("Failed to load let '%s'\n", arg2);
                return true;
        }
        add_set(s->anp, set);
        mprintf("Loaded set \t\t\t [ %s => %s (%d) ]\n", arg2, set->name,
                set->items->num_elements);
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
                eprintf("Failed to load let '%s'\n", arg2);
                return true;
        }
        add_set(s->anp, set);
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
        remove_set(s->anp, set);
        mprintf("Removed set \t\t [ %s ]\n", arg);
        return true;
}

bool cmd_sets(char *cmd, char *fmt, struct session *s)
{
        if (strcmp(cmd, fmt) != 0)
                return false;
        print_sets(s->anp);
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
        if (s->anp->flags->initialized)
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

bool cmd_test_item_num(char *cmd, char *fmt, struct session *s)
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
        print_items(s->anp->asp);
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
        print_item(item, s->pprint, s->scheme);
        return true;
}

bool cmd_show_item_num(char *cmd, char *fmt, struct session *s)
{
        uint32_t arg; /* item number */
        if (sscanf(cmd, fmt, &arg) != 1)
                return false;
        /* find item */
        if (arg == 0 || arg > s->anp->asp->items->num_elements) {
                eprintf("Cannot show item - no such item number '%d'\n", arg);
                return true;
        }
        struct item *item = s->anp->asp->items->elements[arg - 1];
        print_item(item, s->pprint, s->scheme);
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

bool cmd_set_two_stage_forward(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* two-stage forward group name */
        char arg2[MAX_ARG_SIZE]; /* two-stage forward set name */
        if (sscanf(cmd, fmt, arg1, arg2) != 2)
                return false;
        /* find two-stage forward group */
        struct group *g = find_array_element_by_name(s->anp->groups, arg1);
        if (g == NULL) {
                eprintf("Cannot set two-stage forward - no such group '%s'\n", arg1);
                return true;
        }
        /* find two-stage forward set */
        struct set *set = find_array_element_by_name(s->anp->sets, arg2);
        if (!set) {
                eprintf("Cannot set two-stage forward - no such set '%s'\n", arg2);
                return true;
        }
        s->anp->ts_fw_group = g;
        s->anp->ts_fw_set   = set;
        mprintf("Set two-stage forward \t [ %s --> (%s :: %s) --> %s ]\n", 
                s->anp->input->name, s->anp->ts_fw_group->name,
                s->anp->ts_fw_set->name, s->anp->output->name);
        return true;
}

bool cmd_set_one_stage_forward(char *cmd, char *fmt, struct session *s)
{
        s->anp->ts_fw_group = NULL;
        s->anp->ts_fw_set   = NULL;
        mprintf("Set one-stage forward \t [ %s --> %s ]\n", 
                s->anp->input->name, s->anp->output->name);
        return true;   
}

bool cmd_set_two_stage_backward(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* two-stage backward group name */
        char arg2[MAX_ARG_SIZE]; /* two-stage backward set name */
        if (sscanf(cmd, fmt, arg1, arg2) != 2)
                return false;
        /* find two-stage backward group */
        struct group *g = find_array_element_by_name(s->anp->groups, arg1);
        if (g == NULL) {
                eprintf("Cannot set two-stage backward - no such group '%s'\n", arg1);
                return true;
        }
        /* group requires error function */
        if (!g->err_fun) {
                eprintf("Cannot set two-stage backward - group '%s' has no error function\n",
                        arg1);
                return true;
        }
        /* find two-stage backward set */
        struct set *set = find_array_element_by_name(s->anp->sets, arg2);
        if (!set) {
                eprintf("Cannot set two-stage backward - no such set '%s'\n", arg2);
                return true;
        }
        s->anp->ts_bw_group = g;
        s->anp->ts_bw_set   = set;
        mprintf("Set two-stage backward \t [ %s <-- (%s :: %s) <-- %s ]\n", 
                s->anp->input->name, s->anp->ts_bw_group->name,
                s->anp->ts_bw_set->name, s->anp->output->name);
        return true;       
}

bool cmd_set_one_stage_backward(char *cmd, char *fmt, struct session *s)
{
        s->anp->ts_bw_group = NULL;
        s->anp->ts_bw_set   = NULL;
        mprintf("Set one-stage backward \t [ %s <-- %s ]\n",
                s->anp->input->name, s->anp->output->name);
        return true;
}

bool cmd_similarity_matrix(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;
        mprintf("Computing similarity matrix for network '%s'\n", s->anp->name);
        print_sm_summary(s->anp, true, s->pprint, s->scheme);
        return true;
}

bool cmd_similarity_stats(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;
        mprintf("Computing similarity matrix for network '%s'\n", s->anp->name);
        print_sm_summary(s->anp, false, s->pprint, s->scheme);
        return true;
}

bool cmd_confusion_matrix(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;
        mprintf("Computing confusion matrix for network '%s'\n", s->anp->name);
        print_cm_summary(s->anp, true, s->pprint, s->scheme);
        return true;
}

bool cmd_confusion_stats(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;
        mprintf("Computing confusion matrix for network '%s'\n", s->anp->name);
        print_cm_summary(s->anp, false,  s->pprint, s->scheme);
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

bool cmd_dss_scores_num(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* set name */
        uint32_t arg2;           /* item number */
        if (sscanf(cmd, fmt, arg1, &arg2) != 2)
                return false;
        /* find set */
        struct set *set = find_array_element_by_name(s->anp->sets, arg1);
        if (!set) {
                eprintf("Cannot compute scores - no such set '%s'\n", arg1);
                return true;
        }
        /* find item */
        if (arg2 == 0 || arg2 > s->anp->asp->items->num_elements) {
                eprintf("Cannot compute scores - no such item number '%d'\n", arg2);
                return true;
        }
        struct item *item = s->anp->asp->items->elements[arg2 - 1];
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
                eprintf("Cannot compute inferences - no such set '%s'\n", arg1);
                return true;
        }
        /* find item */
        struct item *item = find_array_element_by_name(s->anp->asp->items,
                arg2);
        if (!item) {
                eprintf("Cannot compute inferences - no such item '%s'\n", arg2);
                return true;
        }
        /* require a proper threshold */
        if (arg3 < -1.0 || arg3 > 1.0) {
                eprintf("Cannot compute inferences - invalid score threshold '%lf'\n", arg3);
                return true;

        }
        dss_inferences(s->anp, set, item, arg3);
        return true;
}

bool cmd_dss_inferences_num(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* set name */
        uint32_t arg2;           /* item number */
        double arg3;             /* inference score threshold */
        if (sscanf(cmd, fmt, arg1, &arg2, &arg3) != 3)
                return false;
        /* find set */
        struct set *set = find_array_element_by_name(s->anp->sets, arg1);
        if (!set) {
                eprintf("Cannot compute inferences - no such set '%s'\n", arg1);
                return true;
        }
        /* find item */
        if (arg2 == 0 || arg2 > s->anp->asp->items->num_elements) {
                eprintf("Cannot compute inferences - no such item number '%d'\n", arg2);
                return true;
        }
        struct item *item = s->anp->asp->items->elements[arg2 - 1];
        /* require a proper threshold */
        if (arg3 < -1.0 || arg3 > 1.0) {
                eprintf("Cannot compute inferences - invalid score threshold '%lf'\n", arg3);
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
                eprintf("Cannot compute informativity metrics - no such set '%s'\n", arg1);
                return true;
        }
        /* find item */
        struct item *item = find_array_element_by_name(s->anp->asp->items, arg2);
        if (!item) {
                eprintf("Cannot compute informativity metrics - no such item '%s'\n", arg2);
                return true;
        }
        mprintf("Testing network '%s' with item '%s':\n", s->anp->name, arg2);
        dss_word_info(s->anp, set, item);
        return true;
}

bool cmd_dss_word_info_num(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* set name */
        uint32_t arg2; /* item number */
        if (sscanf(cmd, fmt, arg1, &arg2) != 2)
                return false;
        /* find set */
        struct set *set = find_array_element_by_name(s->anp->sets, arg1);
        if (!set) {
                eprintf("Cannot compute informativity metrics - no such set '%s'\n", arg1);
                return true;
        }
        /* find item */
        if (arg2 == 0 || arg2 > s->anp->asp->items->num_elements) {
                eprintf("Cannot compute informativity metrics - no such item number '%d'\n", arg2);
                return true;
        }
        struct item *item = s->anp->asp->items->elements[arg2 - 1];
        mprintf("Testing network '%s' with item '%s':\n", s->anp->name, item->name);
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
                eprintf("Cannot compute informativity metrics - no such set '%s'\n", arg1);
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
        struct group *gen = find_array_element_by_name(s->anp->groups, arg1);
        if (gen == NULL) {
                eprintf("Cannot compute ERP correlates - no such group '%s'\n", arg1);
                return true;
        }
        /* find 'control' item */
        struct item *item1 = find_array_element_by_name(s->anp->asp->items, arg2);
        if (!item1) {
                eprintf("Cannot compute ERP correlates - no such item '%s'\n", arg2);
                return true;
        }
        /* find 'target' item */
        struct item *item2 = find_array_element_by_name(s->anp->asp->items, arg3);
        if (!item2) {
                eprintf("Cannot compute ERP correlates - no such item '%s'\n", arg3);
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
        struct group *N400_gen = find_array_element_by_name(s->anp->groups, arg1);
        if (N400_gen == NULL) {
                eprintf("Cannot compute ERP correlates - no such group '%s'\n", arg1);
                return true;
        }
        /* find 'P600 generator' group */
        struct group *P600_gen = find_array_element_by_name(s->anp->groups, arg2);
        if (P600_gen == NULL) {
                eprintf("Cannot compute ERP correlates - no such group '%s'\n", arg2);
                return true;
        }
        mprintf("Computing ERP estimates \t [ N400 :: %s | P600 :: %s ]\n",
                arg1, arg2);
        erp_write_values(s->anp, N400_gen, P600_gen, arg3);
        mprintf("Written ERP estimates \t [ %s ]\n", arg3);
        return true;
}
