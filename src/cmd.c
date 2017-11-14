/*
 * Copyright 2012-2017 Harm Brouwer <me@hbrouwer.eu>
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
#include "set.h"
#include "stats.h"
#include "similarity.h"
#include "test.h"
#include "train.h"
#include "modules/dss.h"
#include "modules/erp.h"

/* group types */
enum group_type
{
        gtype_input,
        gtype_output
};

/* vector types */
enum vector_type
{
        vtype_units,
        vtype_error
};

/* matrix types */
enum matrix_type
{
        mtype_weights,
        mtype_gradients,
        mtype_dyn_pars
};

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

        bool req_netw = false; /* require network */
        bool req_init = false; /* require intialized network */
        for (uint32_t i = 0; cmds[i].cmd_base != NULL; i++) {
                /* 
                 * Skip commands that require an active network if
                 * necessary.
                 */
                if (req_netw && !s->anp) {
                        eprintf("Cannot process command: `%s`\n", cmd);
                        eprintf("(no active network - see `help networks`)\n");
                        goto out;
                }
                /* 
                 * Skip commands that require an initialized network if
                 * necessary.
                 */
                else if (req_init && !s->anp->initialized) {
                        eprintf("Cannot process command: `%s`\n", cmd);
                        eprintf("(uninitialized network - use `init` command to initialize)\n");
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
                                char *cmd_args;
                                if (asprintf(&cmd_args, "%s %s",
                                        cmds[i].cmd_base,
                                        cmds[i].cmd_args) < 0)
                                        goto error_out;
                                success = cmds[i].cmd_proc(
                                        cmd, cmd_args, s);
                                free(cmd_args);
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
                else if (strcmp("createNetwork", cmds[i].cmd_base) == 0)
                        req_netw = true;
                /* 
                 * All commands following this `init` require an initialized
                 * network.
                 */
                else if (strcmp("init", cmds[i].cmd_base) == 0)
                        req_init = true;
        }

        /* invalid command */
        if (strlen(cmd) > 1) {
                eprintf("No such command: `%s`\n", cmd); 
                eprintf("(type `help` for help)\n");
        }

out:
        return;

error_out:
        perror("[process_command()]");
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

bool cmd_list_networks(char *cmd, char *fmt, struct session *s)
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        cprintf("Available networks:\n");
        if (s->networks->num_elements == 0) {
                cprintf("(no networks)\n");
        } else {
                for (uint32_t i = 0; i < s->networks->num_elements; i++) {
                        struct network *n = s->networks->elements[i];
                        cprintf("* %s", n->name);
                        if (n == s->anp)
                                cprintf(" (active network)\n");
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

bool cmd_create_group(char *cmd, char *fmt, struct session *s)
{
        char arg[MAX_ARG_SIZE]; /* group name */
        int32_t arg_int;
        if (sscanf(cmd, fmt, arg, &arg_int) != 2)
                return false;

        /* group should not already exist */
        if (find_array_element_by_name(s->anp->groups, arg)) {
                eprintf("Cannot create group - group '%s' already exists in network '%s'\n",
                        arg, s->anp->name);
                return true;
        }

        /* group size should be positive */
        if (!(arg_int > 0)) {
                eprintf("Cannot create group - group size should be positive\n");
                return true;
        }

        struct group *g = create_group(arg, arg_int, false, false);
        add_to_array(s->anp->groups, g);

        mprintf("Created group \t\t [ %s :: %d ]\n", arg, arg_int);

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

bool cmd_list_groups(char *cmd, char *fmt, struct session *s)
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        cprintf("Available groups:\n");
        if (s->anp->groups->num_elements == 0) {
                cprintf("(no groups)\n");
        } else {
                for (uint32_t i = 0; i < s->anp->groups->num_elements; i++) {
                        struct group *g = s->anp->groups->elements[i];
                        cprintf("* %s :: %d", g->name, g->vector->size);
                        if (g == s->anp->input)
                                cprintf(" (input group)\n");
                        else if (g == s->anp->output)
                                cprintf(" (output group)\n");
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
        char *arg_bias;
        if (asprintf(&arg_bias, "%s_bias", arg) < 0)
                goto error_out;
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

bool cmd_set_io_group(char *cmd, char *fmt, struct session *s)
{
        char arg[MAX_ARG_SIZE]; /* group name */
        enum group_type type;
        if (sscanf(cmd, "set InputGroup %s", arg) == 1)
                type = gtype_input;
        else if((sscanf(cmd, "set OutputGroup %s", arg) == 1))
                type = gtype_output;
        else
                return false;

        /* find group */
        struct group *g = find_array_element_by_name(s->anp->groups, arg);
        if (g == NULL) {
                eprintf("Cannot set input group - no such group '%s'\n", arg);
                return true;
        }

        /* set input or output group */
        if (type == gtype_input) {
                s->anp->input = g;
                mprintf("Set input group \t\t [ %s ]\n", arg);
        } else if (type == gtype_output) {
                s->anp->output = g;
                mprintf("Set output group \t\t [ %s ]\n", arg);
        }

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

        /* binary sigmoid function */
        if (strcmp(arg2, "binary_sigmoid") == 0) {
                g->act_fun->fun   = act_fun_binary_sigmoid;
                g->act_fun->deriv = act_fun_binary_sigmoid_deriv;
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
        /* step function */
        else if (strcmp(arg2, "step") == 0) {
                g->act_fun->fun   = act_fun_step;
                g->act_fun->deriv = act_fun_step_deriv;
        }
        /* softplus activation function */
        else if (strcmp(arg2, "softplus") == 0) {
                g->act_fun->fun   = act_fun_softplus;
                g->act_fun->deriv = act_fun_softplus_deriv;
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
        if (strcmp(arg2, "sum_squares") == 0) {
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
                s->anp->initialized = false;
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
        reset_context_groups(s->anp);

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

bool cmd_list_projections(char *cmd, char *fmt, struct session *s)
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        /*
         * List incoming, recurrent, and outgoing projections for each
         * group.
         */
        cprintf("Available projections:\n");
        for (uint32_t i = 0; i < s->anp->groups->num_elements; i++) {
                struct group *g = s->anp->groups->elements[i];

                /* incoming projections */
                cprintf("* ");
                for (uint32_t j = 0; j < g->inc_projs->num_elements; j++) {
                        struct projection *p = g->inc_projs->elements[j];
                        struct group *fg = p->to;
                        if (j > 0 && j < g->inc_projs->num_elements)
                                cprintf(", ");
                        cprintf("%s", fg->name, g->name);
                }
                
                /* recurrent incoming projection */
                if (g->recurrent) {
                        if (g->inc_projs->num_elements > 0)
                                cprintf(", ");
                        cprintf("%s", g->name);
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
                        cprintf("%s", tg->name);
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
                        cprintf("* [%s] => ", g->name);
                        for (uint32_t j = 0;
                                j < g->ctx_groups->num_elements;
                                j++) {
                                struct group *cg = g->ctx_groups->elements[j];
                                if (j > 0 && j < g->out_projs->num_elements)
                                        cprintf(", ");
                                cprintf("%s", cg->name);
                        }
                        cprintf("\n");
                }
        }
        
        return true;
}

/*
 * TODO: Toggle frozen projections.
 */
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
        
        /* free projection, if it exists */
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

bool cmd_create_tunnel_projection(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* 'from' group name */
        char arg2[MAX_ARG_SIZE]; /* 'to' group name */
        int32_t arg_int1, arg_int2, arg_int3, arg_int4;
        if (sscanf(cmd, fmt,
                arg1, &arg_int1, &arg_int2,
                arg2, &arg_int3, &arg_int4) != 6)
                return false;

        /* find 'from' group */
        struct group *fg = find_array_element_by_name(s->anp->groups, arg1);
        if (fg == NULL) {
                eprintf("Cannot set tunnel projection - no such group '%s'\n", arg1);
                return true;
        }
        
        /* find 'to' group */
        struct group *tg = find_array_element_by_name(s->anp->groups, arg2);
        if (tg == NULL) {
                eprintf("Cannot set tunnel projection - no such group '%s'\n", arg2);
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
         * 
         * TODO: Check logic.
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
                        arg1, arg2);
                return true;
        }

        /* 'from' and 'to' ranges should not mismatch */
        if (arg_int2 - arg_int1 != arg_int4 - arg_int3) {
                eprintf("Cannot set tunnel projection - indices [%d:%d] and [%d:%d] cover differ ranges\n",
                        arg_int1, arg_int2, arg_int3, arg_int4);
                return true;
        }

        /* tunnel should be within 'from' group bounds */
        if (arg_int1 < 0 
                || arg_int1 > fg->vector->size
                || arg_int2 < 0
                || arg_int2 > fg->vector->size
                || arg_int2 < arg_int1)
        {
                eprintf("Cannot set tunnel projection - indices [%d:%d] out of bounds\n",
                        arg_int1, arg_int2);
                return true;
        }

        /* tunnel should be within 'to' group bounds */
        if (arg_int3 < 0
                || arg_int3 > tg->vector->size
                || arg_int4 < 0
                || arg_int4 > tg->vector->size
                || arg_int4 < arg_int3)
        {
                eprintf("Cannot set tunnel projection - indices [%d:%d] out of bounds\n",
                        arg_int3, arg_int4);
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
        for (uint32_t r = arg_int1 - 1, c = arg_int3 - 1;
                r < arg_int2 && c < arg_int4;
                r++, c++) 
                weights->elements[r][c] = 1.0;

        mprintf("Created tunnel projection \t [ %s [%d:%d] -> %s [%d:%d] ]\n",
                arg1, arg_int1, arg_int2, arg2, arg_int3, arg_int4);

        return true;
}

bool cmd_set_int_parameter(char *cmd, char *fmt, struct session *s)
{
        /* batch size */
        if (sscanf(cmd, "set BatchSize %d",
                &s->anp->batch_size) == 1)
                mprintf("Set batch size \t\t [ %d ]\n",
                        s->anp->batch_size);
        /* max number of epochs */
        else if (sscanf(cmd, "set MaxEpochs %d",
                &s->anp->max_epochs) == 1)
                mprintf("Set maximum #epochs \t\t [ %d ]\n",
                        s->anp->max_epochs);
        /* report after */
        else if (sscanf(cmd, "set ReportAfter %d",
                &s->anp->report_after) == 1)
                mprintf("Set report after (#epochs) \t [ %d ]\n",
                        s->anp->report_after);
        /* random seed */
        else if (sscanf(cmd, "set RandomSeed %d",
                &s->anp->random_seed) == 1)
                mprintf("Set random seed \t\t [ %d ]\n",
                        s->anp->random_seed);
        /* number of back ticks */
        else if (sscanf(cmd, "set BackTicks %d",
                &s->anp->back_ticks) == 1)
                mprintf("Set BPTT back ticks \t\t [ %d ]\n",
                        s->anp->back_ticks);

        return true;
}

bool cmd_set_double_parameter(char *cmd, char *fmt, struct session *s)
{
        /* random mu */
        if (sscanf(cmd, "set RandomMu %lf",
                &s->anp->random_mu) == 1)
                mprintf("Set random Mu \t\t [ %lf ]\n",
                        s->anp->random_mu);
        /* random sigma */
        else if (sscanf(cmd, "set RandomSigma %lf",
                &s->anp->random_sigma) == 1)
                mprintf("Set random Sigma \t\t [ %lf ]\n",
                        s->anp->random_sigma);
        /* random minimum */
        else if (sscanf(cmd, "set RandomMin %lf",
                &s->anp->random_min) == 1)
                mprintf("Set random minimum \t\t [ %lf ]\n",
                        s->anp->random_min);
        /* random maximum */
        else if (sscanf(cmd, "set RandomMax %lf",
                &s->anp->random_max) == 1)
                mprintf("Set random maximum \t\t [ %lf ]\n",
                        s->anp->random_max);
        /* learning rate */
        else if (sscanf(cmd, "set LearningRate %lf",
                &s->anp->learning_rate) == 1)
                mprintf("Set learning rate \t\t [ %lf ]\n",
                        s->anp->learning_rate);
        /* learning rate scale factor */
        else if (sscanf(cmd, "set LRScaleFactor %lf",
                &s->anp->lr_scale_factor) == 1)
                mprintf("Set LR scale factor \t [ %lf ]\n",
                        s->anp->lr_scale_factor);
        /* learning rate scale after */
        else if (sscanf(cmd, "set LRScaleAfter %lf",
                &s->anp->lr_scale_after) == 1)
                mprintf("Set LR scale after (%%epochs) [ %lf ]\n",
                        s->anp->lr_scale_after);
        /* momentum */
        else if (sscanf(cmd, "set Momentum %lf",
                &s->anp->momentum) == 1)
                mprintf("Set momentum \t\t\t [ %lf ]\n",
                        s->anp->momentum);
        /* momentum scale factor */
        else if (sscanf(cmd, "set MNScaleFactor %lf",
                &s->anp->mn_scale_factor) == 1)
                mprintf("Set MN scale factor \t [ %lf ]\n",
                        s->anp->mn_scale_factor);
        /* momentum scale after */
        else if (sscanf(cmd, "set MNScaleAfter %lf",
                &s->anp->mn_scale_after) == 1)
                mprintf("Set MN scale after (%%epochs) [ %lf ]\n",
                        s->anp->mn_scale_after);
        /* weight decay */
        else if (sscanf(cmd, "set WeightDecay %lf",
                &s->anp->weight_decay) == 1)
                mprintf("Set weight decay \t\t [ %lf ]\n",
                        s->anp->weight_decay);
        /* weight decay scale factor */
        else if (sscanf(cmd, "set WDScaleFactor %lf",
                &s->anp->wd_scale_factor) == 1)
                mprintf("Set WD scale factor \t [ %lf ]\n",
                        s->anp->wd_scale_factor);
        /* weight decay scale after */
        else if (sscanf(cmd, "set WDScaleAfter %lf",
                &s->anp->wd_scale_after) == 1)
                mprintf("Set WD scale after (%%epochs) [ %lf ]\n",
                        s->anp->wd_scale_after);
        /* error threshold */
        else if (sscanf(cmd, "set ErrorThreshold %lf",
                &s->anp->error_threshold) == 1)
                mprintf("Set error threshold \t\t [ %lf ]\n",
                        s->anp->error_threshold);
        /* target radius */
        else if (sscanf(cmd, "set TargetRadius %lf",
                &s->anp->target_radius) == 1)
                mprintf("Set target radius \t\t [ %lf ]\n",
                        s->anp->target_radius);
        /* zero error radius */
        else if (sscanf(cmd, "set ZeroErrorRadius %lf",
                &s->anp->zero_error_radius) == 1)
                mprintf("Set zero-error radius \t [ %lf ]\n",
                        s->anp->zero_error_radius);
        /* rprop initial update value */
        else if (sscanf(cmd, "set RpropInitUpdate %lf",
                &s->anp->rp_init_update) == 1)
                mprintf("Set init update (for Rprop)  [ %lf ]\n",
                        s->anp->rp_init_update);
        /* rprop eta plus */
        else if (sscanf(cmd, "set RpropEtaPlus %lf",
                &s->anp->rp_eta_plus) == 1)
                mprintf("Set Eta+ (for Rprop) \t [ %lf ]\n",
                        s->anp->rp_eta_plus);
        /* rprop eta minus */
        else if (sscanf(cmd, "set RpropEtaMinus %lf",
                &s->anp->rp_eta_minus) == 1)
                mprintf("Set Eta- (for Rprop) \t [ %lf ]\n",
                        s->anp->rp_eta_minus);
        /* delta-bar-delta increment rate */
        else if (sscanf(cmd, "set DBDRateIncrement %lf",
                &s->anp->dbd_rate_increment) == 1)
                mprintf("Set increment rate (for DBD) \t [ %lf ]\n",
                        s->anp->dbd_rate_increment);
        /* delta-bar-delta decrement rate */
        else if (sscanf(cmd, "set DBDRateDecrement %lf",
                &s->anp->dbd_rate_decrement) == 1)
                mprintf("Set decrement rate (for DBD) \t [ %lf ]\n",
                        s->anp->dbd_rate_decrement);

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

        mprintf("Loaded set \t\t\t [ %s => %s :: %d ]\n", arg2, set->name,
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

bool cmd_list_sets(char *cmd, char *fmt, struct session *s) 
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        cprintf("Available sets:\n");
        if (s->anp->sets->num_elements == 0) {
                cprintf("(no sets)\n");
        } else {
                for (uint32_t i = 0; i < s->anp->sets->num_elements; i++) {
                        struct set *set = s->anp->sets->elements[i];
                        cprintf("* %s (%d)", set->name, set->items->num_elements);
                        if (set == s->anp->asp)
                                cprintf(" (active set)\n");
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

bool cmd_list_items(char *cmd, char *fmt, struct session *s)
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        /* there should be an active set */
        if (!s->anp->asp) {
                eprintf("Cannot list items - no active set\n");
                return true;
        }

        cprintf("Available items in set '%s':\n", s->anp->asp->name);
        for (uint32_t i = 0; i < s->anp->asp->items->num_elements; i++) {
                struct item *item = s->anp->asp->items->elements[i];
                cprintf("* \"%s\" %d \"%s\"\n",
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

        mprintf("Set similarity metric \t [ %s ]", arg);

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

bool cmd_test(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf("Testing network '%s'\n", s->anp->name);

        test_network(s->anp);

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

bool cmd_weight_stats(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        struct weight_stats *ws = create_weight_statistics(s->anp);
        print_weight_statistics(s->anp, ws);
        free_weight_statistics(ws);

        return true;
}

bool cmd_show_vector(char *cmd, char *fmt, struct session *s)
{
        char arg[MAX_ARG_SIZE]; /* group name */
        enum vector_type type;
        if (sscanf(cmd, "showUnits %s", arg) == 1)
                type = vtype_units;
        else if (sscanf(cmd, "showError %s", arg) == 1)
                type = vtype_error;
        else 
                return false;

        /* find group */
        struct group *g = find_array_element_by_name(s->anp->groups, arg);
        if (g == NULL) {
                eprintf("Cannot show vector - no such group '%s'\n", arg);
                return true;
        }

        cprintf("\n");
        switch (type) {
        case vtype_units:
                cprintf("Unit vector for '%s':\n\n", arg);
                s->pprint ? pprint_vector(g->vector, s->scheme)
                          : print_vector(g->vector);
                break;
        case vtype_error:
                cprintf("Error vector for '%s':\n\n", arg);
                s->pprint ? pprint_vector(g->error, s->scheme)
                          : print_vector(g->error);
                break;
        }
        cprintf("\n");

        return true;
}

bool cmd_show_matrix(char *cmd, char *fmt, struct session *s)
{
        char arg1[MAX_ARG_SIZE]; /* 'from' group name */
        char arg2[MAX_ARG_SIZE]; /* 'to' group name */
        enum matrix_type type;
        /* weights */
        if (sscanf(cmd, "showWeights %s %s", arg1, arg2) == 2)
                type = mtype_weights;
        /* gradients */
        else if (sscanf(cmd, "showGradients %s %s", arg1, arg2) == 2)
                type = mtype_gradients;
        /* dynamic learning parameters */
        else if (sscanf(cmd, "showDynamicParams %s %s", arg1, arg2) == 2)
                type = mtype_dyn_pars;
        else
                return false;

        /* find 'from' group */
        struct group *fg = find_array_element_by_name(s->anp->groups, arg1);
        if (fg == NULL) {
                eprintf("Cannot show matrix - no such group '%s'\n", arg1);
                return true;
        }

        /* find 'to' group */
        struct group *tg = find_array_element_by_name(s->anp->groups, arg2);
        if (tg == NULL) {
                eprintf("Cannot show matrix - no such group '%s'\n", arg2);
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
                        arg1, arg2);
                return true;
        }

        switch(type) {
        case mtype_weights:
                cprintf("Weight matrix for projection '%s -> %s':\n\n",
                        arg1, arg2);
                s->pprint ? pprint_matrix(fg_to_tg->weights, s->scheme)
                          : print_matrix(fg_to_tg->weights);
                break;
        case mtype_gradients:
                cprintf("Gradient matrix for projection '%s -> %s':\n\n",
                        arg1, arg2);
                s->pprint ? pprint_matrix(fg_to_tg->gradients, s->scheme)
                          : print_matrix(fg_to_tg->gradients);
                break;
        case mtype_dyn_pars:
                cprintf("Dynamic learning parameters for projection '%s -> %s':\n\n",
                        arg1, arg2);
                s->pprint ? pprint_matrix(fg_to_tg->dynamic_params, s->scheme)
                          : print_matrix(fg_to_tg->dynamic_params);
                break;
        }
        cprintf("\n");

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

bool cmd_dss_word_information(char *cmd, char *fmt, struct session *s)
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

        dss_word_information(s->anp, set, item);

        return true;
}

bool cmd_dss_write_word_information(char *cmd, char *fmt, struct session *s)
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

        dss_write_word_information(s->anp, set, arg2);

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

bool cmd_erp_write_estimates(char *cmd, char *fmt, struct session *s)
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

        erp_write_estimates(s->anp, N400_gen, P600_gen, arg3);

        mprintf("Written ERP estimates \t [ %s ]\n", arg3);

        return true;
}
