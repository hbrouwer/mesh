/*
 * cmd.c
 *
 * Copyright 2012-2015 Harm Brouwer <me@hbrouwer.eu>
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

/* modules */
#include "modules/dss.h"
#include "modules/dynsys.h"
#include "modules/erp.h"

/* group types */
#define GTYPE_INPUT  0
#define GTYPE_OUTPUT 1

/* vector types */
#define VTYPE_UNITS 0
#define VTYPE_ERROR 1

/* matrix types */
#define MTYPE_WEIGHTS   0
#define MTYPE_GRADIENTS 1
#define MTYPE_DYN_PARS  2

/* commands */
const static struct command cmds[] = {
        /* quit or exit **************************************************/
        {"quit",                    NULL,            &cmd_quit},
        {"exit",                    NULL,            &cmd_quit},

        /* file loading **************************************************/
        {"loadFile",                "%s",            &cmd_load_file},

        /* network commands **********************************************/
        {"createNetwork",           "%s %s",         &cmd_create_network},
        {"disposeNetwork",          "%s",            &cmd_dispose_network},
        {"listNetworks",            NULL,            &cmd_list_networks},
        {"changeNetwork",           "%s",            &cmd_change_network},

        /* group commands ************************************************/
        {"createGroup",             "%s %d",         &cmd_create_group},
        {"disposeGroup",            "%s",            &cmd_dispose_group},
        {"listGroups",              NULL,            &cmd_list_groups},
        {"attachBias",              "%s",            &cmd_attach_bias},
        {"set InputGroup",          "%s",            &cmd_set_io_group},
        {"set OutputGroup",         "%s",            &cmd_set_io_group},
        {"set ActFunc",             "%s %s",         &cmd_set_act_func},
        {"set ErrFunc",             "%s %s",         &cmd_set_err_func},
        {"toggleActLookup",         NULL,            &cmd_toggle_act_lookup},

        /* projection commands *******************************************/
        {"createProjection",        "%s %s",         &cmd_create_projection},
        {"disposeProjection",       "%s %s",         &cmd_dispose_projection},
        {"createElmanProjection",   "%s %s",         &cmd_create_elman_projection},
        {"disposeElmanProjection",  "%s %s",         &cmd_dispose_elman_projection},
        {"listProjections",         NULL,            &cmd_list_projections},
        {"freezeProjection",        "%s %s",         &cmd_freeze_projection},
        {"createTunnelProjection",  "%s %d %d %s %d %d",
                                                     &cmd_create_tunnel_projection},

        /* integer parameters ********************************************/
        {"set BatchSize",           "%d",            &cmd_set_int_parameter},
        {"set MaxEpochs",           "%d",            &cmd_set_int_parameter},
        {"set ReportAfter",         "%d",            &cmd_set_int_parameter},
        {"set RandomSeed",          "%d",            &cmd_set_int_parameter},
        {"set BackTicks",           "%d",            &cmd_set_int_parameter},

        /* double parameters *********************************************/
        {"set RandomMu",            "%lf",           &cmd_set_double_parameter},
        {"set RandomSigma",         "%lf",           &cmd_set_double_parameter},
        {"set RandomMax",           "%lf",           &cmd_set_double_parameter},
        {"set RandomMin",           "%lf",           &cmd_set_double_parameter},
        {"set LearningRate",        "%lf",           &cmd_set_double_parameter},
        {"set LRScaleFactor",       "%lf",           &cmd_set_double_parameter},
        {"set LRScaleAfter",        "%lf",           &cmd_set_double_parameter},
        {"set Momentum",            "%lf",           &cmd_set_double_parameter},
        {"set MNScaleFactor",       "%lf",           &cmd_set_double_parameter},
        {"set MNScaleAfter",        "%lf",           &cmd_set_double_parameter},
        {"set WeightDecay",         "%lf",           &cmd_set_double_parameter},
        {"set WDScaleFactor",       "%lf",           &cmd_set_double_parameter},
        {"set WDScaleAfter",        "%lf",           &cmd_set_double_parameter},
        {"set ErrorThreshold",      "%lf",           &cmd_set_double_parameter},
        {"set TargetRadius",        "%lf",           &cmd_set_double_parameter},
        {"set ZeroErrorRadius",     "%lf",           &cmd_set_double_parameter},
        {"set RpropInitUpdate",     "%lf",           &cmd_set_double_parameter},
        {"set RpropEtaPlus",        "%lf",           &cmd_set_double_parameter},
        {"set RpropEtaMinus",       "%lf",           &cmd_set_double_parameter},
        {"set DBDRateIncrement",    "%lf",           &cmd_set_double_parameter},
        {"set DBDRateDecrement",    "%lf",           &cmd_set_double_parameter},
        
        /* training and test sets ****************************************/
        {"loadSet",                 "%s %s",         &cmd_load_set},
        {"disposeSet",              "%s",            &cmd_dispose_set},
        {"listSets",                NULL,            &cmd_list_sets},
        {"changeSet",               "%s",            &cmd_change_set},
        {"listItems",               NULL,            &cmd_list_items},
        {"showItem",                "\"%[^\"]\"",    &cmd_show_item},
        {"set TrainingOrder",       "%s",            &cmd_set_training_order},

        /* ranzomization, learning, and updating algorithms **************/
        {"set RandomAlgorithm",     "%s"  ,          &cmd_set_random_algorithm},
        {"set LearningAlgorithm",   "%s",            &cmd_set_learning_algorithm},
        {"set UpdateAlgorithm",     "%s",            &cmd_set_update_algorithm},

        /* similarity metric */
        {"set SimilarityMetric",    "%s",            &cmd_set_similarity_metric},

        /* initialization, resetting, training, and testing **************/
        {"init",                    NULL,            &cmd_init},
        {"reset",                   NULL,            &cmd_reset},
        {"train",                   NULL,            &cmd_train},
        {"testItem",                "\"%[^\"]\"",    &cmd_test_item},        /* swapped */
        {"test",                    NULL,            &cmd_test},

        /* similarity and confusion matrices *****************************/
        {"similarityMatrix",        NULL,            &cmd_similarity_matrix},
        {"confusionMatrix",         NULL,            &cmd_confusion_matrix},

        /* weight statistics */
        {"weightStats",             NULL,            &cmd_weight_stats},

        /* show vectors and matrices *************************************/
        {"showUnits",               "%s",            &cmd_show_vector},
        {"showError",               "%s",            &cmd_show_vector},
        {"showWeights",             "%s",            &cmd_show_matrix},
        {"showGradients",           "%s",            &cmd_show_matrix},
        {"showDynPars",             "%s",            &cmd_show_matrix},

        /* weight matrix saving and loading ******************************/
        {"loadWeights",             "%s",            &cmd_load_weights},
        {"saveWeights",             "%s",            &cmd_save_weights},

        /* pretty printing and color schemes *****************************/
        {"togglePrettyPrinting",    NULL,            &cmd_toggle_pretty_printing},
        {"set ColorScheme",         "%s",            &cmd_set_color_scheme},

        /* event-related potentials module *******************************/
        {"erpContrast",             "%s \"%[^\"]\" \"%[^\"]\"",
                                                     &cmd_erp_contrast},
        {"erpGenerateTable",        "%s %s %s",      &cmd_erp_generate_table},

        /* distributed situation space module ****************************/
        {"dssTest",                 NULL,            &cmd_dss_test},
        {"dssScores",               "%s \"%[^\"]\"", &cmd_dss_scores},
        {"dssWriteScores",          "%s %s",         &cmd_dss_write_scores},
        {"dssInferences",           "%s \"%[^\"]\" %lf",
                                                     &cmd_dss_inferences},
        {"dssWordInformation",      "\"%[^\"]\"",    &cmd_dss_word_information},
        {"dssWriteWordInformation", "%s",            &cmd_dss_write_word_information},

        /* dynamic systems module ****************************************/
        {"dynsysTestItem",          "%s \"%[^\"]\"", &cmd_dynsys_test_item},

        /*****************************************************************/
        {NULL,                      NULL,            NULL}                   /* tail */
};

/**************************************************************************
 *************************************************************************/
void process_command(char *cmd, struct session *s)
{
        /* comment or blank line */
        if (cmd[0] == '%') {
                mprintf("\x1b[1m\x1b[36m%s\x1b[0m", cmd);
                return;
        }
        if (cmd[0] == '\0' || cmd[0] == '#')
                return;

        bool req_network = false;
        bool req_init_network = false;

        /* command */
        for (uint32_t i = 0; cmds[i].cmd_base != NULL; i++) {
                if (req_network && !s->anp) {
                        eprintf("Cannot process command: %s", cmd);
                        eprintf("No active network");
                        return;
                }
                else if (req_init_network && !s->anp->initialized) {
                        eprintf("Cannot process command: %s", cmd);
                        eprintf("Unitizialized network--use 'init' command to initialize");
                        return;
                }
                else if (strncmp(cmd, cmds[i].cmd_base, strlen(cmds[i].cmd_base)) == 0) {
                        bool success;
                        if (cmds[i].cmd_args != NULL) {
                                char *cmd_args;
                                if (asprintf(&cmd_args, "%s %s", cmds[i].cmd_base, cmds[i].cmd_args) < 0)
                                        goto error_out;
                                success = cmds[i].cmd_proc(cmd, cmd_args, s);
                                free(cmd_args);
                        } else {                     
                                success = cmds[i].cmd_proc(cmd, cmds[i].cmd_base, s);
                        }
                        if (success)
                                return;
                }
                else if (strcmp("createNetwork", cmds[i].cmd_base) == 0)
                        req_network = true;
                else if (strcmp("init", cmds[i].cmd_base) == 0)
                        req_init_network = true;
        }

        /* invalid command */
        if (strlen(cmd) > 1) {
                eprintf("invalid command: %s", cmd); 
                eprintf("(type 'help' for a list of valid commands)");
        }

        return;

error_out:
        perror("[process_command()]");
        return;
}

/**************************************************************************
 *************************************************************************/
bool cmd_quit(char *cmd, char *fmt, struct session *s)
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        mprintf("Leaving MESH.");
        
        dispose_session(s);
        exit(EXIT_SUCCESS);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_load_file(char *cmd, char *fmt, struct session *s)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        FILE *fd;
        if (!(fd = fopen(tmp, "r"))) {
                eprintf("Cannot open file '%s'", tmp);
                return true;
        }

        char buf[MAX_BUF_SIZE];
        while (fgets(buf, sizeof(buf), fd)) {
                buf[strlen(buf) - 1] = '\0';
                process_command(buf, s);
        }

        fclose(fd);

        mprintf("Loaded file ... \t\t ( %s )", tmp);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_create_network(char *cmd, char *fmt, struct session *s)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        uint32_t type = 0;
        /* feed forward network */
        if (strcmp(tmp2, "ffn") == 0)
                type = TYPE_FFN;
         /* simple recurrent network */
        else if (strcmp(tmp2, "srn") == 0)
                type = TYPE_SRN;
        /* recurrent network */
        else if (strcmp(tmp2, "rnn") == 0)
                type = TYPE_RNN;
        else {
                eprintf("Cannot create network--invalid network type: '%s'", tmp2);
                return true;
        }

        if (find_array_element_by_name(s->networks, tmp1)) {
                eprintf("Cannot create network--network '%s' already exists", tmp1);
                return true;
        }
        
        struct network *n = create_network(tmp1, type);
        add_to_array(s->networks, n);
        s->anp = n;

        mprintf("Created network ... \t\t ( %s :: %s )", tmp1, tmp2);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_dispose_network(char *cmd, char *fmt, struct session *s)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        struct network *n = find_array_element_by_name(s->networks, tmp);
        if (!n) {
                eprintf("Cannot dispose network--no such network '%s'", tmp);
                return true;
        }

        if (n == s->anp)
                s->anp = NULL;
        remove_from_array(s->networks, n);
        dispose_network(n);

        mprintf("Disposed network ... \t\t ( %s )", tmp);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_list_networks(char *cmd, char *fmt, struct session *s)
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        mprintf("Available networks:");
        if (s->networks->num_elements == 0) {
                cprintf("(No networks)\n");
        } else {
                for (uint32_t i = 0; i < s->networks->num_elements; i++) {
                        struct network *n = s->networks->elements[i];
                        cprintf("* %s", n->name);
                        if (n == s->anp) {
                                cprintf("\t <- Active network\n");
                        } else {
                                cprintf("\n");
                        }
                }
        }

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_change_network(char *cmd, char *fmt, struct session *s)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        struct network *n = find_array_element_by_name(s->networks, tmp);
        if (!n) {
                eprintf("Cannot change to network--no such network '%s'", tmp);
                return true;
        }
        s->anp = n;

        mprintf("Changed to network ... \t ( %s )", tmp);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_create_group(char *cmd, char *fmt, struct session *s)
{
        char tmp[MAX_ARG_SIZE];
        int32_t tmp_int;
        if (sscanf(cmd, fmt, tmp, &tmp_int) != 2)
                return false;

        if (find_array_element_by_name(s->anp->groups, tmp)) {
                eprintf("Cannot create group--group '%s' already exists in network '%s'",
                                tmp, s->anp->name);
                return true;
        }
        if (!(tmp_int > 0)) {
                eprintf("Cannot create group--group size should be positive");
                return true;
        }

        struct group *g = create_group(tmp, tmp_int, false, false);
        add_to_array(s->anp->groups, g);

        mprintf("Created group ... \t\t ( %s :: %d )", tmp, tmp_int);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_dispose_group(char *cmd, char *fmt, struct session *s)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        struct group *g = find_array_element_by_name(s->anp->groups, tmp);
        if (g == NULL) {
                eprintf("Cannot dispose group--no such group '%s'", tmp);
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

        remove_from_array(s->anp->groups, g);
        dispose_group(g);

        mprintf("Disposed group ... \t\t ( %s )", tmp);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_list_groups(char *cmd, char *fmt, struct session *s)
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        mprintf("Available groups:");
        if (s->anp->groups->num_elements == 0) {
                cprintf("(No groups)\n");
        } else {
                for (uint32_t i = 0; i < s->anp->groups->num_elements; i++) {
                        struct group *g = s->anp->groups->elements[i];
                        cprintf("* %s :: %d", g->name, g->vector->size);
                        if (g == s->anp->input) {
                                cprintf("\t\t <- Input group\n");
                        } else if (g == s->anp->output) {
                                cprintf("\t\t <- Output group\n");
                        } else {
                                cprintf("\n");
                        }
                }
        }

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_attach_bias(char *cmd, char *fmt, struct session *s)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        struct group *g = find_array_element_by_name(s->anp->groups, tmp);
        if (g == NULL) {
                eprintf("Cannot attach bias group--no such group '%s'", tmp);
                return true;
        }

        char *tmp_bias;
        if (asprintf(&tmp_bias, "%s_bias", tmp) < 0)
                goto error_out;
        if (find_array_element_by_name(s->anp->groups, tmp_bias)) {
                eprintf("Cannot attach bias group--group '%s' already exists in network '%s'",
                                tmp_bias, s->anp->name);
                return true;
        }
        free(tmp_bias);

        struct group *bg = attach_bias_group(s->anp, g);

        mprintf("Attached bias to group ... \t ( %s -> %s )", bg->name, g->name);

        return true;

error_out:
        perror("[cmd_attach_bias()]");
        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_set_io_group(char *cmd, char *fmt, struct session *s)
{
        uint32_t type;

        char tmp[MAX_ARG_SIZE];
        /* input group */
        if (sscanf(cmd, "set InputGroup %s", tmp) == 1)
                type = GTYPE_INPUT;
        /* output group */
        else if((sscanf(cmd, "set OutputGroup %s", tmp) == 1))
                type = GTYPE_OUTPUT;
        else
                return false;

        struct group *g = find_array_element_by_name(s->anp->groups, tmp);
        if (g == NULL) {
                eprintf("Cannot set input group--no such group '%s'", tmp);
                return true;
        }

        if (type == GTYPE_INPUT) {
                s->anp->input = g;
                mprintf("Set input group ... \t\t ( %s )", tmp);
        } else {
                s->anp->output = g;
                mprintf("Set output group ... \t\t ( %s )", tmp);
        }

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_set_act_func(char *cmd, char *fmt, struct session *s)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        struct group *g = find_array_element_by_name(s->anp->groups, tmp1);
        if (g == NULL) {
                eprintf("Cannot set activation function--no such group '%s'", tmp1);
                return true;
        }

        /* binary sigmoid function */
        if (strcmp(tmp2, "binary_sigmoid") == 0) {
                g->act_fun->fun = act_fun_binary_sigmoid;
                g->act_fun->deriv = act_fun_binary_sigmoid_deriv;
        }
        /* bipolar sigmoid function */
        else if (strcmp(tmp2, "bipolar_sigmoid") == 0) {
                g->act_fun->fun = act_fun_bipolar_sigmoid;
                g->act_fun->deriv = act_fun_bipolar_sigmoid_deriv;
        }
        /* softmax activation function */
        else if (strcmp(tmp2, "softmax") == 0) {
                g->act_fun->fun = act_fun_softmax;
                g->act_fun->deriv = act_fun_softmax_deriv;
        }
        /* hyperbolic tangent function */
        else if (strcmp(tmp2, "tanh") == 0) {
                g->act_fun->fun = act_fun_tanh;
                g->act_fun->deriv = act_fun_tanh_deriv;
        }
        /* linear function */
        else if (strcmp(tmp2, "linear") == 0) {
                g->act_fun->fun = act_fun_linear;
                g->act_fun->deriv = act_fun_linear_deriv;
        }
        /* step function */
        else if (strcmp(tmp2, "step") == 0) {
                g->act_fun->fun = act_fun_step;
                g->act_fun->deriv = act_fun_step_deriv;
        } else {
                eprintf("Cannot set activation function--no such activation function '%s'", tmp2);
                return true;
        }

        mprintf("Set activation function ... \t ( %s :: %s )", tmp1, tmp2);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_set_err_func(char *cmd, char *fmt, struct session *s)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        struct group *g = find_array_element_by_name(s->anp->groups, tmp1);
        if (g == NULL) {
                eprintf("Cannot set error function--no such group '%s'", tmp1);
                return true;
        }

        /* sum of squares */
        if (strcmp(tmp2, "sum_squares") == 0) {
                g->err_fun->fun = error_sum_of_squares;
                g->err_fun->deriv = error_sum_of_squares_deriv;
        }
        /* cross-entropy */
        else if (strcmp(tmp2, "cross_entropy") == 0) {
                g->err_fun->fun = error_cross_entropy;
                g->err_fun->deriv = error_cross_entropy_deriv;
        }
        /* divergence */
        else if (strcmp(tmp2, "divergence") == 0) {
                g->err_fun->fun = error_divergence;
                g->err_fun->deriv = error_divergence_deriv;
        } else {
                eprintf("Cannot set error function--no such error function '%s'", tmp2);
                return true;
        }

        mprintf("Set error function ... \t\t ( %s :: %s )", tmp1, tmp2);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_toggle_act_lookup(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        s->anp->act_lookup = !s->anp->act_lookup;
        
        if (s->anp->act_lookup) {
                initialize_act_lookup_vectors(s->anp);
                mprintf("Toggle activation lookup ... \t ( on )");
        } else {
                mprintf("Toggle activation lookup ... \t ( off )");
        }

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_create_projection(char *cmd, char *fmt, struct session *s)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        struct group *fg = find_array_element_by_name(s->anp->groups, tmp1);
        struct group *tg = find_array_element_by_name(s->anp->groups, tmp2);

        if (fg == NULL) {
                eprintf("Cannot set projection--no such group '%s'", tmp1);
                return true;
        }
        if (tg == NULL) {
                eprintf("Cannot set projection--no such group '%s'", tmp2);
                return true;
        }

        bool exists = false;
        if (fg->recurrent)
                exists = true;
        for (uint32_t i = 0; i < fg->out_projs->num_elements; i++)
                if (((struct projection *)fg->out_projs->elements[i])->to == tg)
                        exists = true;
        if (exists) {
                eprintf("Cannot set projection--projection '%s -> %s' already exists",
                                tmp1, tmp2);
                return true;
        }

        if (fg == tg)
                fg->recurrent = true;
        else {
                /* weight matrix */
                struct matrix *weights = create_matrix(
                                fg->vector->size,
                                tg->vector->size);
                /* gradients matrix */
                struct matrix *gradients = create_matrix(
                                fg->vector->size,
                                tg->vector->size);
                /* previous gradients matrix */
                struct matrix *prev_gradients = create_matrix(
                                fg->vector->size,
                                tg->vector->size);
                /* previous weight deltas matrix */
                struct matrix *prev_deltas = create_matrix(
                                fg->vector->size,
                                tg->vector->size);
                /* dynamic learning parameters matrix */
                struct matrix *dynamic_pars = create_matrix(
                                fg->vector->size,
                                tg->vector->size);

                /* add projections */
                struct projection *op = create_projection(tg, weights, gradients,
                                prev_gradients, prev_deltas, dynamic_pars);
                struct projection *ip = create_projection(fg, weights, gradients,
                                prev_gradients, prev_deltas, dynamic_pars);

                add_to_array(fg->out_projs, op);
                add_to_array(tg->inc_projs, ip);
        }

        mprintf("Created projection ... \t\t ( %s -> %s )", tmp1, tmp2);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_dispose_projection(char *cmd, char *fmt, struct session *s)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        struct group *fg = find_array_element_by_name(s->anp->groups, tmp1);
        struct group *tg = find_array_element_by_name(s->anp->groups, tmp2);

        if (fg == NULL) {
                eprintf("Cannot dispose projection--no such group '%s'", tmp1);
                return  true;
        }
        if (tg == NULL) {
                eprintf("Cannot dispose projection--no such group '%s'", tmp2);
                return true;
        }

        struct projection *fg_to_tg = NULL;
        for (uint32_t i = 0; i < fg->out_projs->num_elements; i++)
                if (((struct projection *)fg->out_projs->elements[i])->to == tg)
                        fg_to_tg = fg->out_projs->elements[i];
        
        struct projection *tg_to_fg = NULL;
        for (uint32_t i = 0; i < tg->inc_projs->num_elements; i++)
                if (((struct projection *)tg->inc_projs->elements[i])->to == fg)
                        tg_to_fg = tg->inc_projs->elements[i];
        
        if (fg_to_tg && tg_to_fg) {
                remove_from_array(fg->out_projs, fg_to_tg);
                remove_from_array(tg->inc_projs, tg_to_fg);
                dispose_projection(fg_to_tg);
                free(tg_to_fg);
                s->anp->initialized = false;
        } else {
                eprintf("Cannot dispose projection--no projection between groups '%s' and '%s')",
                                tmp1, tmp2);
                return true;
        }

        mprintf("Disposed projection ... \t ( %s -> %s )", tmp1, tmp2);

        return true;
}


/**************************************************************************
 *************************************************************************/
bool cmd_create_elman_projection(char *cmd, char *fmt, struct session *s)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        struct group *fg = find_array_element_by_name(s->anp->groups, tmp1);
        struct group *tg = find_array_element_by_name(s->anp->groups, tmp2);

        if (fg == NULL) {
                eprintf("Cannot set Elman-projection--no such group '%s'", tmp1);
                return true;
        }
        if (tg == NULL) {
                eprintf("Cannot set Elman-projection--no such group '%s'", tmp2);
                return true;
        }
        if (fg == tg) {
                eprintf("Cannot set Elman-projection--projection is recurrent for group '%s'", fg->name);
                return true;
        }
        if (fg->vector->size != tg->vector->size) {
                eprintf("Cannot set Elman-projection--groups '%s' and '%s' have unequal vector sizes (%d and %d)",
                                fg->name, tg->name, fg->vector->size, tg->vector->size);
                return true;
        }

        for (uint32_t i = 0; i < fg->ctx_groups->num_elements; i++) {
                if (fg->ctx_groups->elements[i] == tg) {
                        eprintf("Cannot set Elman-projection--Elman-projection '%s -> %s' already exists",
                                tmp1, tmp2);
                        return true;
                }
        }

        add_to_array(fg->ctx_groups, tg);
        reset_context_groups(s->anp);

        mprintf("Created Elman projection ... \t ( %s -> %s )", tmp1, tmp2);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_dispose_elman_projection(char *cmd, char *fmt, struct session *s)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        struct group *fg = find_array_element_by_name(s->anp->groups, tmp1);
        struct group *tg = find_array_element_by_name(s->anp->groups, tmp2);

        if (fg == NULL) {
                eprintf("Cannot dispose Elman-projection--no such group '%s'", tmp1);
                return true;
        }
        if (tg == NULL) {
                eprintf("Cannot dispose Elman-projection--no such group '%s'", tmp2);
                return true;
        }

        bool removed = false;
        for (uint32_t i = 0; i < fg->ctx_groups->num_elements; i++) {
                if (fg->ctx_groups->elements[i] == tg) {
                        remove_from_array(fg->ctx_groups, tg);
                        removed = true;
                        break;
                }
        }
        if (!removed) {
                eprintf("Cannot dispose Elman-projection--no Elman projection from group '%s' to '%s'",
                                tmp1, tmp2);
                return true;
        }

        mprintf("Disposed Elman projection ... \t ( %s -> %s )", tmp1, tmp2);
        
        return true;

}

/**************************************************************************
 *************************************************************************/
bool cmd_list_projections(char *cmd, char *fmt, struct session *s)
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        mprintf("Available projections:");
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
                        for (uint32_t j = 0; j < g->ctx_groups->num_elements; j++) {
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

/**************************************************************************
 *************************************************************************/
bool cmd_freeze_projection(char *cmd, char *fmt, struct session *s)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        struct group *fg = find_array_element_by_name(s->anp->groups, tmp1);
        struct group *tg = find_array_element_by_name(s->anp->groups, tmp2);

        if (fg == NULL) {
                eprintf("Cannot freeze projection--no such group '%s'", tmp1);
                return true;
        }
        if (tg == NULL) {
                eprintf("Cannot freeze projection--no such group '%s'", tmp2);
                return true;
        }

        struct projection *fg_to_tg = NULL;
        for (uint32_t i = 0; i < fg->out_projs->num_elements; i++)
                if (((struct projection *)fg->out_projs->elements[i])->to == tg)
                        fg_to_tg = fg->out_projs->elements[i];

        struct projection *tg_to_fg = NULL;
        for (uint32_t i = 0; i < tg->inc_projs->num_elements; i++)
                if (((struct projection *)tg->inc_projs->elements[i])->to == fg)
                        tg_to_fg = tg->inc_projs->elements[i];
        
        if (fg_to_tg && tg_to_fg) {
                fg_to_tg->frozen = true;
                tg_to_fg->frozen = true;
        } else {
                eprintf("Cannot freeze projection--no projection between groups '%s' and '%s')",
                                tmp1, tmp2);
                return true;
        }

        mprintf("Froze projection ... \t\t ( %s -> %s )", tmp1, tmp2);

        return true;
}

/**************************************************************************
 * This implements machinery for the "tunneling" of a subset of units of
 * a layer, allowing for the segmentation of a single input vector into
 * multiple ones:
 *
 * +---------+    +---------+    +---------+
 * | output1 |    | output2 |    | output3 |
 * +---------+    +---------+    +---------+
 *          \          |           /
 *      +---------+---------+---------+
 *      |         : input0  :         |
 *      +---------+---------+---------+
 *
 * and for the merging of several output vectors into a single vector:
 *
 *      +---------+---------+---------+
 *      |         : output0 :         |
 *      +---------+---------+---------+
 *          /          |           \
 * +---------+    +---------+    +---------+
 * | output1 |    | output2 |    | output3 |
 * +---------+    +---------+    +---------+
 *************************************************************************/
bool cmd_create_tunnel_projection(char *cmd, char *fmt, struct session *s)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        int32_t tmp_int1, tmp_int2, tmp_int3, tmp_int4;
        if (sscanf(cmd, fmt, tmp1, &tmp_int1, &tmp_int2, tmp2, &tmp_int3, &tmp_int4) != 6)
                return false;

        struct group *fg = find_array_element_by_name(s->anp->groups, tmp1);
        struct group *tg = find_array_element_by_name(s->anp->groups, tmp2);

        if (fg == NULL) {
                eprintf("Cannot set tunnel projection--no such group '%s'", tmp1);
                return true;
        }
        if (tg == NULL) {
                eprintf("Cannot set tunnel projection--no such group '%s'", tmp2);
                return true;
        }
        if (fg == tg) {
                eprintf("Cannot set recurrent tunnel projection");
                return true;
        }

        // XXX: This precludes multiple tunnels to the same layer.
        bool exists = false;
        if (fg->recurrent)
                exists = true;
        for (uint32_t i = 0; i < fg->out_projs->num_elements; i++)
                if (((struct projection *)fg->out_projs->elements[i])->to == tg)
                        exists = true;
        if (exists) {
                eprintf("Cannot set tunnel projection--projection '%s -> %s' already exists",
                                tmp1, tmp2);
                return true;
        }

        /* check ranges */
        if (tmp_int2 - tmp_int1 != tmp_int4 - tmp_int3) {
                eprintf("Cannot set tunnel projection--indices [%d:%d] and [%d:%d] cover differ ranges",
                                tmp_int1, tmp_int2, tmp_int3, tmp_int4);
                return true;
        }

        /* check from group bounds */
        if (tmp_int1 < 0 
                        || tmp_int1 > fg->vector->size
                        || tmp_int2 < 0
                        || tmp_int2 > fg->vector->size
                        || tmp_int2 < tmp_int1)
        {
                eprintf("Cannot set tunnel projection--indices [%d:%d] out of bounds",
                                tmp_int1, tmp_int2);
                return true;
        }

        /* check to group bounds */
        if (tmp_int3 < 0 
                        || tmp_int3 > tg->vector->size
                        || tmp_int4 < 0
                        || tmp_int4 > tg->vector->size
                        || tmp_int4 < tmp_int3)
        {
                eprintf("Cannot set tunnel projection--indices [%d:%d] out of bounds",
                                tmp_int3, tmp_int4);
                return true;
        }

        /* weight matrix */
        struct matrix *weights = create_matrix(
                        fg->vector->size,
                        tg->vector->size);
        /* gradients matrix */
        struct matrix *gradients = create_matrix(
                        fg->vector->size,
                        tg->vector->size);
        /* previous gradients matrix */
        struct matrix *prev_gradients = create_matrix(
                        fg->vector->size,
                        tg->vector->size);
        /* previous weight deltas matrix */
        struct matrix *prev_deltas = create_matrix(
                        fg->vector->size,
                        tg->vector->size);
        /* dynamic learning parameters matrix */
        struct matrix *dynamic_pars = create_matrix(
                        fg->vector->size,
                        tg->vector->size);

        /* add projections */
        struct projection *op = create_projection(tg, weights, gradients,
                        prev_gradients, prev_deltas, dynamic_pars);
        struct projection *ip = create_projection(fg, weights, gradients,
                        prev_gradients, prev_deltas, dynamic_pars);

        op->frozen = true;
        ip->frozen = true;

        add_to_array(fg->out_projs, op);
        add_to_array(tg->inc_projs, ip);

        /* setup the weights for tunneling */
        for (uint32_t r = tmp_int1 - 1, c = tmp_int3 - 1; r < tmp_int2 && c < tmp_int4; r++, c++) 
                weights->elements[r][c] = 1.0;

        mprintf("Created tunnel projection ... \t ( %s [%d:%d] -> %s [%d:%d] )",
                        tmp1, tmp_int1, tmp_int2, tmp2, tmp_int3, tmp_int4);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_set_int_parameter(char *cmd, char *fmt, struct session *s)
{
        /* batch size */
        if (sscanf(cmd, "set BatchSize %d", &s->anp->batch_size) == 1)
                mprintf("Set batch size ... \t\t ( %d )", s->anp->batch_size);
        /* max number of epochs */
        else if (sscanf(cmd, "set MaxEpochs %d", &s->anp->max_epochs) == 1)
                mprintf("Set maximum #epochs ... \t ( %d )", s->anp->max_epochs);
        /* report after */
        else if (sscanf(cmd, "set ReportAfter %d", &s->anp->report_after) == 1)
                mprintf("Set report after (#epochs) ... \t ( %d )", s->anp->report_after);
        /* random seed */
        else if (sscanf(cmd, "set RandomSeed %d", &s->anp->random_seed) == 1)
                mprintf("Set random seed ... \t\t ( %d )", s->anp->random_seed);
        /* number of back ticks */
        else if (sscanf(cmd, "set BackTicks %d", &s->anp->back_ticks) == 1)
                mprintf("Set BPTT back ticks ... \t ( %d )", s->anp->back_ticks);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_set_double_parameter(char *cmd, char *fmt, struct session *s)
{
        /* random mu */
        if (sscanf(cmd, "set RandomMu %lf", &s->anp->random_mu) == 1)
                mprintf("Set random Mu ... \t\t ( %lf )", s->anp->random_mu);
        /* random sigma */
        else if (sscanf(cmd, "set RandomSigma %lf", &s->anp->random_sigma) == 1)
                mprintf("Set random Sigma ... \t\t ( %lf )", s->anp->random_sigma);
        /* random minimum */
        else if (sscanf(cmd, "set RandomMin %lf", &s->anp->random_min) == 1)
                mprintf("Set random minimum ... \t\t ( %lf )", s->anp->random_min);
        /* random maximum */
        else if (sscanf(cmd, "set RandomMax %lf", &s->anp->random_max) == 1)
                mprintf("Set random maximum ... \t\t ( %lf )", s->anp->random_max);
        /* learning rate */
        else if (sscanf(cmd, "set LearningRate %lf", &s->anp->learning_rate) == 1)
                mprintf("Set learning rate ... \t\t ( %lf )", s->anp->learning_rate);
        /* learning rate scale factor */
        else if (sscanf(cmd, "set LRScaleFactor %lf", &s->anp->lr_scale_factor) == 1)
                mprintf("Set LR scale factor ... \t ( %lf )", s->anp->lr_scale_factor);
        /* learning rate scale after */
        else if (sscanf(cmd, "set LRScaleAfter %lf", &s->anp->lr_scale_after) == 1)
                mprintf("Set LR scale after (%%epochs) ... ( %lf )", s->anp->lr_scale_after);
        /* momentum */
        else if (sscanf(cmd, "set Momentum %lf", &s->anp->momentum) == 1)
                mprintf("Set momentum ... \t\t ( %lf )", s->anp->momentum);
        /* momentum scale factor */
        else if (sscanf(cmd, "set MNScaleFactor %lf", &s->anp->mn_scale_factor) == 1)
                mprintf("Set MN scale factor ... \t ( %lf )", s->anp->mn_scale_factor);
        /* momentum scale after */
        else if (sscanf(cmd, "set MNScaleAfter %lf", &s->anp->mn_scale_after) == 1)
                mprintf("Set MN scale after (%%epochs) ... ( %lf )", s->anp->mn_scale_after);
        /* weight decay */
        else if (sscanf(cmd, "set WeightDecay %lf", &s->anp->weight_decay) == 1)
                mprintf("Set weight decay ... \t\t ( %lf )", s->anp->weight_decay);
        /* weight decay scale factor */
        else if (sscanf(cmd, "set WDScaleFactor %lf", &s->anp->wd_scale_factor) == 1)
                mprintf("Set WD scale factor ... \t ( %lf )", s->anp->wd_scale_factor);
        /* weight decay scale after */
        else if (sscanf(cmd, "set WDScaleAfter %lf", &s->anp->wd_scale_after) == 1)
                mprintf("Set WD scale after (%%epochs) ... ( %lf )", s->anp->wd_scale_after);
        /* error threshold */
        else if (sscanf(cmd, "set ErrorThreshold %lf", &s->anp->error_threshold) == 1)
                mprintf("Set error threshold ... \t ( %lf )", s->anp->error_threshold);
        /* target radius */
        else if (sscanf(cmd, "set TargetRadius %lf", &s->anp->target_radius) == 1)
                mprintf("Set target radius ... \t\t ( %lf )", s->anp->target_radius);
        /* zero error radius */
        else if (sscanf(cmd, "set ZeroErrorRadius %lf", &s->anp->zero_error_radius) == 1)
                mprintf("Set zero-error radius ... \t ( %lf )", s->anp->zero_error_radius);
        /* rprop initial update value */
        else if (sscanf(cmd, "set RpropInitUpdate %lf", &s->anp->rp_init_update) == 1)
                mprintf("Set init update (for Rprop) ...  ( %lf )", s->anp->rp_init_update);
        /* rprop eta plus */
        else if (sscanf(cmd, "set RpropEtaPlus %lf", &s->anp->rp_eta_plus) == 1)
                mprintf("Set Eta+ (for Rprop) ... \t ( %lf )", s->anp->rp_eta_plus);
        /* rprop eta minus */
        else if (sscanf(cmd, "set RpropEtaMinus %lf", &s->anp->rp_eta_minus) == 1)
                mprintf("Set Eta- (for Rprop) ... \t ( %lf )", s->anp->rp_eta_minus);
        /* delta-bar-delta increment rate */
        else if (sscanf(cmd, "set DBDRateIncrement %lf", &s->anp->dbd_rate_increment) == 1)
                mprintf("Set increment rate (for DBD) ... \t ( %lf )", s->anp->dbd_rate_increment);
        /* delta-bar-delta decrement rate */
        else if (sscanf(cmd, "set DBDRateDecrement %lf", &s->anp->dbd_rate_decrement) == 1)
                mprintf("Set decrement rate (for DBD) ... \t ( %lf )", s->anp->dbd_rate_decrement);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_load_set(char *cmd, char *fmt, struct session *s)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        if (!s->anp->input) {
                eprintf("Cannot load set--input group size unknown");
                return true;
        }
        if (!s->anp->output) {
                eprintf("Cannot load set--output group size unknown");
                return true;
        }

        if (find_array_element_by_name(s->anp->sets, tmp1)) {
                eprintf("Cannot load set--set '%s' already exists", tmp1);
                return true;
        }

        struct set *set = load_set(tmp1, tmp2, s->anp->input->vector->size, s->anp->output->vector->size);
        if (!set) {
                eprintf("Cannot load set--no such file '%s'", tmp2);
                return true;
        }
                
        add_to_array(s->anp->sets, set);
        s->anp->asp = set;

        mprintf("Loaded set ... \t\t\t ( %s => %s :: %d )", tmp2, set->name, set->items->num_elements);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_dispose_set(char *cmd, char *fmt, struct session *s)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        struct set *set = find_array_element_by_name(s->anp->sets, tmp);
        if (!set) {
                eprintf("Cannot change to set--no such set '%s'", tmp);
                return true;
        }

        if (set == s->anp->asp)
                s->anp->asp = NULL;
        remove_from_array(s->anp->sets, set);
        dispose_set(set);

        mprintf("Disposed set ... \t\t ( %s )", tmp);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_list_sets(char *cmd, char *fmt, struct session *s) 
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        mprintf("Available sets:");

        if (s->anp->sets->num_elements == 0) {
                cprintf("(No sets)\n");
        } else {
                for (uint32_t i = 0; i < s->anp->sets->num_elements; i++) {
                        struct set *set = s->anp->sets->elements[i];
                        cprintf("* %s (%d)", set->name, set->items->num_elements);
                        if (set == s->anp->asp) {
                                cprintf("\t <- Active set\n");
                        } else {
                                cprintf("\n");
                        }
                }
        }

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_change_set(char *cmd, char *fmt, struct session *s)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        struct set *set = find_array_element_by_name(s->anp->sets, tmp);
        if (!set) {
                eprintf("Cannot change to set--no such set '%s'", tmp);
                return true;
        }
        
        s->anp->asp = set;

        mprintf("Changed to set ... \t\t ( %s )", tmp);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_list_items(char *cmd, char *fmt, struct session *s)
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        if (!s->anp->asp) {
                eprintf("Cannot list items--no active set");
                return true;
        }

        mprintf("Available items in set '%s':", s->anp->asp->name);
        for (uint32_t i = 0; i < s->anp->asp->items->num_elements; i++) {
                struct item *item = s->anp->asp->items->elements[i];
                cprintf("* \"%s\" %d \"%s\"\n", item->name, item->num_events, item->meta);
        }

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_show_item(char *cmd, char *fmt, struct session *s)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        struct item *item = find_array_element_by_name(s->anp->asp->items, tmp);
        if (!item) {
                eprintf("Cannot show item--no such item '%s'", tmp);
                return true;
        }

        mprintf("");

        pprintf("Name: \"%s\"\n", item->name);
        pprintf("Meta: \"%s\"\n", item->meta);
        pprintf("\n");
        pprintf("Events: %d\n", item->num_events);

        for (uint32_t i = 0; i < item->num_events; i++) {
                /* print event number, and input vector */
                cprintf("\n");
                pprintf("Event: %d\n", i + 1);
                pprintf("Input:\n\n");
                s->pprint == true ? pprint_vector(item->inputs[i], s->pprint_scheme)
                        : print_vector(item->inputs[i]);

                /* print target vector (if available) */
                if (item->targets[i]) {
                        cprintf("\n");
                        pprintf("Target:\n\n");
                        s->pprint == true ? pprint_vector(item->targets[i], s->pprint_scheme)
                                : print_vector(item->targets[i]);
                }
        }

        mprintf("");

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_set_training_order(char *cmd, char *fmt, struct session *s)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        /* ordered */
        if (strcmp(tmp, "ordered") == 0)
                s->anp->training_order = TRAIN_ORDERED;
        /* permuted */
        else if (strcmp(tmp, "permuted") == 0)    
                s->anp->training_order = TRAIN_PERMUTED;
        /* randomized */
        else if (strcmp(tmp, "randomized") == 0)
                s->anp->training_order = TRAIN_RANDOMIZED;
        else {
                eprintf("Invalid training order '%s'", tmp);
                return true;
        }

        mprintf("Set training order ... \t\t ( %s )", tmp);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_set_random_algorithm(char *cmd, char *fmt, struct session *s)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        /* gaussian randomization */
        if (strcmp(tmp, "gaussian") == 0)
                s->anp->random_algorithm = randomize_gaussian;
        /* range randomization */
        else if (strcmp(tmp, "range") == 0)
                s->anp->random_algorithm = randomize_range;
        /* Nguyen-Widrow randomization */
        else if (strcmp(tmp, "nguyen_widrow") == 0)
                s->anp->random_algorithm = randomize_nguyen_widrow;
        /* fan-in method */
        else if (strcmp(tmp, "fan_in") == 0)
                s->anp->random_algorithm = randomize_fan_in;
        /* binary randomization */
        else if (strcmp(tmp, "binary") == 0)
                s->anp->random_algorithm = randomize_binary;
        else {
                eprintf("Invalid randomization algorithm '%s'", tmp);
                return true;
        }

        mprintf("Set random algorithm ... \t ( %s )", tmp);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_set_learning_algorithm(char *cmd, char *fmt, struct session *s)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        /* backpropagation */
        if (strlen(tmp) == 2 && strcmp(tmp, "bp") == 0)
                s->anp->learning_algorithm = train_network_with_bp;
        /* backpropgation through time */
        else if (strlen(tmp) == 4 && strcmp(tmp, "bptt") == 0)
                s->anp->learning_algorithm = train_network_with_bptt;
        else {
                eprintf("Invalid learning algorithm '%s'", tmp);
                return true;
        }
        
        mprintf("Set learning algorithm ... \t ( %s )", tmp);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_set_update_algorithm(char *cmd, char *fmt, struct session *s)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        /* steepest descent */
        if (strcmp(tmp, "steepest") == 0) {
                s->anp->update_algorithm = bp_update_sd;
                s->anp->sd_type = SD_DEFAULT;
        }
        /* bounded steepest descent */
        else if (strcmp(tmp, "bounded") == 0) {
                s->anp->update_algorithm = bp_update_sd;
                s->anp->sd_type = SD_BOUNDED;
        }
        /* resilient propagation plus */
        else if (strcmp(tmp, "rprop+") == 0) {
                s->anp->update_algorithm = bp_update_rprop;
                s->anp->rp_type = RPROP_PLUS;
        }
        /* resilient propagation minus */
        else if (strcmp(tmp, "rprop-") == 0) {
                s->anp->update_algorithm = bp_update_rprop;
                s->anp->rp_type = RPROP_MINUS;
        }
        /* modified resilient propagation plus */
        else if (strcmp(tmp, "irprop+") == 0) {
                s->anp->update_algorithm = bp_update_rprop;
                s->anp->rp_type = IRPROP_PLUS;
        }
        /* modified resilient propagation minus */
        else if (strcmp(tmp, "irprop-") == 0) {
                s->anp->update_algorithm = bp_update_rprop;
                s->anp->rp_type = IRPROP_MINUS;
        }
        /* quickprop */
        else if (strcmp(tmp, "qprop") == 0)
                s->anp->update_algorithm = bp_update_qprop;
        /* delta-bar-delta */
        else if (strcmp(tmp, "dbd") == 0)
                s->anp->update_algorithm = bp_update_dbd;
        else {
                eprintf("Invalid update algorithm '%s'", tmp);
                return true;
        }

        mprintf("Set update algorithm ... \t ( %s )", tmp);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_set_similarity_metric(char *cmd, char *fmt, struct session *s)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        /* inner product */
        if (strcmp(tmp, "inner_product") == 0)
                s->anp->similarity_metric = inner_product;
        /* harmonic mean */
        else if (strcmp(tmp, "harmonic_mean") == 0)
                s->anp->similarity_metric = harmonic_mean;
        /* cosine similarity */
        else if (strcmp(tmp, "cosine") == 0)
                s->anp->similarity_metric = cosine;
        /* tanimoto */
        else if (strcmp(tmp, "tanimoto") == 0)
                s->anp->similarity_metric = tanimoto;
        /* dice */
        else if (strcmp(tmp, "dice") == 0)
                s->anp->similarity_metric = dice;
        /* pearson correlation */
        else if (strcmp(tmp, "pearson_correlation") == 0)
                s->anp->similarity_metric = pearson_correlation;
        else {
                eprintf("Invalid similarity metric '%s'", tmp);
                return true;
        }

        mprintf("Set similarity metric ... \t ( %s )", tmp);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_init(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        init_network(s->anp);

        if (s->anp->initialized)
                mprintf("Initialized network ... \t ( %s )", s->anp->name);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_reset(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        reset_network(s->anp);

        mprintf("Reset network '%s'", s->anp->name);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_train(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf("Training network '%s'", s->anp->name);
        mprintf(" ");

        train_network(s->anp);

        mprintf(" ");

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_test(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf("Testing network '%s'", s->anp->name);
        mprintf(" ");

        test_network(s->anp);

        mprintf(" ");

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_test_item(char *cmd, char *fmt, struct session *s)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        struct item *item = find_array_element_by_name(s->anp->asp->items, tmp);
        if (!item) {
                eprintf("Cannot test network--no such item '%s'", tmp);
                return true;
        }

        mprintf("Testing network '%s' with item '%s'", s->anp->name, tmp);
        mprintf(" ");

        test_network_with_item(s->anp, item, s->pprint, s->pprint_scheme);

        mprintf(" ");

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_similarity_matrix(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf("Computing similarity matrix for network '%s' ...", s->anp->name);
        mprintf(" ");

        // TODO: handle matrix printing
        similarity_matrix(s->anp, false, s->pprint, s->pprint_scheme);

        mprintf(" ");

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_confusion_matrix(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf("Computing confusion matrix for network '%s' ...", s->anp->name);
        mprintf(" ");

        // TODO: handle matrix printing
        confusion_matrix(s->anp, false, s->pprint, s->pprint_scheme);

        mprintf(" ");

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_weight_stats(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf("Weight statistics for network '%s'", s->anp->name);
        mprintf(" ");

        struct weight_stats *ws = create_weight_statistics(s->anp);
     
        pprintf("Number of weights:\t\t%d\n", ws->num_weights);
        pprintf("Cost:\t\t\t%f\n", ws->cost);
        pprintf("Mean:\t\t\t%f\n", ws->mean);
        pprintf("Absolute mean:\t\t%f\n", ws->mean_abs);
        pprintf("Mean dist.:\t\t%f\n", ws->mean_dist);
        pprintf("Variance:\t\t\t%f\n", ws->variance);
        pprintf("Minimum:\t\t\t%f\n", ws->minimum);
        pprintf("Maximum:\t\t\t%f\n", ws->maximum);
        
        dispose_weight_statistics(ws);

        mprintf(" ");

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_show_vector(char *cmd, char *fmt, struct session *s)
{
        uint32_t type;

        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, "showUnits %s", tmp) == 1)
                type = VTYPE_UNITS;
        else if (sscanf(cmd, "showError %s", tmp) == 1)
                type = VTYPE_ERROR;
        else 
                return false;

        struct group *g = find_array_element_by_name(s->anp->groups, tmp);
        if (g == NULL) {
                eprintf("Cannot show vector--no such group '%s'", tmp);
                return true;
        }

        if (type == VTYPE_UNITS) {
                mprintf("Unit vector for '%s'", tmp);
                mprintf(" ");
                if (s->pprint) {
                        pprint_vector(g->vector, s->pprint_scheme);
                } else {
                        print_vector(g->vector);
                }
        }
        if (type == VTYPE_ERROR) {
                mprintf("Error vector for '%s'", tmp);
                mprintf(" ");
                if (s->pprint) {
                        pprint_vector(g->error, s->pprint_scheme);
                } else {
                        print_vector(g->error);
                }
        }

        mprintf(" ");

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_show_matrix(char *cmd, char *fmt, struct session *s)
{
        uint32_t type;

        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        /* weights */
        if (sscanf(cmd, "showWeights %s %s", tmp1, tmp2) == 2)
                type = MTYPE_WEIGHTS;
        /* gradients */
        else if (sscanf(cmd, "showGradients %s %s", tmp1, tmp2) == 2)
                type = MTYPE_GRADIENTS;
        /* dynamic learning parameters */
        else if (sscanf(cmd, "showDynPars %s %s", tmp1, tmp2) == 2)
                type = MTYPE_DYN_PARS;
        else
                return false;

        struct group *fg = find_array_element_by_name(s->anp->groups, tmp1);
        struct group *tg = find_array_element_by_name(s->anp->groups, tmp2);

        if (fg == NULL) {
                eprintf("Cannot show matrix--no such group '%s'", tmp1);
                return true;
        }
        if (tg == NULL) {
                eprintf("Cannot show matrix--no such group '%s'", tmp2);
                return true;
        }

        struct projection *fg_to_tg = NULL;
        for (uint32_t i = 0; i < fg->out_projs->num_elements; i++) {
                if (((struct projection *)fg->out_projs->elements[i])->to == tg) {
                        fg_to_tg = (struct projection *)fg->out_projs->elements[i];
                        break;
                }
        }
        if (fg_to_tg) {
                if (type == MTYPE_WEIGHTS) {
                        mprintf("Weight matrix for projection '%s -> %s'", tmp1, tmp2);
                        mprintf(" ");
                        if (s->pprint) {
                                pprint_matrix(fg_to_tg->weights, s->pprint_scheme);
                        } else {
                                print_matrix(fg_to_tg->weights);
                        }
                }
                if (type == MTYPE_GRADIENTS) {
                        mprintf("Gradient matrix for projection '%s -> %s'", tmp1, tmp2);
                        mprintf(" ");                        
                        if (s->pprint) {
                                pprint_matrix(fg_to_tg->gradients, s->pprint_scheme);
                        } else {
                                print_matrix(fg_to_tg->gradients);
                        }
                }
                if (type == MTYPE_DYN_PARS) {
                        mprintf("Dynamic learning parameters for projection '%s -> %s'",
                                        tmp1, tmp2);
                        mprintf(" ");                        
                        if (s->pprint) {
                                pprint_matrix(fg_to_tg->dynamic_pars, s->pprint_scheme);
                        } else {
                                print_matrix(fg_to_tg->dynamic_pars);
                        }
                }

                mprintf(" ");
        } else {
                eprintf("Cannot show matrix--no projection between groups '%s' and '%s'",
                                tmp1, tmp2);
                return true;
        }

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_save_weights(char *cmd, char *fmt, struct session *s)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        if (save_weight_matrices(s->anp, tmp)) {
                mprintf("Saved weights ... \t\t ( %s )", tmp);
        } else {
                eprintf("Cannot save weights to file '%s'", tmp);
        }

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_load_weights(char *cmd, char *fmt, struct session *s)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        if (load_weight_matrices(s->anp, tmp)) {
                mprintf("Loaded weights ... \t\t ( %s )", tmp);
        } else {
                eprintf("Cannot load weights from file '%s'", tmp);
        }

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_toggle_pretty_printing(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        s->pprint = !s->pprint;

        if (s->pprint) {
                mprintf("Toggled pretty printing ... \t ( on )");
        } else {
                mprintf("Toggled pretty printing ... \t ( off )");
        }

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_set_color_scheme(char *cmd, char *fmt, struct session *s)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        /* blue and red */
        if (strcmp(tmp, "blue_red") == 0)
                s->pprint_scheme = SCHEME_BLUE_RED;
        /* blue and yellow */
        else if (strcmp(tmp, "blue_yellow") == 0)
                s->pprint_scheme = SCHEME_BLUE_YELLOW;
        /* grayscale */
        else if (strcmp(tmp, "grayscale") == 0)
                s->pprint_scheme = SCHEME_GRAYSCALE;
        /* spacepigs */
        else if (strcmp(tmp, "spacepigs") == 0)
                s->pprint_scheme = SCHEME_SPACEPIGS;
        /* moody blues */
        else if (strcmp(tmp, "moody_blues") == 0)
                s->pprint_scheme = SCHEME_MOODY_BLUES;
        /* for John */
        else if (strcmp(tmp, "for_john") == 0)
                s->pprint_scheme = SCHEME_FOR_JOHN;
        /* gray and orange */
        else if (strcmp(tmp, "gray_orange") == 0)
                s->pprint_scheme = SCHEME_GRAY_ORANGE;
        else {
                eprintf("Cannot set color scheme--no such scheme '%s'", tmp);
                return true;
        }

        mprintf("Set color scheme ... \t\t ( %s )", tmp);

        return true;
}

/**************************************************************************
 * Event-related potentials (ERP) module.
 *************************************************************************/

/**************************************************************************
 *************************************************************************/
bool cmd_erp_contrast(char *cmd, char *fmt, struct session *s)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE], tmp3[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp1, tmp2, tmp3) != 3)
                return false;

        struct group *gen = find_array_element_by_name(s->anp->groups, tmp1);
        if (gen == NULL) {
                eprintf("Cannot compute ERP correlates--no such group '%s'", tmp1);
                return true;
        }

        struct item *item1 = find_array_element_by_name(s->anp->asp->items, tmp2);
        struct item *item2 = find_array_element_by_name(s->anp->asp->items, tmp3);

        if (!item1) {
                eprintf("Cannot compute ERP correlates--no such item '%s'", tmp2);
                return true;
        }
        if (!item2) {
                eprintf("Cannot compute ERP correlates--no such item '%s'", tmp3);
                return true;
        }

        mprintf(" ");

        erp_contrast(s->anp, gen, item1, item2);

        mprintf(" ");

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_erp_generate_table(char *cmd, char *fmt, struct session *s)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE], tmp3[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp1, tmp2, tmp3) != 3)
                return false;

        struct group *n400_gen = find_array_element_by_name(s->anp->groups, tmp1);
        struct group *p600_gen = find_array_element_by_name(s->anp->groups, tmp2);

        if (n400_gen == NULL) {
                eprintf("Cannot compute ERP correlates--no such group '%s'", tmp1);
                return true;
        }
        if (p600_gen == NULL) {
                eprintf("Cannot compute ERP correlates--no such group '%s'", tmp2);
                return true;
        }

        erp_generate_table(s->anp, n400_gen, p600_gen, tmp3);

        return true;
}

/**************************************************************************
 * Distributed situation space (DSS) module.
 *************************************************************************/

/**************************************************************************
 *************************************************************************/
bool cmd_dss_test(char *cmd, char *fmt, struct session *s)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf("Testing network '%s':", s->anp->name);
        mprintf(" ");

        dss_test(s->anp);

        mprintf(" ");

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_dss_scores(char *cmd, char *fmt, struct session *s)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        struct set *set = find_array_element_by_name(s->anp->sets, tmp1);
        if (!set) {
                eprintf("Cannot compute scores--no such set '%s'", tmp1);
                return true;
        }

        struct item *item = find_array_element_by_name(s->anp->asp->items, tmp2);
        if (!item) {
                eprintf("Cannot compute scores--no such item '%s'", tmp2);
                return true;
        }

        mprintf(" ");

        dss_scores(s->anp, set, item);

        mprintf(" ");

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_dss_write_scores(char *cmd, char *fmt, struct session *s)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        struct set *set = find_array_element_by_name(s->anp->sets, tmp1);
        if (!set) {
                eprintf("Cannot compute scores--no such set '%s'", tmp1);
                return true;
        }

        mprintf(" ");

        dss_write_scores(s->anp, set, tmp2);

        mprintf(" ");

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_dss_inferences(char *cmd, char *fmt, struct session *s)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        double tmp3;
        if (sscanf(cmd, fmt, tmp1, tmp2, &tmp3) != 3)
                return false;

        struct set *set = find_array_element_by_name(s->anp->sets, tmp1);
        if (!set) {
                eprintf("Cannot compute inferences--no such set '%s'", tmp1);
                return true;
        }

        struct item *item = find_array_element_by_name(s->anp->asp->items, tmp2);
        if (!item) {
                eprintf("Cannot compute inferences--no such item '%s'", tmp2);
                return true;
        }

        if (tmp3 < -1.0 || tmp3 > 1.0) {
                eprintf("Cannot compute inferences--invalid score threshold '%lf'", tmp3);
                return true;

        }

        mprintf(" ");

        dss_inferences(s->anp, set, item, tmp3);

        mprintf(" ");

        return true;

}

/**************************************************************************
 *************************************************************************/
bool cmd_dss_word_information(char *cmd, char *fmt, struct session *s)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        struct item *item = find_array_element_by_name(s->anp->asp->items, tmp);
        if (!item) {
                eprintf("Cannot compute word information--no such item '%s'", tmp);
                return true;
        }
        
        mprintf("Testing network '%s' with item '%s':", s->anp->name, tmp);
        mprintf(" ");

        dss_word_information(s->anp, item);

        mprintf(" ");

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_dss_write_word_information(char *cmd, char *fmt,
                struct session *s)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        mprintf(" ");

        dss_write_word_information(s->anp, tmp);

        mprintf(" ");

        return true;
}

/**************************************************************************
 * Dynamic systems module.
 *************************************************************************/

/**************************************************************************
 *************************************************************************/
bool cmd_dynsys_test_item(char *cmd, char *fmt, struct session *s)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        struct group *group = find_array_element_by_name(s->anp->groups, tmp1);
        if (group == NULL) {
                eprintf("Cannot test network--no such group '%s'", tmp1);
                return true;
        }

        struct item *item = find_array_element_by_name(s->anp->asp->items, tmp2);
        if (!item) {
                eprintf("Cannot test network--no such item '%s'", tmp2);
                return true;
        }
        
        mprintf("Testing network '%s' with item '%s':", s->anp->name, tmp2);
        mprintf(" ");

        dynsys_test_item(s->anp, group, item);

        mprintf(" ");

        return true;
}
