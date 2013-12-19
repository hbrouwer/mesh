/*
 * cmd.c
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

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */
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

#include "mods/erps.h"

#define VTYPE_UNITS 0
#define VTYPE_ERROR 1

#define MTYPE_WEIGHTS   0
#define MTYPE_GRADIENTS 1
#define MTYPE_DYN_PARS  2

/**************************************************************************
 *************************************************************************/
void process_command(char *cmd, struct session *s)
{
        /* blank line or comment*/
        if (cmd[0] == '\0' || cmd[0] == '#') return;

        /* quit or exit */
        cmd_quit(cmd, "quit", s, "Leaving MESH."); 
        cmd_quit(cmd, "exit", s, "Leaving MESH.");

        /* load file */
        if (cmd_load_file(cmd,
                                "loadFile %s",
                                s,
                                "Loaded file ... \t\t ( %s )"
                                )) goto done;

        /* network commands */
        if (cmd_create_network(cmd,
                                "createNetwork %s %s",
                                s,
                                "Created network ... \t\t ( %s :: %s )"
                                )) goto done;
        if (cmd_dispose_network(cmd,
                                "disposeNetwork %s %s",
                                s,
                                "Disposed network ... \t\t ( %s )"
                                )) goto done;
        if (cmd_list_networks(cmd,
                                "listNetworks",
                                s,
                                "Available networks:"
                                )) goto done;
        if (cmd_change_network(cmd,
                                "changeNetwork %s", s,
                                "Changed to network ... \t ( %s )"
                                )) goto done;

        /*
         * All commands below require an active network, hence 
         * report an error if there is none.
         */
        if (!s->anp) {
                eprintf("Cannot process command: %s", cmd);
                eprintf("No active network.");
                return;
        }

        /* group commands */
        if (cmd_create_group(cmd,
                                "createGroup %s %d",
                                s->anp,
                                "Created group ... \t\t ( %s :: %d )"
                                )) goto done;
        if (cmd_dispose_group(cmd,
                                "disposeGroup %s",
                                s->anp,
                                "Disposed group ... \t\t ( %s )"
                                )) goto done;
        if (cmd_list_groups(cmd,
                                "listGroups",
                                s->anp,
                                "Available groups:"
                                )) goto done;
        if (cmd_attach_bias(cmd,
                                "attachBias %s",
                                s->anp,
                                "Attached bias to group ... \t ( %s -> %s )"
                                )) goto done;
        if (cmd_set_input_group(cmd,
                                "set InputGroup %s",
                                s->anp,
                                "Set input group ... \t\t ( %s )"
                                )) goto done;
        if (cmd_set_output_group(cmd,
                                "set OutputGroup %s",
                                s->anp,
                                "Set output group ... \t\t ( %s )"
                                )) goto done;
        if (cmd_set_act_func(cmd,
                                "set ActFunc %s %s",
                                s->anp,
                                "Set activation function ... \t ( %s :: %s )"
                                )) goto done;
        if (cmd_set_err_func(cmd,
                                "set ErrFunc %s %s",
                                s->anp,
                                "Set error function ... \t\t ( %s :: %s )"
                                )) goto done;
        if (cmd_toggle_act_lookup(cmd,
                                "toggleActLookup",
                                s->anp,
                                "Toggle activation lookup ... \t ( %s )"
                                )) goto done;

        /* projection commands */
        if (cmd_create_projection(cmd,
                                "createProjection %s %s",
                                s->anp,
                                "Created projection ... \t\t ( %s -> %s )"
                                )) goto done;
        if (cmd_dispose_projection(cmd,
                                "disposeProjection %s %s",
                                s->anp,
                                "Disposed projection ... \t\t ( %s -> %s )"
                                )) goto done;        
        if (cmd_create_elman_projection(cmd,
                                "createElmanProjection %s %s",
                                s->anp,
                                "Created Elman projection ... \t ( %s -> %s )"
                                )) goto done;
        if (cmd_dispose_elman_projection(cmd,
                                "disposeElmanProjection %s %s",
                                s->anp,
                                "Disposed Elman projection ... \t\t ( %s -> %s )"
                                )) goto done;
        if (cmd_list_projections(cmd,
                                "listProjections",
                                s->anp,
                                "Available projections:"
                                )) goto done;
        if (cmd_freeze_projection(cmd,
                                "freezeProjection %s %s",
                                s->anp,
                                "Froze projection ... \t\t ( %s -> %s )"
                                )) goto done;
        
        /* set integer parameters */
        if (cmd_set_int_parameter(cmd,
                                "set BatchSize %d",
                                &s->anp->batch_size,
                                "Set batch size ... \t\t ( %d )"
                                )) goto done;
        if (cmd_set_int_parameter(cmd,
                                "set MaxEpochs %d",
                                &s->anp->max_epochs,
                                "Set maximum #epochs ... \t ( %d )"
                                )) goto done;
        if (cmd_set_int_parameter(cmd,
                                "set ReportAfter %d",
                                &s->anp->report_after,
                                "Set report after (#epochs) ... \t ( %d )"
                                )) goto done;
        if (cmd_set_int_parameter(cmd,
                                "set RandomSeed %d",
                                &s->anp->random_seed,
                                "Set random seed ... \t\t ( %d )"
                                )) goto done;
        if (cmd_set_int_parameter(cmd,
                                "set BackTicks %d",
                                &s->anp->back_ticks,
                                "Set BPTT back ticks ... \t ( %d )"
                                )) goto done;
        
        /* set double parameters */
        if (cmd_set_double_parameter(cmd,
                                "set RandomMu %lf",
                                &s->anp->random_mu,
                                "Set random Mu ... \t\t ( %lf )"
                                )) goto done;
        if (cmd_set_double_parameter(cmd,
                                "set RandomSigma %lf",
                                &s->anp->random_sigma,
                                "Set random Sigma ... \t\t ( %lf )"
                                )) goto done;
        if (cmd_set_double_parameter(cmd,
                                "set RandomMin %lf",
                                &s->anp->random_min,
                                "Set random minimum ... \t\t ( %lf )"
                                )) goto done;
        if (cmd_set_double_parameter(cmd,
                                "set RandomMax %lf",
                                &s->anp->random_max,
                                "Set random maximum ... \t\t ( %lf )"
                                )) goto done;
        if (cmd_set_double_parameter(cmd,
                                "set LearningRate %lf",
                                &s->anp->learning_rate,
                                "Set learning rate ... \t\t ( %lf )"
                                )) goto done;
        if (cmd_set_double_parameter(cmd,
                                "set LRScaleFactor %lf",
                                &s->anp->lr_scale_factor,
                                "Set LR scale factor ... \t ( %lf )"
                                )) goto done;
        if (cmd_set_double_parameter(cmd,
                                "set LRScaleAfter %lf",
                                &s->anp->lr_scale_after,
                                "Set LR scale after (%%epochs) ... ( %lf )"
                                )) goto done;
        if (cmd_set_double_parameter(cmd,
                                "set Momentum %lf",
                                &s->anp->momentum,
                                "Set momentum ... \t\t ( %lf )"
                                )) goto done;
        if (cmd_set_double_parameter(cmd,
                                "set MNScaleFactor %lf",
                                &s->anp->mn_scale_factor,
                                "Set MN scale factor ... \t ( %lf )"
                                )) goto done;
        if (cmd_set_double_parameter(cmd,
                                "set MNScaleAfter %lf",
                                &s->anp->mn_scale_after,
                                "Set MN scale after (%%epochs) ... ( %lf )"
                                )) goto done;
        if (cmd_set_double_parameter(cmd,
                                "set WeightDecay %lf",
                                &s->anp->weight_decay,
                                "Set weight decay ... \t\t ( %lf )"
                                )) goto done;
        if (cmd_set_double_parameter(cmd,
                                "set WDScaleFactor %lf",
                                &s->anp->wd_scale_factor,
                                "Set WD scale factor ... \t ( %lf )"
                                )) goto done;
        if (cmd_set_double_parameter(cmd,
                                "set WDScaleAfter %lf",
                                &s->anp->wd_scale_after,
                                "Set WD scale after (%%epochs) ... ( %lf )"
                                )) goto done;
        if (cmd_set_double_parameter(cmd,
                                "set ErrorThreshold %lf",
                                &s->anp->error_threshold,
                                "Set error threshold ... \t ( %lf )"
                                )) goto done;
        if (cmd_set_double_parameter(cmd,
                                "set TargetRadius %lf",
                                &s->anp->target_radius,
                                "Set target radius ... \t\t ( %lf )"
                                )) goto done;
        if (cmd_set_double_parameter(cmd,
                                "set ZeroErrorRadius %lf",
                                &s->anp->zero_error_radius,
                                "Set zero-error radius ... \t ( %lf )"
                                )) goto done;
        if (cmd_set_double_parameter(cmd,
                                "set RpropInitUpdate %lf",
                                &s->anp->rp_init_update,
                                "Set init update (for Rprop) ...  ( %lf )"
                                )) goto done;
        if (cmd_set_double_parameter(cmd,
                                "set RpropEtaPlus %lf",
                                &s->anp->rp_eta_plus,
                                "Set Eta+ (for Rprop) ... \t ( %lf )"
                                )) goto done;
        if (cmd_set_double_parameter(cmd,
                                "set RpropEtaMinus %lf",
                                &s->anp->rp_eta_minus,
                                "Set Eta- (for Rprop) ... \t ( %lf )"
                                )) goto done;
        if (cmd_set_double_parameter(cmd,
                                "set DBDRateIncrement %lf",
                                &s->anp->dbd_rate_increment,
                                "Set increment rate (for Delta-Bar-Delta) ... \t ( %lf )"
                                )) goto done;
        if (cmd_set_double_parameter(cmd,
                                "set DBDRateDecrement %lf",
                                &s->anp->dbd_rate_decrement,
                                "Set decrement rate (for Delta-Bar-Delta) ... \t ( %lf )"
                                )) goto done;

        /* item set commands */
        if (cmd_load_set(cmd,
                                "loadSet %s %s",
                                s->anp,
                                "Loaded set ... \t\t\t ( %s => %s :: %d )"
                                )) goto done;
        if (cmd_dispose_set(cmd,
                                "disposeSet %s",
                                s->anp,
                                "Disposed set ... \t\t ( %s )"
                                )) goto done;
        if (cmd_list_sets(cmd,
                                "listSets",
                                s->anp,
                                "Available sets:"
                                )) goto done;        
        if (cmd_change_set(cmd, "changeSet %s",
                                s->anp,
                                "Changed to set ... \t\t ( %s )"
                                )) goto done;
        if (cmd_list_items(cmd,
                                "listItems",
                                s->anp,
                                "Available items in set '%s':"
                                )) goto done;
        if (cmd_set_training_order(cmd,
                                "set TrainingOrder %s",
                                &s->anp->training_order,
                                "Set training order ... \t\t ( %s )"
                                )) goto done;

        /* ranzomization, learning, and updating algorithms */
        if (cmd_set_rand_algorithm(cmd,
                                "set RandomAlgorithm %s",
                                s->anp,
                                "Set random algorithm ... \t ( %s )"
                                )) goto done;
        if (cmd_set_learning_algorithm(cmd,
                                "set LearningAlgorithm %s",
                                s->anp,
                                "Set learning algorithm ... \t ( %s )"
                                )) goto done;
        if (cmd_set_update_algorithm(cmd,
                                "set UpdateAlgorithm %s",
                                s->anp,
                                "Set update algorithm ... \t ( %s )"
                                )) goto done;

        /* similarity metric */
        if (cmd_set_similarity_metric(cmd,
                                "set SimilarityMetric %s",
                                s->anp,
                                "Set similarity metric ... \t ( %s )"
                                )) goto done;

        /* initialization */
        if (cmd_init(cmd, "init", s->anp, "Initialized network '%s'")) goto done;

        /*
         * All commands below require an initialized network, hence 
         * report an error if the active network is not.
         */
        if (!s->anp->initialized) {
                eprintf("Cannot process command: %s", cmd);
                eprintf("unitizialized network--use 'init' command to initialize");
                return;
        }

        /* reset, training, and testing */
        if (cmd_reset(cmd,
                                "reset",
                                s->anp,
                                "Reset network '%s'"
                                )) goto done;
        if (cmd_train(cmd,
                                "train", 
                                s->anp,
                                "Training network '%s'"
                                )) goto done;
        if (cmd_test(cmd,
                                "test",
                                s->anp,
                                "Testing network '%s'"
                                )) goto done;
        if (cmd_test_item(cmd,
                                "testItem \"%[^\"]\"",
                                s,
                                "Testing network '%s' with item '%s'"
                                )) goto done;

        /* similarity and confusion matrices */
        if (cmd_similarity_matrix(cmd,
                                "similarityMatrix",
                                s->anp,
                                "Computing similarity matrix for network '%s' ..."
                                )) goto done;
        if (cmd_confusion_matrix(cmd,
                                "confusionMatrix",
                                s->anp,
                                "Computing confusion matrix for network '%s' ..."
                                )) goto done;

        /* weight statistics */
        if (cmd_weight_stats(cmd,
                                "weightStats",
                                s->anp,
                                "Weight statistics for network '%s'"
                                )) goto done;
       
        /* show vectors and matrices */
        if (cmd_show_vector(cmd,
                                "showUnits %s",
                                s,
                                "Unit vector for '%s'",
                                VTYPE_UNITS
                                )) goto done;
        if (cmd_show_vector(cmd,
                                "showError %s",
                                s,
                                "Error vector for '%s'",
                                VTYPE_ERROR
                                )) goto done;
        if (cmd_show_matrix(cmd,
                                "showWeights %s %s",
                                s,
                                "Weight matrix for projection '%s -> %s'",
                                MTYPE_WEIGHTS
                                )) goto done;
        if (cmd_show_matrix(cmd,
                                "showGradients %s %s",
                                s,
                                "Gradient matrix for projection '%s -> %s'",
                                MTYPE_GRADIENTS
                                )) goto done;
        if (cmd_show_matrix(cmd,
                                "showDynPars %s %s",
                                s,
                                "Dynamic learning parameters for projection '%s -> %s'",
                                MTYPE_DYN_PARS
                                )) goto done;
        
        /* weight matrix saving and loading */
        if (cmd_save_weights(cmd,
                                "saveWeights %s",
                                s->anp,
                                "Saved weights ... \t\t ( %s )"
                                )) goto done;
        if (cmd_load_weights(cmd,
                                "loadWeights %s",
                                s->anp,
                                "Loaded weights ... \t\t ( %s )"
                                )) goto done;

        /* pretty printing and color schemes */
        if (cmd_toggle_pretty_printing(cmd,
                                "togglePrettyPrinting",
                                s,
                                "Toggled pretty printing ... \t ( %s )"
                                )) goto done;        
        if (cmd_set_colorscheme(cmd,
                                "set ColorScheme %s",
                                s,
                                "Set color scheme ... \t\t ( %s )"
                                )) goto done;

        /* event-related potentials module commands */
        if (cmd_erp_generate_table(cmd,
                                "erpGenerateTable %s",
                                s,
                                "ERP amplitude table:"
                                )) goto done;

        /* invalid command */
        if (strlen(cmd) > 1) {
                eprintf("invalid command: %s", cmd); 
                eprintf("(type 'help' for a list of valid commands)");
        }

done:
        return;
}

/**************************************************************************
 *************************************************************************/
void cmd_quit(char *cmd, char *fmt, struct session *s, char *msg)
{
        if (strcmp(cmd, fmt) != 0)
                return;
        
        mprintf(msg);
        
        dispose_session(s);
        exit(EXIT_SUCCESS);
}

/**************************************************************************
 *************************************************************************/
bool cmd_load_file(char *cmd, char *fmt, struct session *s, char *msg)
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

        mprintf(msg, tmp);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_create_network(char *cmd, char *fmt, struct session *s, char *msg)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        uint32_t type = 0;
        if (strcmp(tmp2, "ffn") == 0)
                type = TYPE_FFN;
        else if (strcmp(tmp2, "srn") == 0)
                type = TYPE_SRN;
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

        mprintf(msg, tmp1, tmp2);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_dispose_network(char *cmd, char *fmt, struct session *s, char *msg)
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

        mprintf(msg, tmp);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_list_networks(char *cmd, char *fmt, struct session *s, char *msg)
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        mprintf(msg);

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
bool cmd_change_network(char *cmd, char *fmt, struct session *s, char *msg)
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

        mprintf(msg, tmp);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_create_group(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp[MAX_ARG_SIZE];
        uint32_t tmp_int;
        if (sscanf(cmd, fmt, tmp, &tmp_int) != 2)
                return false;

        if (find_array_element_by_name(n->groups, tmp)) {
                eprintf("Cannot create group--group '%s' already exists in network '%s'",
                                tmp, n->name);
                return true;
        }

        struct group *g = create_group(tmp, tmp_int, false, false);
        add_to_array(n->groups, g);

        mprintf(msg, tmp, tmp_int);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_dispose_group(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        struct group *g = find_array_element_by_name(n->groups, tmp);
        if (g == NULL) {
                eprintf("Cannot dispose group--no such group '%s'", tmp);
                return true;
        }

        /* 
         * Remove any outgoing projections from a group g' to
         * group g.
         */
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

        /*
         * Remove any incoming projections to group g from a
         * group g'.
         */
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

        /*
         * Remove any Elman projections from a group g' to
         * group g.
         */
        for (uint32_t i = 0; i < n->groups->num_elements; i++) {
                struct group *fg = n->groups->elements[i];
                for (uint32_t j = 0; j < fg->ctx_groups->num_elements; j++) {
                        if (fg->ctx_groups->elements[j] == g) {
                                remove_from_array(fg->ctx_groups, g);
                                break;
                        }
                }
        }

        remove_from_array(n->groups, g);
        dispose_group(g);

        mprintf(msg, tmp);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_list_groups(char *cmd, char *fmt, struct network *n, char *msg)
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        mprintf(msg);

        if (n->groups->num_elements == 0) {
                cprintf("(No groups)\n");
        } else {
                for (uint32_t i = 0; i < n->groups->num_elements; i++) {
                        struct group *g = n->groups->elements[i];
                        cprintf("* %s :: %d", g->name, g->vector->size);
                        if (g == n->input) {
                                cprintf("\t\t <- Input group\n");
                        } else if (g == n->output) {
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
bool cmd_attach_bias(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        struct group *g = find_array_element_by_name(n->groups, tmp);
        if (g == NULL) {
                eprintf("Cannot attach bias group--no such group '%s'", tmp);
                return true;
        }

        char *tmp_bias;
        size_t block_size = (strlen(tmp) + 1) * sizeof(char);
        if (!(tmp_bias = malloc(block_size)))
                goto error_out;
        memset(tmp_bias, 0, block_size);
        asprintf(&tmp_bias, "%s_bias", tmp);
        if (find_array_element_by_name(n->groups, tmp_bias)) {
                eprintf("Cannot attach bias group--group '%s' already exists in network '%s'",
                                tmp_bias, n->name);
                return true;
        }

        struct group *bg = attach_bias_group(n, g);

        mprintf(msg, bg->name, g->name);

        return true;

error_out:
        perror("[cmd_attach_bias()]");
        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_set_input_group(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        struct group *g = find_array_element_by_name(n->groups, tmp);
        if (g == NULL) {
                eprintf("Cannot set input group--no such group '%s'", tmp);
                return true;
        }

        n->input = g;

        mprintf(msg, tmp);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_set_output_group(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        struct group *g = find_array_element_by_name(n->groups, tmp);
        if (g == NULL) {
                eprintf("Cannot set output group--no such group '%s'", tmp);
                return true;
        }

        n->output = g;

        mprintf(msg, tmp);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_set_act_func(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        struct group *g = find_array_element_by_name(n->groups, tmp1);
        if (g == NULL) {
                eprintf("Cannot set activation function--no such group '%s'",
                                tmp1);
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
        } 
        
        else {
                eprintf("Cannot set activation function--no such activation function '%s'", tmp2);
                return true;
        }

        mprintf(msg, tmp1, tmp2);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_set_err_func(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        struct group *g = find_array_element_by_name(n->groups, tmp1);
        if (g == NULL) {
                eprintf("Cannot set error function--no such group '%s'",
                                tmp1);
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
        }

        else {
                eprintf("Cannot set error function--no such error function '%s'", tmp2);
                return true;
        }

        mprintf(msg, tmp1, tmp2);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_toggle_act_lookup(char *cmd, char *fmt, struct network *n, char *msg)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        n->act_lookup = !n->act_lookup;
        
        if (n->act_lookup) {
                initialize_act_lookup_vectors(n);
                mprintf(msg, "on");
        } else {
                mprintf(msg, "off");
        }

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_create_projection(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        struct group *fg = find_array_element_by_name(n->groups, tmp1);
        struct group *tg = find_array_element_by_name(n->groups, tmp2);

        if (fg == NULL) {
                eprintf("Cannot set projection--no such group '%s'",
                                tmp1);
                return true;
        }
        if (tg == NULL) {
                eprintf("Cannot set projection--no such group '%s'",
                                tmp2);
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
                struct matrix *weights = create_matrix(
                                fg->vector->size,
                                tg->vector->size);
                struct matrix *gradients = create_matrix(
                                fg->vector->size,
                                tg->vector->size);
                struct matrix *prev_gradients = create_matrix(
                                fg->vector->size,
                                tg->vector->size);
                struct matrix *prev_deltas = create_matrix(
                                fg->vector->size,
                                tg->vector->size);
                struct matrix *dynamic_pars = create_matrix(
                                fg->vector->size,
                                tg->vector->size);

                struct projection *op;
                op = create_projection(tg, weights, gradients, prev_gradients,
                                prev_deltas, dynamic_pars, false);
                add_to_array(fg->out_projs, op);

                struct projection *ip;
                ip = create_projection(fg, weights, gradients, prev_gradients,
                                prev_deltas, dynamic_pars, false);
                add_to_array(tg->inc_projs, ip);

        }

        mprintf(msg, tmp1, tmp2);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_dispose_projection(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        struct group *fg = find_array_element_by_name(n->groups, tmp1);
        struct group *tg = find_array_element_by_name(n->groups, tmp2);

        if (fg == NULL) {
                eprintf("Cannot dispose projection--no such group '%s'",
                                tmp1);
                return true;
        }
        if (tg == NULL) {
                eprintf("Cannot dispose projection--no such group '%s'",
                                tmp2);
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
        } else {
                eprintf("Cannot dispose projection--no projection between groups '%s' and '%s')",
                                tmp1, tmp2);
                return true;
        }

        mprintf(msg, tmp1, tmp2);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_create_elman_projection(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        struct group *fg = find_array_element_by_name(n->groups, tmp1);
        struct group *tg = find_array_element_by_name(n->groups, tmp2);

        if (fg == NULL) {
                eprintf("Cannot set Elman-projection--no such group '%s'", tmp1);
                return true;
        }
        if (tg == NULL) {
                eprintf("Cannot set Elman-projection--no such group '%s'", tmp2);
                return true;
        }
        if (fg == tg) {
                eprintf("Cannot set Elman-projection--projection is recurrent for group '%s'",
                                fg->name);
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
        reset_context_groups(n);

        mprintf(msg, tmp1, tmp2);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_dispose_elman_projection(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        struct group *fg = find_array_element_by_name(n->groups, tmp1);
        struct group *tg = find_array_element_by_name(n->groups, tmp2);

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

        mprintf(msg, tmp1, tmp2);
        
        return true;

}

/**************************************************************************
 *************************************************************************/
bool cmd_list_projections(char *cmd, char *fmt, struct network *n, char *msg)
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        mprintf(msg);

        for (uint32_t i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];
                
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
bool cmd_freeze_projection(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        struct group *fg = find_array_element_by_name(n->groups, tmp1);
        struct group *tg = find_array_element_by_name(n->groups, tmp2);

        if (fg == NULL) {
                eprintf("Cannot freeze projection--no such group '%s'",
                                tmp1);
                return true;
        }
        if (tg == NULL) {
                eprintf("Cannot freeze projection--no such group '%s'",
                                tmp2);
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

        mprintf(msg, tmp1, tmp2);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_set_int_parameter(char *cmd, char *fmt, uint32_t *par, char *msg)
{
        if (sscanf(cmd, fmt, par) != 1)
                return false;

        mprintf(msg, *par);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_set_double_parameter(char *cmd, char *fmt, double *par, char *msg)
{
        if (sscanf(cmd, fmt, par) != 1)
                return false;

        mprintf(msg, *par);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_load_set(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        if (!n->input) {
                eprintf("Cannot load set--input group size unknown");
                return true;
        }
        if (!n->output) {
                eprintf("Cannot load set--output group size unknown");
                return true;
        }

        if (find_array_element_by_name(n->sets, tmp1)) {
                eprintf("Cannot load set--set '%s' already exists", tmp1);
                return true;
        }

        struct set *s = load_set(tmp1, tmp2, n->input->vector->size, n->output->vector->size);
        if (!s) {
                eprintf("Cannot load set--no such file '%s'", tmp2);
                return true;
        }
                
        add_to_array(n->sets, s);
        n->asp = s;

        mprintf(msg, tmp2, s->name, s->items->num_elements);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_dispose_set(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        struct set *s = find_array_element_by_name(n->sets, tmp);
        if (!s) {
                eprintf("Cannot change to set--no such set '%s'", tmp);
                return true;
        }

        if (s == n->asp)
                n->asp = NULL;
        remove_from_array(n->sets, s);
        dispose_set(s);

        mprintf(msg, tmp);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_list_sets(char *cmd, char *fmt, struct network *n, char *msg)
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        mprintf(msg);

        if (n->sets->num_elements == 0) {
                cprintf("(No sets)\n");
        } else {
                for (uint32_t i = 0; i < n->sets->num_elements; i++) {
                        struct set *s = n->sets->elements[i];
                        cprintf("* %s", s->name);
                        if (s == n->asp) {
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
bool cmd_change_set(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        struct set *s = find_array_element_by_name(n->sets, tmp);
        if (!s) {
                eprintf("Cannot change to set--no such set '%s'", tmp);
                return true;
        }

        n->asp = s;

        mprintf(msg, tmp);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_list_items(char *cmd, char *fmt, struct network *n, char *msg)
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        if (!n->asp) {
                eprintf("Cannot list items--no active set");
                return true;
        }

        mprintf(msg, n->asp->name);

        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                struct item *item = n->asp->items->elements[i];
                cprintf("* \"%s\" %d \"%s\"\n", item->name, item->num_events, item->meta);
        }

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_set_training_order(char *cmd, char *fmt, uint32_t *training_order,
                char *msg)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        if (strcmp(tmp, "ordered") == 0)
                *training_order = TRAIN_ORDERED;
        else if (strcmp(tmp, "permuted") == 0)
                *training_order = TRAIN_PERMUTED;
        else if (strcmp(tmp, "randomized") == 0)
                *training_order = TRAIN_RANDOMIZED;
        else {
                eprintf("Invalid training order '%s'", tmp);
                return true;
        }

        mprintf(msg, tmp);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_set_rand_algorithm(char *cmd, char *fmt, struct network *n,
                char *msg)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        if (strcmp(tmp, "gaussian") == 0)
                n->random_algorithm = randomize_gaussian;
        else if (strcmp(tmp, "range") == 0)
                n->random_algorithm = randomize_range;
        else if (strcmp(tmp, "nguyen_widrow") == 0)
                n->random_algorithm = randomize_nguyen_widrow;
        else if (strcmp(tmp, "fan_in") == 0)
                n->random_algorithm = randomize_fan_in;
        else if (strcmp(tmp, "binary") == 0)
                n->random_algorithm = randomize_binary;
        else {
                eprintf("Invalid randomization algorithm '%s'", tmp);
                return true;
        }

        if (n->random_algorithm)
                mprintf(msg, tmp);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_set_learning_algorithm(char *cmd, char *fmt, struct network *n,
                char *msg)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        if (strlen(tmp) == 2 && strcmp(tmp, "bp") == 0)
                n->learning_algorithm = train_network_with_bp;
        else if (strlen(tmp) == 4 && strcmp(tmp, "bptt") == 0)
                n->learning_algorithm = train_network_with_bptt;
        else {
                eprintf("Invalid learning algorithm '%s'", tmp);
                return true;
        }
        
        if (n->learning_algorithm)
                mprintf(msg, tmp);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_set_update_algorithm(char *cmd, char *fmt, struct network *n,
                char *msg)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        if (strcmp(tmp, "steepest") == 0) {
                n->update_algorithm = bp_update_sd;
                n->sd_type = SD_DEFAULT;
        }
        else if (strcmp(tmp, "bounded") == 0) {
                n->update_algorithm = bp_update_sd;
                n->sd_type = SD_BOUNDED;
        }
        else if (strcmp(tmp, "rprop+") == 0) {
                n->update_algorithm = bp_update_rprop;
                n->rp_type = RPROP_PLUS;
        }
        else if (strcmp(tmp, "rprop+") == 0) {
                n->update_algorithm = bp_update_rprop;
                n->rp_type = RPROP_PLUS;
        }
        else if (strcmp(tmp, "rprop-") == 0) {
                n->update_algorithm = bp_update_rprop;
                n->rp_type = RPROP_MINUS;
        }
        else if (strcmp(tmp, "irprop+") == 0) {
                n->update_algorithm = bp_update_rprop;
                n->rp_type = IRPROP_PLUS;
        }
        else if (strcmp(tmp, "irprop-") == 0) {
                n->update_algorithm = bp_update_rprop;
                n->rp_type = IRPROP_MINUS;
        }        
        else if (strcmp(tmp, "qprop") == 0) {
                n->update_algorithm = bp_update_qprop;
        }
        else if (strcmp(tmp, "dbd") == 0) {
                n->update_algorithm = bp_update_dbd;
        }
        else {
                eprintf("Invalid update algorithm '%s'", tmp);
                return true;
        }

        if (n->update_algorithm)
                mprintf(msg, tmp);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_set_similarity_metric(char *cmd, char *fmt, struct network *n,
                char *msg)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        if (strcmp(tmp, "inner_product") == 0)
                n->similarity_metric = inner_product;
        else if (strcmp(tmp, "harmonic_mean") == 0)
                n->similarity_metric = harmonic_mean;
        else if (strcmp(tmp, "cosine") == 0)
                n->similarity_metric = cosine;
        else if (strcmp(tmp, "tanimoto") == 0)
                n->similarity_metric = tanimoto;
        else if (strcmp(tmp, "dice") == 0)
                n->similarity_metric = dice;
        else if (strcmp(tmp, "pearson_correlation") == 0)
                n->similarity_metric = pearson_correlation;
        else {
                eprintf("Invalid similarity metric '%s'", tmp);
                return true;
        }

        if (n->similarity_metric)
                mprintf(msg, tmp);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_init(char *cmd, char *fmt, struct network *n, char *msg)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        init_network(n);

        mprintf(msg, n->name);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_reset(char *cmd, char *fmt, struct network *n, char *msg)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        reset_network(n);

        mprintf(msg, n->name);

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_train(char *cmd, char *fmt, struct network *n, char *msg)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf(msg, n->name);
        mprintf(" ");

        train_network(n);

        mprintf(" ");

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_test(char *cmd, char *fmt, struct network *n, char *msg)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf(msg, n->name);
        mprintf(" ");

        test_network(n);

        mprintf(" ");

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_test_item(char *cmd, char *fmt, struct session *s, char *msg)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        struct item *item = find_array_element_by_name(s->anp->asp->items, tmp);
        if (!item) {
                eprintf("Cannot test network--no such item '%s'", tmp);
                return true;
        }

        mprintf(msg, s->anp->name, tmp);
        mprintf(" ");

        test_network_with_item(s->anp, item, s->pprint, s->pprint_scheme);

        mprintf(" ");

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_similarity_matrix(char *cmd, char *fmt, struct network *n,
                char *msg)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf(msg, n->name);
        mprintf(" ");

        similarity_matrix(n);

        mprintf(" ");

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_confusion_matrix(char *cmd, char *fmt, struct network *n,
                char *msg)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf(msg, n->name);
        mprintf(" ");

        confusion_matrix(n);

        mprintf(" ");

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_weight_stats(char *cmd, char *fmt, struct network *n, char *msg)
{
        if (strlen(cmd) != strlen(fmt) || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf(msg, n->name);
        mprintf(" ");

        struct weight_stats *ws = create_weight_statistics(n);
     
        pprintf("Number of weights:\t\t%d\n", ws->num_weights);
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
bool cmd_show_vector(char *cmd, char *fmt, struct session *s, char *msg,
                uint32_t type)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        struct group *g = find_array_element_by_name(s->anp->groups, tmp);
        if (g == NULL) {
                eprintf("Cannot show vector--no such group '%s'", tmp);
                return true;
        }

        mprintf(msg, tmp);
        mprintf(" ");

        if (type == VTYPE_UNITS) {
                if (s->pprint) {
                        pprint_vector(g->vector, s->pprint_scheme);
                } else {
                        print_vector(g->vector);
                }
        }
        if (type == VTYPE_ERROR) {
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
bool cmd_show_matrix(char *cmd, char *fmt, struct session *s, char *msg,
                uint32_t type)
{
        char tmp1[MAX_ARG_SIZE], tmp2[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        struct group *fg = find_array_element_by_name(s->anp->groups, tmp1);
        struct group *tg = find_array_element_by_name(s->anp->groups, tmp2);

        if (fg == NULL) {
                eprintf("Cannot show matrix--no such group '%s'",
                                tmp1);
                return true;
        }
        if (tg == NULL) {
                eprintf("Cannot show matrix--no such group '%s'",
                                tmp2);
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
                mprintf(msg, tmp1, tmp2);
                mprintf(" ");

                if (type == MTYPE_WEIGHTS) {
                        if (s->pprint) {
                                pprint_matrix(fg_to_tg->weights, s->pprint_scheme);
                        } else {
                                print_matrix(fg_to_tg->weights);
                        }
                }
                if (type == MTYPE_GRADIENTS) {
                        if (s->pprint) {
                                pprint_matrix(fg_to_tg->gradients, s->pprint_scheme);
                        } else {
                                print_matrix(fg_to_tg->gradients);
                        }
                }
                if (type == MTYPE_DYN_PARS) {
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
bool cmd_save_weights(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        if (save_weight_matrices(n, tmp)) {
                mprintf(msg, tmp);
        } else {
                eprintf("Cannot save weights to file '%s'", tmp);
        }

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_load_weights(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        if (load_weight_matrices(n, tmp)) {
                mprintf(msg, tmp);
        } else {
                eprintf("Cannot load weights from file '%s'", tmp);
        }

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_toggle_pretty_printing(char *cmd, char *fmt, struct session *s, char *msg)
{
        if (strlen(cmd) != strlen(fmt) 
                        || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        s->pprint = !s->pprint;

        if (s->pprint) {
                mprintf(msg, "on");
        } else {
                mprintf(msg, "off");
        }

        return true;
}

/**************************************************************************
 *************************************************************************/
bool cmd_set_colorscheme(char *cmd, char *fmt, struct session *s, char *msg)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        if (strcmp(tmp, "blue_red") == 0)
                s->pprint_scheme = SCHEME_BLUE_RED;
        else if (strcmp(tmp, "blue_yellow") == 0)
                s->pprint_scheme = SCHEME_BLUE_YELLOW;
        else if (strcmp(tmp, "grayscale") == 0)
                s->pprint_scheme = SCHEME_GRAYSCALE;
        else if (strcmp(tmp, "spacepigs") == 0)
                s->pprint_scheme = SCHEME_SPACEPIGS;
        else if (strcmp(tmp, "moody_blues") == 0)
                s->pprint_scheme = SCHEME_MOODY_BLUES;
        else if (strcmp(tmp, "for_john") == 0)
                s->pprint_scheme = SCHEME_FOR_JOHN;
        else if (strcmp(tmp, "gray_orange") == 0)
                s->pprint_scheme = SCHEME_GRAY_ORANGE;
        else {
                eprintf("Cannot set color scheme--no such scheme '%s'", tmp);
                return true;
        }

        mprintf(msg, tmp);

        return true;
}

/**************************************************************************
 * Module commands
 *************************************************************************/

/**************************************************************************
 *************************************************************************/
bool cmd_erp_generate_table(char *cmd, char *fmt, struct session *s, char *msg)
{
        char tmp[MAX_ARG_SIZE];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        erp_generate_table(s->anp, tmp);

        return true;
}
