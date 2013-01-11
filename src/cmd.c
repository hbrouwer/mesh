/*
 * cmd.c
 *
 * Copyright 2012, 2013 Harm Brouwer <me@hbrouwer.eu>
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

#include "act.h"
#include "bp.h"
#include "cmd.h"
#include "engine.h"
#include "error.h"
#include "main.h"
#include "math.h"
#include "matrix.h"
#include "network.h"
#include "pprint.h"
#include "set.h"
#include "stats.h"

#define VTYPE_UNITS 0
#define VTYPE_ERROR 1

#define MTYPE_WEIGHTS 0
#define MTYPE_GRADIENTS 1
#define MTYPE_DYN_PARS 2

void process_command(char *cmd, struct session *s)
{
        /* blank line */
        if (cmd[0] == '\0') {
                return;
        }

        /* comment */
        if (cmd[0] == '#') {
                printf("%s\n", cmd);
                return;
        }

        cmd_quit(cmd, "quit", s, "Quitting...");
        cmd_quit(cmd, "exit", s, "Quitting...");

        if (cmd_create_network(cmd, "createNetwork %s %s", s,
                                "created network: [%s:%s]"))
                return;
        if (cmd_load_network(cmd, "loadNetwork %s", s,
                                "loaded network: [%s]"))
                return;
        if (cmd_dispose_network(cmd, "disposeNetwork %s %s", s,
                                "disposed network: [%s]"))
                return;
        if (cmd_list_networks(cmd, "listNetworks", s,
                                "network listing:"))
                return;
        if (cmd_change_network(cmd, "changeNetwork %s", s,
                                "changed to network: [%s]"))
                return;        

        /* all other commands require an active network */
        if (!s->anp) {
                eprintf("no active network");
                return;
        }

        if (cmd_create_group(cmd, "createGroup %s %d",
                                s->anp,
                                "created group: [%s:%d]"))
                return;
        if (cmd_dispose_group(cmd, "disposeGroup %s",
                                s->anp,
                                "disposed group: [%s]"))
                return;
        if (cmd_attach_bias(cmd, "attachBias %s",
                                s->anp,
                                "attached bias to group: [%s]"))
                return;

        if (cmd_set_input_group(cmd, "set InputGroup %s",
                                s->anp,
                                "set input group: [%s]"))
                return;
        if (cmd_set_output_group(cmd, "set OutputGroup %s",
                                s->anp,
                                "set output group: [%s]"))
                return;

        if (cmd_set_act_func(cmd, "set ActFunc %s %s",
                                s->anp,
                                "set activation function: [%s:%s]"))
                return;
        if (cmd_set_err_func(cmd, "set ErrFunc %s %s",
                                s->anp,
                                "set error function: [%s:%s]"))
                return;

        if (cmd_create_projection(cmd, "createProjection %s %s",
                                s->anp,
                                "created projection: [%s -> %s]"))
                return;
        if (cmd_create_elman_projection(cmd, "createElmanProjection %s %s",
                                s->anp,
                                "created elman projection: [%s -> %s]"))
                return;
        if (cmd_dispose_projection(cmd, "disposeProjection %s %s",
                                s->anp,
                                "disposed projection: [%s -> %s]"))
                return;
        if (cmd_freeze_projection(cmd, "freezeProjection %s %s",
                                s->anp,
                                "froze projection: [%s -> %s]"))
                return;
        
        /*
         * Int parameters.
         */
        
        if (cmd_set_int_parameter(cmd, "set BatchSize %d",
                                &s->anp->batch_size,
                                "set batch size: [%d]"))
                return;
        if (cmd_set_int_parameter(cmd, "set MaxEpochs %d",
                                &s->anp->max_epochs,
                                "set maximum number of epochs: [%d]"))
                return;
        if (cmd_set_int_parameter(cmd, "set ReportAfter %d",
                                &s->anp->report_after,
                                "set report training status after (number of epochs): [%d]"))
                return;
        if (cmd_set_int_parameter(cmd, "set RandomSeed %d",
                                &s->anp->random_seed,
                                "set random seed: [%d]"))
                return;
        if (cmd_set_int_parameter(cmd, "set HistoryLength %d",
                                &s->anp->history_length,
                                "set BPTT history length: [%d]"))
                return;
        
        /*
         * Double parameters.
         */
        
        if (cmd_set_double_parameter(cmd, "set RandomMu %lf",
                                &s->anp->random_mu,
                                "set random mu: [%lf]"))
                return;
        if (cmd_set_double_parameter(cmd, "set RandomSigma %lf",
                                &s->anp->random_sigma,
                                "set random sigma: [%lf]"))
                return;
        if (cmd_set_double_parameter(cmd, "set LearningRate %lf",
                                &s->anp->learning_rate,
                                "set learning rate: [%lf]"))
                return;
        if (cmd_set_double_parameter(cmd, "set LRScaleFactor %lf",
                                &s->anp->lr_scale_factor,
                                "set learning rate scale factor: [%lf]"))
                return;
        if (cmd_set_double_parameter(cmd, "set LRScaleAfter %lf",
                                &s->anp->lr_scale_after,
                                "set learning rate scale after (fraction of epochs): [%lf]"))
                return;
        if (cmd_set_double_parameter(cmd, "set Momentum %lf",
                                &s->anp->momentum,
                                "set momentum: [%lf]"))
                return;
        if (cmd_set_double_parameter(cmd, "set MNScaleFactor %lf",
                                &s->anp->mn_scale_factor,
                                "set momentum scale factor: [%lf]"))
                return;
        if (cmd_set_double_parameter(cmd, "set MNScaleAfter %lf",
                                &s->anp->mn_scale_after,
                                "set momentum scale after (fraction of epochs): [%lf]"))
                return;
        if (cmd_set_double_parameter(cmd, "set WeightDecay %lf",
                                &s->anp->weight_decay,
                                "set weight decay: [%lf]"))
                return;
        if (cmd_set_double_parameter(cmd, "set ErrorThreshold %lf",
                                &s->anp->error_threshold,
                                "set error threshold: [%lf]"))
                return;
        if (cmd_set_double_parameter(cmd, "set InitUpdate %lf",
                                &s->anp->rp_init_update,
                                "set initial update value (for Rprop): [%lf]"))
                return;
        if (cmd_set_double_parameter(cmd, "set EtaPlus %lf",
                                &s->anp->rp_eta_plus,
                                "set eta+ (for Rprop): [%lf]"))
                return;
        if (cmd_set_double_parameter(cmd, "set EtaMinus %lf",
                                &s->anp->rp_eta_minus,
                                "set eta- (for Rprop): [%lf]"))
                return;
        if (cmd_set_double_parameter(cmd, "set RateIncrement %lf",
                                &s->anp->dbd_rate_increment,
                                "set increment rate (for Delta-Bar-Delta): [%lf]"))
                return;
        if (cmd_set_double_parameter(cmd, "set RateDecrement %lf",
                                &s->anp->dbd_rate_decrement,
                                "set decrement rate (for Delta-Bar-Delta): [%lf]"))
                return;

        if (cmd_load_item_set(cmd, "loadTrainingSet %s",
                                s->anp,
                                true,
                                "loaded training set: [%s (%d elements)]"))
                return;
        if (cmd_load_item_set(cmd, "loadTestSet %s",
                                s->anp,
                                false,
                                "loaded test set: [%s (%d elements)]"))
                return;
        
        if (cmd_set_training_order(cmd, "set TrainingOrder %s",
                                &s->anp->training_order,
                                "set training order: [%s]"))
                return;

        if (cmd_set_learning_algorithm(cmd, "set LearningAlgorithm %s",
                                s->anp,
                                "set learning algorithm: [%s]"))
                return;
        if (cmd_set_update_algorithm(cmd, "set UpdateAlgorithm %s",
                                s->anp,
                                "set update algorithm: [%s]"))
                return;

        if (cmd_init(cmd, "init",
                                s->anp,
                                "initializing network: [%s]"))
                return;

        /* all other commands require an initialized active network */
        if (!s->anp->initialized) {
                eprintf("unitizialized network--use 'init' command to initialize");
                return;
        }

        if (cmd_reset(cmd, "reset",
                                s->anp,
                                "resetting network: [%s]"))
                return;
        if (cmd_train(cmd, "train", 
                                s->anp,
                                "starting training of network: [%s]"))
                return;
        if (cmd_test(cmd, "test",
                                s->anp,
                                "starting testing of network: [%s]"))
                return;
        // XXX: check this ...
        if (cmd_test_item(cmd, "testItem %[^]",
                                s->anp,
                                "starting testing of network [%s] for item: [%s]"))
                return;

        if (cmd_compare_vectors(cmd, "compareVectors %s %s %s",
                                s->anp,
                                "comparing vectors of group [%s] for items [%s] and [%s]:"))
                return;

        if (cmd_weight_stats(cmd, "weightStats",
                                s->anp,
                                "weight statistics for network: [%s]"))
                return;
        
        if (cmd_show_vector(cmd, "showUnits %s",
                                s->anp,
                                "unit vector for: [%s]",
                                VTYPE_UNITS))
                return;
        if (cmd_show_vector(cmd, "showError %s",
                                s->anp,
                                "error vector for: [%s]",
                                VTYPE_ERROR))
                return;

        if (cmd_show_matrix(cmd, "showWeights %s %s",
                                s->anp,
                                "weight matrix for projection: [%s -> %s]",
                                MTYPE_WEIGHTS))
                return;
        if (cmd_show_matrix(cmd, "showGradients %s %s",
                                s->anp,
                                "gradient matrix for projection: [%s -> %s]",
                                MTYPE_GRADIENTS))
                return;
        if (cmd_show_matrix(cmd, "showDynPars %s %s",
                                s->anp,
                                "dynamic learning parameters matrix for projection: [%s -> %s]",
                                MTYPE_DYN_PARS))
                return;

        /*
         * Weight matrix saving and loading.
         */

        if (cmd_save_weights(cmd, "saveWeights %s",
                                s->anp,
                                "saved weights: [%s]"))
                return;
        if (cmd_load_weights(cmd, "loadWeights %s",
                                s->anp,
                                "loaded weights: [%s]"))
                return;

        /*
         * Invalid command.
         */

        if (strlen(cmd) > 1) {
                eprintf("invalid command: %s", cmd); 
                eprintf("(type 'help' for a list of valid commands)");
        }

        return;
}

/*
 * ########################################################################
 * ## Commands                                                           ##
 * ########################################################################
 */

/*
 * Quit program.
 *
 * Syntax: quit (or) exit
 */

void cmd_quit(char *cmd, char *fmt, struct session *s, char *msg)
{
        if (strcmp(cmd, fmt) != 0)
                return;

        mprintf(msg);
        dispose_session(s);
        exit(EXIT_SUCCESS);
}

/*
 * Create network.
 *
 * Syntax: createNetwork <type> <name>
 */

bool cmd_create_network(char *cmd, char *fmt, struct session *s, char *msg)
{
        char tmp1[64], tmp2[64];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        /* network type */
        int type = 0;
        if (strcmp(tmp2, "ffn") == 0)
                type = TYPE_FFN;
        else if (strcmp(tmp2, "srn") == 0)
                type = TYPE_SRN;
        else if (strcmp(tmp2, "rnn") == 0)
                type = TYPE_RNN;
        else {
                eprintf("cannot create network--invalid network type: [%s]", tmp2);
                return true;
        }

        /* check network name */
        if (find_network_by_name(s, tmp1)) {
                eprintf("cannot create network--network [%s] already exists", tmp1);
                return true;
        }
        
        /* create network */
        struct network *n = create_network(tmp1, type);

        /* add to session */
        s->networks->elements[s->networks->num_elements++] = n;
        if (s->networks->num_elements == s->networks->max_elements)
                increase_network_array_size(s->networks);
        s->anp = n;

        mprintf(msg, tmp1, tmp2);

        return true;
}

bool cmd_load_network(char *cmd, char *fmt, struct session *s, char *msg)
{
        char tmp[64];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        mprintf("attempting to load network: [%s]", tmp);

        FILE *fd;
        if (!(fd = fopen(tmp, "r"))) {
                eprintf("cannot open file: [%s]", tmp);
                return true;
        }

        char buf[1024];
        while (fgets(buf, sizeof(buf), fd)) {
                buf[strlen(buf) - 1] = '\0';
                process_command(buf, s);
        }

        fclose(fd);

        mprintf(msg, tmp);

        return true;
}

bool cmd_dispose_network(char *cmd, char *fmt, struct session *s, char *msg)
{
        char tmp[64];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        struct network *n = find_network_by_name(s, tmp);
        if (!n) {
                eprintf("cannot dispose network: [%s]--no such network", tmp);
                return true;
        }

        int i;
        for (i = 0; i < s->networks->num_elements; i++)
                if (s->networks->elements[i] == n)
                        break;

        for (int j = i; j < s->networks->num_elements - 1; j++)
                s->networks->elements[j] = s->networks->elements[j + 1];
        s->networks->elements[s->networks->num_elements - 1] = NULL;

        if (n == s->anp)
                s->anp = s->networks->elements[0];
        dispose_network(n);
        s->networks->num_elements--;

        mprintf(msg, tmp);

        return true;
}

bool cmd_list_networks(char *cmd, char *fmt, struct session *s, char *msg)
{
        if (strcmp(cmd, fmt) != 0)
                return false;

        mprintf(msg);

        for (int i = 0; i < s->networks->num_elements; i++)
                printf("%s\n", s->networks->elements[i]->name);

        return true;
}

bool cmd_change_network(char *cmd, char *fmt, struct session *s, char *msg)
{
        char tmp[64];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        struct network *n = find_network_by_name(s, tmp);
        if (!n) {
                eprintf("cannot change to network: [%s]--no such network", tmp);
                return true;
        }

        s->anp = n;

        mprintf(msg, tmp);

        return true;
}

bool cmd_create_group(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp[64];
        int tmp_int;
        if (sscanf(cmd, fmt, tmp, &tmp_int) != 2)
                return false;

        struct group *g = create_group(tmp, tmp_int, false, false);
        add_to_group_array(n->groups, g);

        mprintf(msg, tmp, tmp_int);

        return true;
}

bool cmd_dispose_group(char *cmd, char *fmt, struct network *n, char *msg)
{
        return false;
}

bool cmd_attach_bias(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp[64];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        struct group *g = find_group_by_name(n, tmp);
        if (g == NULL) {
                eprintf("cannot attach bias--group [%s] unknown", tmp);
                return true;
        }

        attach_bias_group(n, g);

        mprintf(msg, tmp);

        return true;
}

bool cmd_set_input_group(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp[64];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        struct group *g = find_group_by_name(n, tmp);
        if (g == NULL) {
                eprintf("cannot set input group--group [%s] unknown", tmp);
                return true;
        }

        n->input = g;

        mprintf(msg, tmp);

        return true;
}

bool cmd_set_output_group(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp[64];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        struct group *g = find_group_by_name(n, tmp);
        if (g == NULL) {
                eprintf("cannot set output group--group [%s] unknown", tmp);
                return true;
        }

        n->output = g;

        mprintf(msg, tmp);

        return true;
}

bool cmd_set_act_func(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp1[64], tmp2[64];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        struct group *g = find_group_by_name(n, tmp1);
        if (g == NULL) {
                eprintf("cannot set activation function--group [%s] unknown",
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
                eprintf("no such activation function: [%s]", tmp2);
                return true;
        }

        mprintf(msg, tmp1, tmp2);

        return true;
}

bool cmd_set_err_func(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp1[64], tmp2[64];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        struct group *g = find_group_by_name(n, tmp1);
        if (g == NULL) {
                eprintf("cannot set error function--group [%s] unknown",
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
                eprintf("no such error function: [%s]", tmp2);
                return true;
        }

        mprintf(msg, tmp1, tmp2);

        return true;
}

bool cmd_create_projection(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp1[64], tmp2[64];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        struct group *fg = find_group_by_name(n, tmp1);
        struct group *tg = find_group_by_name(n, tmp2);

        if (fg == NULL) {
                eprintf("cannot set projection--'from' group [%s] unknown",
                                tmp1);
                return true;
        }
        if (tg == NULL) {
                eprintf("cannot set projection--'to' group [%s] unknown",
                                tmp2);
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
                struct matrix *prev_weight_deltas = create_matrix(
                                fg->vector->size,
                                tg->vector->size);
                struct matrix *dyn_learning_pars = create_matrix(
                                fg->vector->size,
                                tg->vector->size);

                struct projection *op;
                op = create_projection(tg, weights, gradients, prev_gradients,
                                prev_weight_deltas, dyn_learning_pars, false);
                add_to_projs_array(fg->out_projs, op);

                struct projection *ip;
                ip = create_projection(fg, weights, gradients, prev_gradients,
                                prev_weight_deltas, dyn_learning_pars, false);
                add_to_projs_array(tg->inc_projs, ip);

        }

        mprintf(msg, tmp1, tmp2);

        return true;
}

bool cmd_create_elman_projection(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp1[64], tmp2[64];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        struct group *fg = find_group_by_name(n, tmp1);
        struct group *tg = find_group_by_name(n, tmp2);

        if (fg == NULL) {
                eprintf("cannot set Elman-projection--'from' group [%s] unknown",
                                tmp1);
                return true;
        }
        if (tg == NULL) {
                eprintf("cannot set Elman-projection--'from' group [%s] unknown",
                                tmp2);
                return true;
        }

        if (fg == tg) {
                eprintf("cannot set Elman-projection--'from' and 'to' are the same [%s]",
                                fg->name);
                return true;
        }

        if (fg->vector->size != tg->vector->size) {
                eprintf("cannot set Elman-projection--'from' and 'to' "
                                "group have unequal vector sizes (%d and %d)",
                                fg->vector->size, tg->vector->size);
                return true;
        }

        fg->context_group = tg;

        reset_context_groups(n);

        mprintf(msg, tmp1, tmp2);

        return true;
}

bool cmd_dispose_projection(char *cmd, char *fmt, struct network *n, char *msg)
{
        return false;
}

bool cmd_freeze_projection(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp1[64], tmp2[64];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        struct group *fg = find_group_by_name(n, tmp1);
        struct group *tg = find_group_by_name(n, tmp2);

        if (fg == NULL) {
                eprintf("cannot freeze projection--'from' group [%s] unknown",
                                tmp1);
                return true;
        }
        if (tg == NULL) {
                eprintf("cannot freeze projection--'to' group [%s] unknown",
                                tmp2);
                return true;
        }

        struct projection *fg_to_tg = NULL;
        for (int i = 0; i < fg->out_projs->num_elements; i++)
                if (fg->out_projs->elements[i]->to == tg)
                        fg_to_tg = fg->out_projs->elements[i];

        struct projection *tg_to_fg = NULL;
        for (int i = 0; i < tg->inc_projs->num_elements; i++)
                if (tg->inc_projs->elements[i]->to == fg)
                        tg_to_fg = tg->inc_projs->elements[i];

        if (fg_to_tg && tg_to_fg) {
                fg_to_tg->frozen = true;
                tg_to_fg->frozen = true;
        } else {
                eprintf("cannot freeze projection--no projection between groups (%s and %s)",
                                tmp1, tmp2);
                return true;
        }

        mprintf(msg, tmp1, tmp2);

        return true;
}

bool cmd_set_double_parameter(char *cmd, char *fmt, double *par, char *msg)
{
        if (sscanf(cmd, fmt, par) != 1)
                return false;

        mprintf(msg, *par);

        return true;
}

bool cmd_set_int_parameter(char *cmd, char *fmt, int *par, char *msg)
{
        if (sscanf(cmd, fmt, par) != 1)
                return false;

        mprintf(msg, *par);

        return true;
}

bool cmd_load_item_set(char *cmd, char *fmt, struct network *n, bool train,
                char *msg)
{
        char tmp[64];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        if (!n->input) {
                eprintf("cannot load set--'input' group size unknown");
                return true;
        }
        if (!n->output) {
                eprintf("cannot load set--'output' group size unknown");
                return true;
        }

        struct set *s = load_set(tmp, n->input->vector->size, n->output->vector->size);
        
        if (train)
                n->training_set = s;
        else
                n->test_set = s;

        if (s)
                mprintf(msg, tmp, s->num_elements);

        return true;
}

bool cmd_set_training_order(char *cmd, char *fmt, int *training_order,
                char *msg)
{
        char tmp[64];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        if (strcmp(tmp, "ordered") == 0)
                *training_order = TRAIN_ORDERED;
        else if (strcmp(tmp, "permuted") == 0)
                *training_order = TRAIN_PERMUTED;
        else if (strcmp(tmp, "randomized") == 0)
                *training_order = TRAIN_RANDOMIZED;
        else {
                eprintf("invalid training order: %s", tmp);
                return true;
        }

        mprintf(msg, tmp);

        return true;
}

bool cmd_set_learning_algorithm(char *cmd, char *fmt, struct network *n,
                char *msg)
{
        char tmp[64];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        if (strlen(tmp) == 2 && strcmp(tmp, "bp") == 0)
                n->learning_algorithm = train_network_with_bp;
        else if (strlen(tmp) == 4 && strcmp(tmp, "bptt") == 0)
                n->learning_algorithm = train_network_with_bptt;
        else {
                eprintf("invalid learning algorithm: %s", tmp);
                return true;
        }
                        
        if (n->learning_algorithm)
                mprintf(msg, tmp);

        return true;
}

bool cmd_set_update_algorithm(char *cmd, char *fmt, struct network *n,
                char *msg)
{
        char tmp[64];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        if (strcmp(tmp, "steepest") == 0) {
                n->update_algorithm = bp_update_sd;
                n->sd_type = SD_DFLT;
        }
        else if (strcmp(tmp, "dougs") == 0) {
                n->update_algorithm = bp_update_sd;
                n->sd_type = SD_DOUG;
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
                eprintf("invalid upadate algorithm: %s", tmp);
                return true;
        }
        
        if (n->update_algorithm)
                mprintf(msg, tmp);

        return true;
}

bool cmd_init(char *cmd, char *fmt, struct network *n, char *msg)
{
        if (strlen(cmd) != strlen(fmt) 
                        || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf(msg, n->name);

        init_network(n);

        return true;
}

bool cmd_reset(char *cmd, char *fmt, struct network *n, char *msg)
{
        if (strlen(cmd) != strlen(fmt) 
                        || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf(msg, n->name);

        reset_network(n);

        return true;
}

bool cmd_train(char *cmd, char *fmt, struct network *n, char *msg)
{
        if (strlen(cmd) != strlen(fmt) 
                        || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf(msg, n->name);

        train_network(n);

        return true;
}

bool cmd_test(char *cmd, char *fmt, struct network *n, char *msg)
{
        if (strlen(cmd) != strlen(fmt) 
                        || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf(msg, n->name);
                
        test_network(n);

        return true;
}

// TODO: extend such that set (train/test) can be specificied
bool cmd_test_item(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp[64];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        mprintf(msg, n->name, tmp);

        struct element *e = find_element_by_name(n->test_set, tmp);
        if (!e) {
                eprintf("cannot test network--element [%s] unknown", tmp);
                return true;
        }

        test_network_with_item(n, e);

        return true;
}

// TODO: extend such that set (train/test) can be specificied
bool cmd_compare_vectors(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp1[64], tmp2[64], tmp3[64];
        if (sscanf(cmd, fmt, tmp1, tmp2, tmp3) != 3)
                return false;

        mprintf(msg, tmp1, tmp2, tmp3);

        struct group *g = find_group_by_name(n, tmp1);
        if (!g) {
                eprintf("cannot compare vectors--group [%s] unknown", tmp1);
                return true;
        }

        struct element *e1 = find_element_by_name(n->test_set, tmp2);
        if (!e1) {
                eprintf("cannot compare vectors--item [%s] unknown", tmp2);
                return true;
        }

        struct element *e2 = find_element_by_name(n->test_set, tmp3);
        if (!e2) {
                eprintf("cannot compare vectors--item [%s] unknown", tmp3);
                return true;
        }

        struct vector *v1 = create_vector(g->vector->size);
        struct vector *v2 = create_vector(g->vector->size);

        test_network_with_item(n, e1);
        copy_vector(v1, g->vector);
        test_network_with_item(n, e2);
        copy_vector(v2, g->vector);

        cprintf("");
        mprintf("vectors in group [%s] for 1: [%s] and 2: [%s]",
                        tmp1, tmp2, tmp3);
        printf("1: ");
        pprint_vector(v1);
        printf("2: ");
        pprint_vector(v2);

        cprintf("");
        cprintf("inner product:\t\t[%f]",
                        inner_product(v1, v2));
        cprintf("cosine similarity:\t[%f]",
                        cosine_similarity(v1, v2));
        cprintf("Pearson's correlation:\t[%f]",
                        pearson_correlation(v1, v2));
        cprintf("");

        dispose_vector(v1);
        dispose_vector(v2);

        return true;
}

bool cmd_weight_stats(char *cmd, char *fmt, struct network *n, char *msg)
{
        if (strlen(cmd) != strlen(fmt) 
                        || strncmp(cmd, fmt, strlen(cmd)) != 0)
                return false;

        mprintf(msg, n->name);

        struct weight_stats *ws = create_weight_statistics(n);
        
        cprintf("");
        cprintf("mean:\t\t[%f]", ws->mean);
        cprintf("mean abs.:\t[%f]", ws->mean_abs);
        cprintf("mean dist.:\t[%f]", ws->mean_dist);
        cprintf("variance:\t[%f]", ws->variance);
        cprintf("minimum:\t[%f]", ws->minimum);
        cprintf("maximum:\t[%f]", ws->maximum);
        cprintf("");
        
        dispose_weight_statistics(ws);

        return true;
}

bool cmd_show_vector(char *cmd, char *fmt, struct network *n, char *msg, int type)
{
        char tmp[64];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        mprintf(msg, tmp);

        struct group *g = find_group_by_name(n, tmp);

        if (g == NULL) {
                eprintf("cannot show vector---group [%s] unknown",
                                tmp);
                return true;
        }

        if (type == VTYPE_UNITS)
                pprint_vector(g->vector);
        if (type == VTYPE_ERROR)
                pprint_vector(g->error);

        return true;
}

bool cmd_show_matrix(char *cmd, char *fmt, struct network *n, char *msg, int type)
{
        char tmp1[64], tmp2[64];
        if (sscanf(cmd, fmt, tmp1, tmp2) != 2)
                return false;

        mprintf(msg, tmp1, tmp2);

        struct group *fg = find_group_by_name(n, tmp1);
        struct group *tg = find_group_by_name(n, tmp2);

        if (fg == NULL) {
                eprintf("cannot show matrix---'from' group [%s] unknown",
                                tmp1);
                return true;
        }
        if (tg == NULL) {
                eprintf("cannot show matrix---'to' group [%s] unknown",
                                tmp2);
                return true;
        }

        struct projection *fg_to_tg = NULL;
        for (int i = 0; i < fg->out_projs->num_elements; i++)
                if (fg->out_projs->elements[i]->to == tg)
                        fg_to_tg = fg->out_projs->elements[i];

        if (fg_to_tg) {
                if (type == MTYPE_WEIGHTS)
                        pprint_matrix(fg_to_tg->weights);
                if (type == MTYPE_GRADIENTS)
                        pprint_matrix(fg_to_tg->gradients);
                if (type == MTYPE_DYN_PARS)
                        pprint_matrix(fg_to_tg->dyn_learning_pars);
        } else {
                eprintf("cannot show matrix--no projection between groups (%s and %s)",
                                tmp1, tmp2);
                return true;
        }

        return true;
}

bool cmd_save_weights(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp[64];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        mprintf("attempting to save weights: [%s]", tmp);

        save_weight_matrices(n, tmp);

        mprintf(msg, tmp);

        return true;
}

bool cmd_load_weights(char *cmd, char *fmt, struct network *n, char *msg)
{
        char tmp[64];
        if (sscanf(cmd, fmt, tmp) != 1)
                return false;

        mprintf("attempting to load weights: [%s]", tmp);

        load_weight_matrices(n, tmp);

        mprintf(msg, tmp);

        return true;
}
