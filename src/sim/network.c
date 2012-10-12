/*
 * network.c
 *
 * Copyright 2012 Harm Brouwer <me@hbrouwer.eu>
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

#include <math.h>

#include "act.h"
#include "bp.h"
#include "engine.h"
#include "error.h"
#include "network.h"
#include "pprint.h"
#include "stats.h"

/*
 * ########################################################################
 * ## Network construction                                               ##
 * ########################################################################
 */

/*
 * Creates a new network.
 */

struct network *create_network(char *name, int type)
{
        struct network *n;
        if (!(n = malloc(sizeof(struct network))))
                goto error_out;
        memset(n, 0, sizeof(struct network));

        int block_size = (strlen(name) + 1) * sizeof(char);
        if (!(n->name = malloc(block_size)))
                goto error_out;
        memset(n->name, 0, block_size);
        strncpy(n->name, name, strlen(name));

        n->type = type;

        n->groups = create_group_array(MAX_GROUPS);

        block_size = sizeof(struct status);
        if (!(n->status = malloc(block_size)))
                goto error_out;
        memset(n->status, 0, block_size);

        return n;

error_out:
        perror("[create_network()]");
        return NULL;
}

/*
 * Iniatilizes a network.
 */

void initialize_network(struct network *n)
{
        mprintf("attempting to initialize network: [%s]", n->name);

        srand(n->random_seed);

        randomize_weight_matrices(n->input, n);

        /* for Rprop and DBD */
        n->rp_init_update = 0.1;
        initialize_dyn_learning_pars(n->input, n);

        if (n->batch_size == 0)
                n->batch_size = n->training_set->num_elements;

        /* initialize unfolded network */
        if (n->learning_algorithm == train_network_bptt)
                n->unfolded_net = ffn_init_unfolded_network(n);

        if (!n->unfolded_net && n->load_weights_file)
                load_weights(n);
        if (n->unfolded_net && n->load_weights_file)
                load_weights(n->unfolded_net->stack[0]);

        // XXX: todo--sanity checks!

        mprintf("initialized network: [%s]", n->name);
}

/*
 * Disposes a network.
 */

void dispose_network(struct network *n)
{
        free(n->name);
        free(n->status);

        if (n->unfolded_net)
                ffn_dispose_unfolded_network(n->unfolded_net);

        dispose_groups(n->groups);
        dispose_group_array(n->groups);

        dispose_set(n->training_set);
        dispose_set(n->test_set);

        free(n);
}

/*
 * Creates a new group array.
 */

struct group_array *create_group_array(int max_elements)
{
        struct group_array *gs;
        if (!(gs = malloc(sizeof(struct group_array))))
                goto error_out;
        memset(gs, 0, sizeof(struct group_array));

        gs->num_elements = 0;
        gs->max_elements= max_elements;

        int block_size = gs->max_elements * sizeof(struct group *);
        if (!(gs->elements = malloc(block_size)))
                goto error_out;
        memset(gs->elements, 0, block_size);

        return gs;

error_out:
        perror("[create_group_array()]");
        return NULL;
}

/*
 * Adds a group to a group array.
 */

void add_to_group_array(struct group_array *gs, struct group *g)
{
        gs->elements[gs->num_elements++] = g;
        if (gs->num_elements == gs->max_elements)
                increase_group_array_size(gs);
}

/*
 * Increases the size of a group array.
 */

void increase_group_array_size(struct group_array *gs)
{
        gs->max_elements = gs->max_elements + MAX_GROUPS;
        
        /* reallocate memory for the group array */
        int block_size = gs->max_elements * sizeof(struct group *);
        if (!(gs->elements = realloc(gs->elements, block_size)))
                goto error_out;

        /* make sure the new array cells are empty */
        for (int i = gs->num_elements; i < gs->max_elements; i++)
                gs->elements[i] = NULL;

        return;

error_out:
        perror("[increase_group_array_size()]");
        return;
}

/*
 * Disposes a group array.
 */

void dispose_group_array(struct group_array *gs)
{
        free(gs->elements);
        free(gs);
}

/*
 * Creates a new group.
 */

struct group *create_group(
                char *name,
                struct act_fun *act_fun,
                struct err_fun *err_fun,
                int size,
                bool bias, 
                bool recurrent)
{
        struct group *g;
        if (!(g = malloc(sizeof(struct group))))
                goto error_out;
        memset(g, 0, sizeof(struct group));

        int block_size = (strlen(name) + 1) * sizeof(char);
        if (!(g->name = malloc(block_size)))
                goto error_out;
        memset(g->name, 0, block_size);
        strncpy(g->name, name, strlen(name));

        g->vector = create_vector(size);
        g->error = create_vector(size);

        g->act_fun = act_fun;
        g->err_fun = err_fun;

        g->inc_projs = create_projs_array(MAX_PROJS);
        g->out_projs = create_projs_array(MAX_PROJS);

        g->bias = bias;
        g->recurrent = recurrent;

        if(g->bias)
                g->vector->elements[0] = 1.0;

        return g;

error_out:
        perror("[create_group()]");
        return NULL;
}

/*
 * Attaches a bias group to a specified group.
 */

void attach_bias_group(struct network *n, struct group *g)
{
        /*
         * Create a new "bias" group.
         */
        char *tmp;
        int block_size = (strlen(g->name) + 6) * sizeof(char);
        if (!(tmp = malloc(block_size)))
                goto error_out;
        memset(tmp, 0, sizeof(block_size));

        sprintf(tmp, "%s_bias", g->name);
        struct group *bg = create_group(tmp, g->act_fun, g->err_fun, 1, true, false);

        free(tmp);

        /* 
         * Add it to the network's group array.
         */
        add_to_group_array(n->groups, bg);

        /* weight matrix */
        struct matrix *weights = create_matrix(
                        bg->vector->size,
                        g->vector->size);

        /* error vector */
        struct vector *error = create_vector(
                        bg->vector->size);

        /* gradients matrix */
        struct matrix *gradients = create_matrix(
                        bg->vector->size,
                        g->vector->size);

        /* previous gradients matrix */
        struct matrix *prev_gradients = create_matrix(
                        bg->vector->size,
                        g->vector->size);

        /* previous weight deltas matrix */
        struct matrix *prev_weight_deltas = create_matrix(
                        bg->vector->size,
                        g->vector->size);

        /* dynamic learning parameters matrix */
        struct matrix *dyn_learning_pars = create_matrix(
                        bg->vector->size,
                        g->vector->size);

        /* 
         * Create a projection from the bias group to the bias receiving 
         * group.
         */
        struct projection *op;
        op = create_projection(g, weights, error, gradients, prev_gradients,
                                prev_weight_deltas, dyn_learning_pars, false);
        add_to_projs_array(bg->out_projs, op);
        
        /*
         * Create a projection from the bias receiving group to the bias
         * group.
         */
        struct projection *ip;
        ip = create_projection(bg, weights, error, gradients, prev_gradients,
                                prev_weight_deltas, dyn_learning_pars, false);
        add_to_projs_array(g->inc_projs, ip);

        return;

error_out:
        perror("[attach_bias_group()]");
        return;
}

/*
 * Dispose a group.
 */

void dispose_group(struct group *g)
{
        free(g->name);
        dispose_vector(g->vector);
        dispose_vector(g->error);
        if (!g->bias) {
                free(g->act_fun);
                free(g->err_fun);
        }

        for (int j = 0; j < g->inc_projs->num_elements; j++)
                dispose_projection(g->inc_projs->elements[j]);
        for (int j = 0; j < g->out_projs->num_elements; j++)
                free(g->out_projs->elements[j]);

        dispose_projs_array(g->inc_projs);
        dispose_projs_array(g->out_projs);

        free(g);
}


/*
 * Dispose all the groups in the specfied groups array.
 */

void dispose_groups(struct group_array *groups)
{
        for (int i = 0; i < groups->num_elements; i++)
                dispose_group(groups->elements[i]);
}

/*
 * Shifts a context group chain. If group g has a context group c, then
 * the activity vector of g is copied into that of c. However, if c has
 * itself a context group c', then the activity pattern of c is first
 * copied into c', and so forth.
 */

void shift_context_group_chain(struct network *n, struct group *g,
                struct vector *v)
{
        if (g->context_group)
                shift_context_group_chain(n, g->context_group, g->vector);
        
        copy_vector(g->vector, v);
}

/*
 * This resets all the context groups in a network to their initial value.
 */

void reset_context_groups(struct network *n)
{
        for (int i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];
                if (g->context_group)
                        fill_vector_with_value(g->context_group->vector, 0.5);
        }
}

/*
 * This resets all the recurrent groups in a network to their intitial
 * value.
 */

void reset_recurrent_groups(struct network *n) {
        for (int i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];
                if (g->recurrent)
                        fill_vector_with_value(g->vector, 0.0);
        }
}
        
/*
 * Creates a new projection array.
 */

struct projs_array *create_projs_array(int max_elements)
{
        struct projs_array *ps;
        if (!(ps = malloc(sizeof(struct projs_array))))
                goto error_out;
        memset(ps, 0, sizeof(struct projs_array));

        ps->num_elements = 0;
        ps->max_elements = max_elements;

        int block_size = ps->max_elements * sizeof(struct projection *);
        if (!(ps->elements = malloc(block_size)))
                goto error_out;
        memset(ps->elements, 0, block_size);

        return ps;

error_out:
        perror("[create_projs_array()]");
        return NULL;

}

/*
 * Adds a projection to a projection array.
 */

void add_to_projs_array(struct projs_array *ps, struct projection *p)
{
        ps->elements[ps->num_elements++] = p;
        if (ps->num_elements == ps->max_elements)
                increase_projs_array_size(ps);
}

/*
 * Increases the size of a projection array.
 */

void increase_projs_array_size(struct projs_array *ps)
{
        ps->max_elements = ps->max_elements + MAX_PROJS;

        /* reallocate memory for the projection array */
        int block_size = ps->max_elements * sizeof(struct projection *);
        if (!(ps->elements = realloc(ps->elements, block_size)))
                goto error_out;

        /* make sure the new array cells are empty */
        for (int i = ps->num_elements; i < ps->max_elements; i++)
                ps->elements[i] = NULL;

        return;

error_out:
        perror("[increase_projs_array_size()]");
        return;
}

/*
 * Disposes a projection array.
 */

void dispose_projs_array(struct projs_array *ps)
{
        free(ps->elements);
        free(ps);
}

/*
 * Creates a new projection.
 */

struct projection *create_projection(
                struct group *to,
                struct matrix *weights,
                struct vector *error,
                struct matrix *gradients,
                struct matrix *prev_gradients,
                struct matrix *prev_weight_deltas,
                struct matrix *dyn_learning_pars,
                bool recurrent)
{
        struct projection *p;

        if (!(p = malloc(sizeof(struct projection))))
                goto error_out;
        memset(p, 0, sizeof(struct projection));

        p->to = to;
        p->weights = weights;
        p->error = error;
        p->gradients = gradients;
        p->prev_gradients = prev_gradients;
        p->prev_weight_deltas = prev_weight_deltas;
        p->dyn_learning_pars = dyn_learning_pars;
        p->recurrent = recurrent;

        return p;

error_out:
        perror("[create_projection()]");
        return NULL;
}
/*
 * Disposes a projection.
 */


void dispose_projection(struct projection *p)
{
        dispose_matrix(p->weights);
        dispose_vector(p->error);
        dispose_matrix(p->gradients);
        dispose_matrix(p->prev_gradients);
        dispose_matrix(p->prev_weight_deltas);
        dispose_matrix(p->dyn_learning_pars);

        free(p);
}

/*
 * This randomizes the weights for all incoming projections of a group g,
 * and then recursively does the same for all groups towards which
 * g projects.
 */

void randomize_weight_matrices(struct group *g, struct network *n)
{
        for (int i = 0; i < g->inc_projs->num_elements; i++)
                randomize_matrix(g->inc_projs->elements[i]->weights, 
                                n->random_mu, n->random_sigma);

        for (int i = 0; i < g->out_projs->num_elements; i++)
                randomize_weight_matrices(g->out_projs->elements[i]->to, n);
}

/*
 * Initializes dynamic learning parameters for Rprop and Delta-Bar-Delta.
 */

void initialize_dyn_learning_pars(struct group *g, struct network *n)
{
        double v = 0.0;
        if (n->update_algorithm == bp_update_dbd) {
                v = n->learning_rate;
        } else {
                v = n->rp_init_update;
        }

        for (int i = 0; i < g->inc_projs->num_elements; i++)
                fill_matrix_with_value(g->inc_projs->elements[i]->dyn_learning_pars, v);

        for (int i = 0; i < g->out_projs->num_elements; i++)
                initialize_dyn_learning_pars(g->out_projs->elements[i]->to, n);
}

/*
 * ########################################################################
 * ## Network loading                                                    ##
 * ########################################################################
 */
/*
 * Loads a network from a file
 */

struct network *load_network(char *filename)
{
        mprintf("attempting to load network: [%s]", filename);

        struct network *n = NULL;

        FILE *fd;
        if (!(fd = fopen(filename, "r")))
                goto error_out;

        char buf[1024];
        while (fgets(buf, sizeof(buf), fd)) {
                char tmp1[64], tmp2[64], input[64], output[64];

                if (sscanf(buf, "Network %s %s %s %s", tmp1, tmp2, input, output)) {
                        int type = 0;
                        if (strcmp(tmp1, "ffn") == 0)
                                type = TYPE_FFN;
                        if (strcmp(tmp1, "srn") == 0)
                                type = TYPE_SRN;
                        n = create_network(tmp2,type);
                        mprintf("created network: [%s:%s:(%s -> %s)]",
                                        tmp1, tmp2, input, output);
                }

                load_int_parameter(buf, "RandomSeed %d", &n->random_seed,
                                "set random seed: [%d]");
                load_double_parameter(buf, "RandomMu %lf", &n->random_mu,
                                "set random mu: [%lf]");                
                load_double_parameter(buf, "RandomSigma %lf", &n->random_sigma,
                                "set random sigma: [%lf]");

                load_double_parameter(buf, "LearningRate %lf", &n->learning_rate,
                                "set learning rate: [%lf]");
                load_double_parameter(buf, "LRScaleFactor %lf", &n->lr_scale_factor,
                                "set LR scale factor: [%lf]");
                load_double_parameter(buf, "LRScaleAfter %lf", &n->lr_scale_after,
                                "set LR scaling after (fraction of epochs): [%lf]");
                load_double_parameter(buf, "Momentum %lf", &n->momentum,
                                "set momentum: [%lf]");
                load_double_parameter(buf, "MNScaleFactor %lf", &n->mn_scale_factor,
                                "set momentum scale factor: [%lf]");
                load_double_parameter(buf, "MNScaleAfter %lf", &n->mn_scale_after,
                                "set momentum scaling after (fraction of epochs): [%lf]");
                load_double_parameter(buf, "WeightDecay %lf", &n->weight_decay,
                                "set weight decay: [%lf]");
                load_double_parameter(buf, "ErrorThreshold %lf", &n->error_threshold,
                                "set error threshold: [%lf]");

                load_int_parameter(buf, "BatchSize %d", &n->batch_size,
                                "set batch size: [%d]");
                load_int_parameter(buf, "MaxEpochs %d", &n->max_epochs,
                                "set maximum number of epochs: [%d]");
                load_int_parameter(buf, "ReportAfter %d", &n->report_after,
                                "report training status after (number of epochs): [%d]");

                load_int_parameter(buf, "HistoryLength %d", &n->history_length,
                                "set BPTT history length: [%d]");

                load_learning_algorithm(buf, "LearningMethod %s", n,
                                "set learning algorithm: [%s]");
                load_update_algorithm(buf, "UpdateMethod %s", n,
                                "set update algorithm: [%s]");

                load_group(buf, "Group %s %s %s %d", n, input, output,
                                "added group: [%s (%s:%s:%d)]");

                load_bias(buf, "AttachBias %s", n,
                                "attached bias to group: [%s]");

                load_projection(buf, "Projection %s %s", n,
                                "added projection: [%s -> %s]");
                load_freeze_projection(buf, "FreezeProjection %s %s", n,
                                "froze projection: [%s -> %s]");

                load_recurrent_group(buf, "RecurrentGroup %s", n,
                                "added recurrent projection: [%s <=> %s]");

                load_elman_projection(buf, "ElmanProjection %s %s", n,
                                "added Elman-projection: [%s <=> %s]");

                load_item_set(buf, "TrainingSet %s", n, true,
                                "loaded training set: [%s]");
                load_item_set(buf, "TestSet %s", n, false,
                                "loaded test set: [%s]");

                load_training_order(buf, "TrainingOrder %s", n,
                                "set training order: [%s]");
        }

        fclose(fd);

        mprintf("loaded network: [%s]", filename);

        return n;

error_out:
        perror("[load_network()]");
        return NULL;
}

void load_double_parameter(char *buf, char *fmt, double *par, char *msg)
{
        if (sscanf(buf, fmt, par))
                mprintf(msg, *par);
}

void load_int_parameter(char *buf, char *fmt, int *par, char *msg)
{
        if (sscanf(buf, fmt, par))
                mprintf(msg, *par);
}

void load_learning_algorithm(char *buf, char *fmt, struct network *n,
                char *msg)
{
        char tmp[64];
        if (sscanf(buf, fmt, tmp) == 0)
                return;

        /* backpropagation */
        if (strncmp(tmp, "bp", 2) == 0)
                n->learning_algorithm = train_network_bp;

        /* backpropagation through time */
        if (strncmp(tmp, "bptt", 4) == 0)
                n->learning_algorithm = train_network_bptt;
                        
        if (n->learning_algorithm)
                mprintf(msg, tmp);
}

void load_update_algorithm(char *buf, char *fmt, struct network *n,
                char *msg)
{
        char tmp[64];
        if (sscanf(buf, fmt, tmp) == 0)
                return;

        /* steepest descent */
        if (strcmp(tmp, "steepest") == 0)
                n->update_algorithm = bp_update_sd;

        if (strcmp(tmp, "rprop+") == 0) {
                n->update_algorithm = bp_update_rprop;
                n->rp_type = RPROP_PLUS;
        }

        if (strcmp(tmp, "rprop+") == 0) {
                n->update_algorithm = bp_update_rprop;
                n->rp_type = RPROP_PLUS;
        }

        if (strcmp(tmp, "rprop-") == 0) {
                n->update_algorithm = bp_update_rprop;
                n->rp_type = RPROP_MINUS;
        }

        if (strcmp(tmp, "irprop+") == 0) {
                n->update_algorithm = bp_update_rprop;
                n->rp_type = IRPROP_PLUS;
        }

        if (strcmp(tmp, "irprop-") == 0) {
                n->update_algorithm = bp_update_rprop;
                n->rp_type = IRPROP_MINUS;
        }        

        if (strcmp(tmp, "qprop") == 0) {
                n->update_algorithm = bp_update_qprop;
        }

        if (strcmp(tmp, "dbd") == 0) {
                n->update_algorithm = bp_update_dbd;
        }

        if (n->update_algorithm)
                mprintf(msg, tmp);
}

void load_item_set(char *buf, char *fmt, struct network *n, bool train,
                char *msg)
{
        char tmp[64];
        if (sscanf(buf, fmt, tmp) == 0)
                return;

        if (!n->input) {
                eprintf("cannot load set--'input' group size unknown");
                return;
        }
        if (!n->output) {
                eprintf("cannot load set--'output' group size unknown");
        }

        struct set *s = load_set(tmp, n->input->vector->size,
                        n->output->vector->size);

        if (train)
                n->training_set = s;
        else
                n->test_set = s;

        if (s)
                mprintf(msg, tmp);
}

void load_group(char *buf, char *fmt, struct network *n, char *input,
                char *output, char *msg)
{
        char tmp1[64], tmp2[64], tmp3[64];
        int tmp_int;
        if (sscanf(buf, fmt, tmp1, tmp2, tmp3, &tmp_int) == 0)
                return;

        struct act_fun *act_fun = load_activation_function(tmp2);
        struct err_fun *err_fun = load_error_function(tmp3);
        struct group *g = create_group(tmp1, act_fun, err_fun, tmp_int, false, false);

        if (strcmp(tmp1, input) == 0)
                n->input = g;
        if (strcmp(tmp1, output) == 0)
                n->output = g;

        add_to_group_array(n->groups, g);

        mprintf(msg, tmp1, tmp2, tmp3, tmp_int);
}

struct act_fun *load_activation_function(char *act_fun)
{
        struct act_fun *a;
        if (!(a = malloc(sizeof(struct act_fun))))
                goto error_out;
        memset(a, 0, sizeof(struct act_fun));

        /* binary sigmoid function */
        if (strcmp(act_fun, "binary_sigmoid") == 0) {
                a->fun = act_fun_binary_sigmoid;
                a->deriv = act_fun_binary_sigmoid_deriv;
        }

        /* bipolar sigmoid function */
        if (strcmp(act_fun, "bipolar_sigmoid") == 0) {
                a->fun = act_fun_bipolar_sigmoid;
                a->deriv = act_fun_bipolar_sigmoid_deriv;
        }

        /* softmax activation function */
        if (strcmp(act_fun, "softmax") == 0) {
                a->fun = act_fun_softmax;
                a->deriv = act_fun_softmax_deriv;
        }

        /* hyperbolic tangent function */
        if (strcmp(act_fun, "tanh") == 0) {
                a->fun = act_fun_tanh;
                a->deriv = act_fun_tanh_deriv;
        }

        /* linear function */
        if (strcmp(act_fun, "linear") == 0) {
                a->fun = act_fun_linear;
                a->deriv = act_fun_linear_deriv;
        }

        /* step function */
        if (strcmp(act_fun, "step") == 0) {
                a->fun = act_fun_step;
                a->deriv = act_fun_step_deriv;
        }

        return a;

error_out:
        perror("[load_activation_function()]");
        return NULL;
}

struct err_fun *load_error_function(char *err_fun)
{
        struct err_fun *e;
        if (!(e = malloc(sizeof(struct err_fun))))
                goto error_out;
        memset(e, 0, sizeof(struct err_fun));

        /* sum of squares */
        if (strcmp(err_fun, "sum_squares") == 0) {
                e->fun = error_sum_of_squares;
                e->deriv = error_sum_of_squares_deriv;
        }

        /* cross-entropy */
        if (strcmp(err_fun, "cross_entropy") == 0) {
                e->fun = error_cross_entropy;
                e->deriv = error_cross_entropy_deriv;
        }

        /* divergence */
        if (strcmp(err_fun, "divergence") == 0) {
                e->fun = error_divergence;
                e->deriv = error_divergence_deriv;
        }

        return e;

error_out:
        perror("[load_error_function()]");
        return NULL;
}

void load_bias(char *buf, char *fmt, struct network *n, char *msg)
{
        char tmp1[64];
        if (sscanf(buf, fmt, tmp1) == 0)
                return;

        struct group *g = find_group_by_name(n, tmp1);
        if (g == NULL) {
                eprintf("cannot set bias--group (%s) unknown", tmp1);
                return;
        }

        attach_bias_group(n, g);

        mprintf(msg, tmp1);
}


void load_projection(char *buf, char *fmt, struct network *n, char *msg)
{
        char tmp1[64], tmp2[64];
        if (sscanf(buf, fmt, tmp1, tmp2) == 0)
                return;

        struct group *fg = find_group_by_name(n, tmp1);
        struct group *tg = find_group_by_name(n, tmp2);

        if (fg == NULL) {
                eprintf("cannot set projection--'from' group (%s) unknown",
                                tmp1);
                return;
        }
        if (tg == NULL) {
                eprintf("cannot set projection--'to' group (%s) unknown",
                                tmp2);
                return;
        }

        struct matrix *weights = create_matrix(
                        fg->vector->size,
                        tg->vector->size);
        struct vector *error = create_vector(
                        fg->vector->size);
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
        op = create_projection(tg, weights, error, gradients, prev_gradients,
                        prev_weight_deltas, dyn_learning_pars, false);
        add_to_projs_array(fg->out_projs, op);

        struct projection *ip;
        ip = create_projection(fg, weights, error, gradients, prev_gradients,
                        prev_weight_deltas, dyn_learning_pars, false);
        add_to_projs_array(tg->inc_projs, ip);

        mprintf(msg, tmp1, tmp2);
}

void load_freeze_projection(char *buf, char *fmt, struct network *n,
                char *msg)
{
        char tmp1[64], tmp2[64];
        if (sscanf(buf, fmt, tmp1, tmp2) == 0)
                return;

        struct group *fg = find_group_by_name(n, tmp1);
        struct group *tg = find_group_by_name(n, tmp2);

        if (fg == NULL) {
                eprintf("cannot freeze projection--'from' group (%s) unknown",
                                tmp1);
                return;
        }
        if (tg == NULL) {
                eprintf("cannot freeze projection--'to' group (%s) unknown",
                                tmp2);
                return;
        }

        for (int i = 0; i < fg->out_projs->num_elements; i++)
                if (fg->out_projs->elements[i]->to == tg)
                        fg->out_projs->elements[i]->frozen = true;

        for (int i = 0; i < tg->inc_projs->num_elements; i++)
                if (tg->inc_projs->elements[i]->to == fg)
                        tg->inc_projs->elements[i]->frozen = true;

        mprintf(msg, tmp1, tmp2);
}

void load_recurrent_group(char *buf, char *fmt, struct network *n, 
                char *msg)
{
        char tmp[64];
        if (sscanf(buf, fmt, tmp) == 0)
                return;

        struct group *g = find_group_by_name(n, tmp);

        if (g == NULL) {
                eprintf("cannot set recurrent group--group (%s) unknown",
                                tmp);
                return;
        }

        g->recurrent = true;

        mprintf(msg, tmp, tmp);
}

void load_elman_projection(char *buf, char *fmt, struct network *n, 
                char *msg)
{
        char tmp1[64], tmp2[64];
        if (sscanf(buf, fmt, tmp1, tmp2) == 0)
                return;

        struct group *fg = find_group_by_name(n, tmp1);
        struct group *tg = find_group_by_name(n, tmp2);

        if (fg == NULL) {
                eprintf("cannot set Elman-projection--'from' group (%s) unknown",
                                tmp1);
                return;
        }
        if (tg == NULL) {
                eprintf("cannot set Elman-projection--'from' group (%s) unknown",
                                tmp2);
                return;
        }

        if (fg->vector->size != tg->vector->size) {
                eprintf("cannot set Elman-projection--'from' and 'to' "
                                "group have unequal vector sizes (%d and %d)",
                                fg->vector->size, tg->vector->size);
                return;
        }

        fg->context_group = tg;

        reset_context_groups(n);

        mprintf(msg, tmp1, tmp2);
}

void load_training_order(char *buf, char *fmt, struct network *n,
                char *msg)
{
        char tmp[64];
        if (sscanf(buf, fmt, tmp) == 0)
                return;

        if (strcmp(tmp, "ordered") == 0)
                n->training_order = TRAIN_ORDERED;
        if (strcmp(tmp, "permuted") == 0)
                n->training_order = TRAIN_PERMUTED;
        if (strcmp(tmp, "randomized") == 0)
                n->training_order = TRAIN_RANDOMIZED;

        mprintf(msg, tmp);
}

struct group *find_group_by_name(struct network *n, char *name)
{
        for (int i = 0; i < n->groups->num_elements; i++) {
                char *gn = n->groups->elements[i]->name;
                if (strcmp(gn, name) == 0)
                        return n->groups->elements[i];
        }

        return NULL;
}

/*
 * ########################################################################
 * ## Weight matrix loading and saving                                   ##
 * ########################################################################
 */

void load_weights(struct network *n)
{
        FILE *fd;
        if (!(fd = fopen(n->load_weights_file, "r")))
                goto error_out;

        char buf[4096];
        while (fgets(buf, sizeof(buf), fd)) {
                char tmp1[64], tmp2[64];

                if (!(sscanf(buf, "Projection %s %s", tmp1, tmp2)))
                        continue;

                /* find the groups for the projection */
                struct group *g1, *g2;
                if ((g1 = find_group_by_name(n, tmp1)) == NULL) {
                        eprintf("group does not exist: %s", tmp1);
                        return;
                }
                if ((g2 = find_group_by_name(n, tmp2)) == NULL) {
                        eprintf("group does not exist: %s", tmp2);
                        return;
                }

                /* find the projection */
                struct projection *p;
                for (int i = 0; i < g1->out_projs->num_elements; i++) {
                        p = g1->out_projs->elements[i];
                        if (p->to == g2)
                                break;
                }

                /* read the matrix values */
                for (int r = 0; r < p->weights->rows; r++) {
                        if (!fgets(buf, sizeof(buf), fd))
                                goto error_out;
                        char *tokens = strtok(buf, " ");
                        for (int c = 0; c < p->weights->cols; c++) {
                                if (!sscanf(tokens, "%lf", &p->weights->elements[r][c]))
                                        goto error_out;
                                tokens = strtok(NULL, " ");
                        }
                }
        }

        fclose(fd);

        return;

error_out:
        perror("[load_weight_matrices()]");
        return;
}

void save_weights(struct network *n)
{
        FILE *fd;
        if (!(fd = fopen(n->save_weights_file, "w")))
                goto error_out;

        save_weight_matrices(n->input, fd);

        fclose(fd);

        return;

error_out:
        perror("[save_weight_matrices()]");
        return;
}

void save_weight_matrices(struct group *g, FILE *fd)
{
        /*
         * write the weight matrices of all of the current group's
         * incoming projections
         */
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *p = g->inc_projs->elements[i];
                
                fprintf(fd, "Projection %s %s\n", p->to->name, g->name);
                
                for (int r = 0; r < p->weights->rows; r++) {
                        for (int c = 0; c < p->weights->cols; c++) {
                                fprintf(fd, "%f", p->weights->elements[r][c]);
                                if (c < p->weights->cols - 1)
                                        fprintf(fd, " ");
                        }
                        fprintf(fd, "\n");
                }
                fprintf(fd, "\n");
        }

        /* 
         * repeat the above for all of the current group's
         * outgoing projections
         */
        for (int i = 0; i < g->out_projs->num_elements; i++)
                if (!g->out_projs->elements[i]->recurrent)
                        save_weight_matrices(g->out_projs->elements[i]->to, fd);
}

/*
 * ########################################################################
 * ## Experimental code                                                  ##
 * ########################################################################
 */

void print_weights(struct network *n)
{
        print_projection_weights(n->output);
}

void print_projection_weights(struct group *g)
{
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct group *ng = g->inc_projs->elements[i]->to;

                printf("%s -> %s\n", ng->name, g->name);

                pprint_matrix(g->inc_projs->elements[i]->weights);

                printf("\n");

                print_projection_weights(ng);
        }
}

void print_weight_stats(struct network *n)
{
        struct weight_stats *ws = weight_statistics(n);

        pprint_weight_stats(ws);
}

void print_network_topology(struct network *n)
{
        print_groups(n->output);
}

void print_groups(struct group *g)
{
        printf("\n%s: ", g->name);
        pprint_vector(g->vector);

        for (int i = 0; i < g->inc_projs->num_elements; i++)
                print_groups(g->inc_projs->elements[i]->to);
}
