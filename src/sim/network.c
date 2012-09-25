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
#include "error.h"
#include "network.h"
#include "stats.h"
#include "train.h"

struct network *create_network(char *name)
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

        n->groups = create_group_array(MAX_GROUPS);

        return n;

error_out:
        perror("[create_network()]");
        return NULL;
}

void initialize_network(struct network *n)
{
        mprintf("attempting to initialize network: [%s]", n->name);

        srand(n->random_seed);

        if (!n->load_weights)
                randomize_weight_matrices(n->input, n);

        /* initialize unfolded network */
        if (n->learning_algorithm == train_bptt)
                n->unfolded_net = ffn_init_unfolded_network(n);

        if (!n->unfolded_net && n->load_weights)
                load_weights(n);
        if (n->unfolded_net && n->load_weights)
                load_weights(n->unfolded_net->stack[0]);

        // XXX: todo--sanity checks!

        mprintf("initialized network: [%s]", n->name);
}


void dispose_network(struct network *n)
{
        free(n->name);
        /* dispose_group_array(n->groups); */

        if (n->unfolded_net)
                ffn_dispose_unfolded_network(n->unfolded_net);

        /* dispose_groups(n->output); */
        dispose_groups(n->groups);
        dispose_group_array(n->groups);

        dispose_vector(n->target);
        
        dispose_set(n->training_set);
        dispose_set(n->test_set);

        free(n);
}

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

void increase_group_array_size(struct group_array *gs)
{
        gs->max_elements = gs->max_elements + MAX_GROUPS;
        
        int block_size = gs->max_elements * sizeof(struct group *);
        if (!(gs->elements = realloc(gs->elements, block_size)))
                goto error_out;
        for (int i = gs->num_elements; i < gs->max_elements; i++)
                gs->elements[i] = NULL;

        return;

error_out:
        perror("[increase_group_array_size()]");
        return;
}

void dispose_group_array(struct group_array *gs)
{
        free(gs->elements);
        free(gs);
}

struct group *create_group(char *name, struct act *act, int size, bool bias, 
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
        g->act = act;

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

void attach_bias_group(struct network *n, struct group *g)
{
        char *tmp;
        int block_size = (strlen(g->name) + 6) * sizeof(char);
        if (!(tmp = malloc(block_size)))
                goto error_out;
        memset(tmp, 0, sizeof(block_size));

        sprintf(tmp, "%s_bias", g->name);
        struct group *bg = create_group(tmp, g->act, 1, true, false);

        free(tmp);

        n->groups->elements[n->groups->num_elements++] = bg;
        if (n->groups->num_elements == n->groups->max_elements)
                increase_group_array_size(n->groups);

        struct matrix *weights = create_matrix(
                        bg->vector->size,
                        g->vector->size);
        struct vector *error = create_vector(
                        bg->vector->size);
        struct matrix *deltas = create_matrix(
                        bg->vector->size,
                        g->vector->size);
        struct matrix *prev_deltas = create_matrix(
                        bg->vector->size,
                        g->vector->size);

        bg->out_projs->elements[bg->out_projs->num_elements++] =
                create_projection(g, weights, error, deltas, prev_deltas,
                                false);
        if (bg->out_projs->num_elements == bg->out_projs->max_elements)
                increase_projs_array_size(bg->out_projs);
        
        g->inc_projs->elements[g->inc_projs->num_elements++] =
                create_projection(bg, weights, error, deltas, prev_deltas,
                                false);
        if (g->inc_projs->num_elements == g->out_projs->max_elements)
                increase_projs_array_size(g->inc_projs);

        return;

error_out:
        perror("[attach_bias_group()]");
        return;
}

/*
void dispose_groups(struct group *g)
{
        printf("%s\n", g->name);

        for (int i = 0; i < g->inc_projs->num_elements; i++)
                dispose_groups(g->inc_projs->elements[i]->to);

        free(g->name);
        dispose_vector(g->vector);

        for (int i = 0; i < g->inc_projs->num_elements; i++)
                dispose_projection(g->inc_projs->elements[i]);
        for (int i = 0; i < g->out_projs->num_elements; i++)
                free(g->out_projs->elements[i]);

        dispose_projs_array(g->inc_projs);
        dispose_projs_array(g->out_projs);

        free(g);
}
*/

void dispose_groups(struct group_array *groups)
{
        for (int i = 0; i < groups->num_elements; i++) {
                struct group *g = groups->elements[i];

                free(g->name);
                dispose_vector(g->vector);
                if (!g->bias)
                        free(g->act);

                for (int j = 0; j < g->inc_projs->num_elements; j++)
                        dispose_projection(g->inc_projs->elements[j]);
                for (int j = 0; j < g->out_projs->num_elements; j++)
                        free(g->out_projs->elements[j]);

                dispose_projs_array(g->inc_projs);
                dispose_projs_array(g->out_projs);

                free(g);
        }
}

void shift_context_group_chain(struct network *n, struct group *g,
                struct vector *v)
{
        if (g->context_group)
                shift_context_group_chain(n, g->context_group, g->vector);
        
        copy_vector(g->vector, v);
}

void reset_context_groups(struct network *n)
{
        for (int i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];
                if (g->context_group) {
                        g = g->context_group;
                        for (int j = 0; j < g->vector->size; j++)
                                g->vector->elements[j] = 0.5;
                }
        }
}

void reset_recurrent_groups(struct network *n) {
        for (int i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];
                if (g->recurrent) {
                        for (int j = 0; j < g->vector->size; j++)
                                g->vector->elements[j] = 0.0;
                }
        }
}
        

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

void increase_projs_array_size(struct projs_array *ps)
{
        ps->max_elements = ps->max_elements + MAX_PROJS;

        int block_size = ps->max_elements * sizeof(struct projection *);
        if (!(ps->elements = realloc(ps->elements, block_size)))
                goto error_out;
        for (int i = ps->num_elements; i < ps->max_elements; i++)
                ps->elements[i] = NULL;

        return;

error_out:
        perror("[increase_projs_array_size()]");
        return;
}

void dispose_projs_array(struct projs_array *ps)
{
        free(ps->elements);
        free(ps);
}

struct projection *create_projection(
                struct group *to,
                struct matrix *weights,
                struct vector *error,
                struct matrix *deltas,
                struct matrix *prev_deltas,
                bool recurrent)
{
        struct projection *p;

        if (!(p = malloc(sizeof(struct projection))))
                goto error_out;
        memset(p, 0, sizeof(struct projection));

        p->to = to;
        p->weights = weights;
        p->error = error;
        p->deltas = deltas;
        p->prev_deltas = prev_deltas;
        p->recurrent = recurrent;

        return p;

error_out:
        perror("[create_projection()]");
        return NULL;
}

void dispose_projection(struct projection *p)
{
        dispose_matrix(p->weights);
        dispose_vector(p->error);
        dispose_matrix(p->deltas);
        dispose_matrix(p->prev_deltas);

        free(p);
}

void randomize_weight_matrices(struct group *g, struct network *n)
{
        for (int i = 0; i < g->inc_projs->num_elements; i++)
                randomize_matrix(g->inc_projs->elements[i]->weights, 
                                n->random_mu, n->random_sigma);

        for (int i = 0; i < g->out_projs->num_elements; i++)
                randomize_weight_matrices(g->out_projs->elements[i]->to, n);
}

struct network *load_network(char *filename)
{
        mprintf("attempting to load network: [%s]", filename);

        struct network *n = NULL;

        FILE *fd;
        if (!(fd = fopen(filename, "r")))
                goto error_out;

        char buf[1024];
        while (fgets(buf, sizeof(buf), fd)) {
                char tmp[64], input[64], output[64];

                if (sscanf(buf, "Network %s %s %s", tmp, input, output)) {
                        n = create_network(tmp);
                        mprintf("created network: [%s (%s -> %s)]",
                                        tmp, input, output);
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
                                "set weight decay: [%lf] *** CHEAT ALERT ***");
                load_double_parameter(buf, "ErrorThreshold %lf", &n->error_threshold,
                                "set error threshold: [%lf]");

                load_int_parameter(buf, "MaxEpochs %d", &n->max_epochs,
                                "set maximum number of epochs: [%d]");
                load_int_parameter(buf, "ReportAfter %d", &n->report_after,
                                "report training status after (number of epochs): [%d]");

                load_int_parameter(buf, "HistoryLength %d", &n->history_length,
                                "set BPTT history length: [%d]");

                load_learning_algorithm(buf, "LearningMethod %s", n,
                                "set learning algorithm: [%s]");
                load_error_function(buf, "ErrorFunction %s", n,
                                "set error function: [%s]");

                load_group(buf, "Group %s %s %d", n, input, output,
                                "added group: [%s (%s:%d)]");

                load_bias(buf, "AttachBias %s", n,
                                "attached bias to group: [%s]");

                load_projection(buf, "Projection %s %s", n,
                                "added projection: [%s -> %s]");

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

        /* 'standard' backpropagation */
        if (strncmp(tmp, "bp", 2) == 0)
                n->learning_algorithm = train_bp;

        /* backpropagation through time */
        if (strncmp(tmp, "bptt", 4) == 0)
                n->learning_algorithm = train_bptt;
                        
        if (n->learning_algorithm)
                mprintf(msg, tmp);
}

void load_error_function(char *buf, char *fmt, struct network *n,
                char *msg)
{
        struct error *e;
        if (!(e = malloc(sizeof(struct error))))
                goto error_out;
        memset(e, 0, sizeof(struct error));

        char tmp[64];
        if (sscanf(buf, fmt, tmp) == 0)
                return;

        /* sum of squares */
        if (strcmp(tmp, "sse") == 0) {
                e->fun = error_sum_of_squares;
                e->deriv = error_sum_of_squares_deriv;
        }

        /* cross-entropy */
        if (strcmp(tmp, "cee") == 0) {
                e->fun = error_cross_entropy;
                e->deriv = error_cross_entropy_deriv;
        }

        n->error = e;

        if (n->error)
                mprintf(msg, tmp);

        return;

error_out:
        perror("[load_error_function()]");
        return;
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
        char tmp1[64], tmp2[64];
        int tmp_int;
        if (sscanf(buf, fmt, tmp1, tmp2, &tmp_int) == 0)
                return;

        struct act *act = load_activation_function(tmp2);
        struct group *g = create_group(tmp1, act, tmp_int, false, false);

        if (strcmp(tmp1, input) == 0)
                n->input = g;
        if (strcmp(tmp1, output) == 0) {
                n->output = g;
                /* also create a target vector */
                n->target = create_vector(g->vector->size);
        }

        n->groups->elements[n->groups->num_elements++] = g;
        if (n->groups->num_elements == n->groups->max_elements)
                increase_group_array_size(n->groups);

        mprintf(msg, tmp1, tmp2, tmp_int);
}

struct act *load_activation_function(char *act_fun)
{
        struct act *a;
        if (!(a = malloc(sizeof(struct act))))
                goto error_out;
        memset(a, 0, sizeof(struct act));

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
        struct matrix *deltas = create_matrix(
                        fg->vector->size,
                        tg->vector->size);
        struct matrix *prev_deltas = create_matrix(
                        fg->vector->size,
                        tg->vector->size);

        fg->out_projs->elements[fg->out_projs->num_elements++] =
                create_projection(tg, weights, error, deltas, prev_deltas,
                                false);
        if (fg->out_projs->num_elements == fg->out_projs->max_elements)
                increase_projs_array_size(fg->out_projs);
        
        tg->inc_projs->elements[tg->inc_projs->num_elements++] =
                create_projection(fg, weights, error, deltas, prev_deltas,
                                false);
        if (tg->inc_projs->num_elements == tg->out_projs->max_elements)
                increase_projs_array_size(tg->inc_projs);

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

        n->srn = true;

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

void load_weights(struct network *n)
{
        FILE *fd;
        if (!(fd = fopen(n->weights_file, "r")))
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
        if (!(fd = fopen(n->weights_file, "w")))
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

void print_units(struct network *n)
{
        printf("\n");
        print_group_units_compact(n->output);
        printf("\n");
}

void print_group_units(struct group *g)
{
        for (int i = 0; i < g->vector->size; i++)
                printf("%s(%d)\t", g->name, i);
        printf("\n");
        print_vector(g->vector);
        printf("\n");

        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct group *ng = g->inc_projs->elements[i]->to;
                print_group_units(ng);
        }
}

void print_group_units_compact(struct group *g)
{
        printf("%s:\n", g->name);
        printf("[ ");
        for (int i = 0; i < g->vector->size; i++) {
                double val = g->vector->elements[i];

                print_value_as_symbols(val);

                if (i < g->vector->size - 1)
                        printf (" | ");
        }

        printf(" ]\n\n");

        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct group *ng = g->inc_projs->elements[i]->to;
                print_group_units_compact(ng);
        }
}

void print_weights(struct network *n)
{
        printf("\n");
        struct weight_stats *ws = gather_weight_stats(n);
        double range = ws->maximum - ws->minimum;
        print_projection_weights_compact(range, ws->minimum, n->output);
        printf("\n");
}

void print_projection_weights(struct group *g)
{
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct group *ng = g->inc_projs->elements[i]->to;

                printf("%s -> %s\n", ng->name, g->name);

                print_matrix(g->inc_projs->elements[i]->weights);

                printf("\n");

                print_projection_weights(ng);
        }
}

void print_projection_weights_compact(double range, double minimum, 
                struct group *g)
{
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct group *ng = g->inc_projs->elements[i]->to;

                printf("%s -> %s\n", ng->name, g->name);

                struct matrix *m = g->inc_projs->elements[i]->weights;

                for (int r = 0; r < m->rows; r++) {
                        printf("[ ");
                        for (int c = 0; c < m->cols; c++) {
                                double val = m->elements[r][c] + fabs(minimum);

                                print_value_as_symbols(val / range);

                                if (c < m->cols - 1)
                                        printf(" | ");
                        }
                        printf(" ]\n");
                }
                
                printf("\n");

                print_projection_weights_compact(range, minimum, ng);
        }
}

/*
 * ++
 *  +
 *  -
 * --
 */

void print_value_as_symbols(double value)
{
        if (value >= 0.75)
                printf("++");
        if (value >= 0.50 && value < 0.75)
                printf(" +");
        if (value >= 0.25 && value < 0.50)
                printf(" -");
        if (value < 0.25)
                printf("--");
}

void print_weight_stats(struct network *n)
{
        struct weight_stats *ws = gather_weight_stats(n);

        printf("___weight statistics___\n");
        printf("mean      : %f\n", ws->mean);
        printf("mean abs. : %f\n", ws->mean_abs);
        printf("mean dist.: %f\n", ws->mean_dist);
        printf("variance  : %f\n", ws->variance);
        printf("minimum   : %f\n", ws->minimum);
        printf("maximum   : %f\n", ws->maximum);
        printf("\n");

        free(ws);
}
