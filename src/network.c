/*
 * network.c
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
 * Initialize a network.
 */

void init_network(struct network *n)
{
        n->initialized = false;

        /*
         * ################################################################
         * ## Verify network sanity                                      ##
         * ################################################################
         */

        if (n->groups->num_elements == 0) {
                eprintf("cannot initialize network--network has no groups");
                return;
        }
        if (!n->input) {
                eprintf("cannot initialize network--network has no 'input' group");
                return;
        }
        if (!n->output) {
                eprintf("cannot initialize network--network has no 'output group");
                return;
        }

        /* seed random number generator */
        srand(n->random_seed);

        /* randomize weights matrices */
        randomize_weight_matrices(n->input, n);

        /* initialize dynamic learning paramters */
        initialize_dyn_learning_pars(n->input, n);

        /* 
         * Initialize activation function lookup
         * vectors (if required).
         */
        if (n->use_act_lookup)
                initialize_act_lookup_vectors(n);
        
        if (n->batch_size == 0)
                n->batch_size = n->training_set->num_elements;

        /* initialize unfolded network (if required) */
        if (n->learning_algorithm == train_network_with_bptt)
                n->unfolded_net = rnn_init_unfolded_network(n);

        n->initialized = true;
}

/*
 * Resets a network.
 */

void reset_network(struct network *n)
{
        /* randomize weights matrices */
        randomize_weight_matrices(n->input, n);

        /* initialize dynamic learning paramters */
        initialize_dyn_learning_pars(n->input, n);
}

/*
 * Disposes a network.
 */

void dispose_network(struct network *n)
{
        free(n->name);
        free(n->status);

        if (n->unfolded_net)
                rnn_dispose_unfolded_network(n->unfolded_net);

        dispose_groups(n->groups);
        dispose_group_array(n->groups);

        if (n->training_set)
                dispose_set(n->training_set);
        if (n->test_set)
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

struct group *create_group(char *name, int size, bool bias, bool recurrent)
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

        if (!(g->act_fun = malloc(sizeof(struct act_fun))))
                goto error_out;
        memset(g->act_fun, 0, sizeof(struct act_fun));
        g->act_fun->fun = act_fun_linear;
        g->act_fun->deriv = act_fun_linear_deriv;

        if (!(g->err_fun = malloc(sizeof(struct err_fun))))
                goto error_out;
        memset(g->err_fun, 0, sizeof(struct err_fun));

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
        struct group *bg = create_group(tmp, 1, true, false);

        bg->act_fun->fun = g->act_fun->fun;
        bg->act_fun->deriv = g->act_fun->deriv;
        
        bg->err_fun->fun = g->err_fun->fun;
        bg->err_fun->deriv = g->err_fun->deriv;

        free(tmp);

        /* 
         * Add it to the network's group array.
         */
        add_to_group_array(n->groups, bg);

        /* weight matrix */
        struct matrix *weights = create_matrix(
                        bg->vector->size,
                        g->vector->size);

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
        op = create_projection(g, weights, gradients, prev_gradients,
                                prev_weight_deltas, dyn_learning_pars, false);
        add_to_projs_array(bg->out_projs, op);
        
        /*
         * Create a projection from the bias receiving group to the bias
         * group.
         */
        struct projection *ip;
        ip = create_projection(bg, weights, gradients, prev_gradients,
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
                if (g->act_fun->lookup)
                        dispose_vector(g->act_fun->lookup);
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
 * This resets the error vector for groups.
 */
void reset_error_signals(struct network *n)
{
        for (int i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];
                zero_out_vector(g->error);
        }
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
                n->random_algorithm(g->inc_projs->elements[i]->weights, n);

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
 * Initializes activation lookup vector.
 */

void initialize_act_lookup_vectors(struct network *n)
{
        for (int i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];

                if (g == n->input || g->bias)
                        continue;

                g->act_fun->lookup = create_act_lookup_vector(g->act_fun->fun);
        }
}

/*
 * Find a group by name.
 */

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
 * ## Weight matrix saving and loading                                   ##
 * ########################################################################
 */

/*
 * Save weight matrices to file.
 */

void save_weight_matrices(struct network *n, char *fn)
{
        FILE *fd;
        if (!(fd = fopen(fn, "w")))
                goto error_out;

        if (n->type == TYPE_FFN)
                save_weight_matrix(n->input, fd);
        if (n->type == TYPE_SRN)
                save_weight_matrix(n->input, fd);
        if (n->type == TYPE_RNN)
                save_weight_matrix(n->unfolded_net->stack[0]->input, fd);

        fclose(fd);

        return;

error_out:
        perror("[save_weight_matrices()]");
        return;
}

void save_weight_matrix(struct group *g, FILE *fd)
{
        /*
         * Write the weight matrices of all of the current group's
         * incoming projections.
         */
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *p = g->inc_projs->elements[i];
                
                fprintf(fd, "%s -> %s\n", p->to->name, g->name);
                
                for (int r = 0; r < p->weights->rows; r++) {
                        for (int c = 0; c < p->weights->cols; c++) {
                                fprintf(fd, "%f", p->weights->elements[r][c]);
                                if (c < p->weights->cols - 1)
                                        fprintf(fd, " ");
                        }
                        fprintf(fd, "\n");
                }

                mprintf("wrote weights for projection: [%s -> %s]",
                                p->to->name, g->name);
        }

        /* 
         * Repeat the above for all of the current group's
         * outgoing projections.
         */
        for (int i = 0; i < g->out_projs->num_elements; i++)
                if (!g->out_projs->elements[i]->recurrent)
                        save_weight_matrix(g->out_projs->elements[i]->to, fd);
}

/*
 * Load weight matrices from file.
 */

void load_weight_matrices(struct network *n, char *fn)
{
        FILE *fd;
        if (!(fd = fopen(fn, "r")))
                goto error_out;

        struct network *np = NULL;
        if (n->type == TYPE_FFN)
                np = n;
        if (n->type == TYPE_SRN)
                np = n;
        if (n->type == TYPE_RNN)
                np = n->unfolded_net->stack[0];

        char buf[8192];
        while (fgets(buf, sizeof(buf), fd)) {
                char tmp1[64], tmp2[64];

                if (sscanf(buf, "%s -> %s", tmp1, tmp2) != 2)
                        continue;

                /* find the groups for the projection */
                struct group *g1, *g2;
                if ((g1 = find_group_by_name(np, tmp1)) == NULL) {
                        eprintf("group does not exist: %s", tmp1);
                        continue;
                }
                if ((g2 = find_group_by_name(np, tmp2)) == NULL) {
                        eprintf("group does not exist: %s", tmp2);
                        continue;
                }

                /* find the projection */
                struct projection *p = NULL;
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

                mprintf("read weights for projection: [%s -> %s]",
                                tmp1, tmp2);
        }

        fclose(fd);

        return;

error_out:
        perror("[load_weight_matrices()]");
        return;
}
