/*
 * network.h
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

#ifndef NETWORK_H
#define NETWORK_H

#include "ffn_unfold.h"
#include "main.h"
#include "matrix.h"
#include "set.h"
#include "vector.h"

#define MAX_GROUPS 5

#define MAX_PROJS 2

/*
 * Network
 */

struct network
{
        char *name;                 /* name of the network */

        struct group_array *groups; /* groups in the network */

        double learning_rate;       /* learning rate */
        double momentum;            /* momentum */
        double weight_decay;        /* weight decay */

        double mse_threshold;       /* mean squared error threshold */

        int max_epochs;             /* maximum number of training epochs */
        int epoch_length;           /* length of a training epoch */

        int history_length;         /* number of timesteps for BPTT */

        double (*act_fun)           /* activation function for hidden groups */
                (double);      
        double (*act_fun_deriv)     /* derivative of the activation function */
                (double); 

        double (*out_act_fun)       /* activation function for the ouput group */
                (double);
        double (*out_act_fun_deriv) /* derivative of the activation function */
                (double); 

        void (*learning_algorithm)  /* learning algorithm */
                (struct network *n);

        struct vector               /* error measure */
                *(*error_measure) (struct network *n);

        struct group *input;        /* input group for this network */
        struct group *output;       /* output group for this network */

        struct vector *target;      /* target vector for a training instance */

        struct set *training_set;   /* training set */
        struct set *test_set;       /* test set */

        /*
         * ################################################################
         * ## Data structure for unfolded feed forward networks.         ##
         * ################################################################
         */

        struct ffn_unfolded_network /* unfolded feed forward network */
                *unfolded_net;
};

/*
 * Group array
 */ 
struct group_array
{
        int num_elements;          /* number of groups */
        int max_elements;          /* maximum number of groups */
        struct group **elements;   /* the actual groups */
};

/*
 * Neuronal group
 */

struct group
{
        char *name;                 /* name of the group */

        struct vector *vector;      /* the "neurons" of this group */

        struct projs_array          /* incoming projections */
                *inc_projs;
        struct projs_array          /* outgoing projections */
                *out_projs; 

        struct group *elman_proj;   /* context group for Elman-type topologies */

        bool bias;                  /* flags whether this is a bias group */

        bool recurrent;             /* flags whether this is a recurrent group
                                       for BPTT */
};

struct projs_array
{
        int num_elements;           /* number of projections */
        int max_elements;           /* maximum number of projections */
        struct projection           /* the actual projections */
                **elements;
};

/*
 * Group projection
 */
struct projection
{
        struct group *to;           /* the group towards which is projected */

        struct matrix *weights;     /* projection weights */

        struct vector *error;       /* projection error for BP */

        struct matrix *deltas;      /* projection deltas for BP */
        struct matrix *prev_deltas; /* previous projection deltas for momentum
                                       in BP */

        bool recurrent;             /* flags whether this is a recurrent
                                       projection for BPTT */
};

/*
 * Function prototypes
 */

struct network *create_network(char *name);
void dispose_network(struct network *n);

struct group_array *create_group_array(int max_elements);
void increase_group_array_size(struct group_array *gs);
void dispose_group_array(struct group_array *gs);

struct group *create_group(char *name, int size, bool bias, bool recurrent);
void attach_bias_group(struct network *n, struct group *g);
void dispose_groups(struct group *g);

struct projs_array *create_projs_array(int max_elements);
void increase_projs_array_size(struct projs_array *ps);
void dispose_projs_array(struct projs_array *ps);

struct projection *create_projection(
                struct group *to,
                struct matrix *weights,
                struct vector *error,
                struct matrix *deltas,
                struct matrix *prev_deltas,
                bool recurrent);
void dispose_projection(struct projection *p);

struct network *load_network(char *filename);

void load_double_parameter(char *buf, char *fmt, double *par, char *msg);
void load_int_parameter(char *buf, char *fmt, int *par, char *msg);
void load_act_function(char *buf, char *fmt, struct network *n,
                bool output, char *msg);
void load_learning_algorithm(char *buf, char *fmt, struct network *n,
                char *msg);
void load_error_measure(char *buf, char *fmt, struct network *n,
                char *msg);
void load_item_set(char *buf, char *fmt, struct network *n, bool train,
                char *msg);
void load_group(char *buf, char *fmt, struct network *n, char *input,
                char *output, char *msg);
void load_projection(char *buf, char *fmt, struct network *n, char *msg);
void load_elman_projection(char *buf, char *fmt, struct network *n,
                char *msg);
void load_recurrent_group(char *buf, char *fmt, struct network *n,
                char *msg);

struct group *find_group_by_name(struct network *n, char *name);

/* experimental */
void print_units(struct network *n);
void print_group_units(struct group *g);
void print_group_units_compact(struct group *g);
void print_weights(struct network *n);
void print_projection_weights(struct group *g);
void print_weight_stats(struct network *n);

#endif /* NETWORK_H */
