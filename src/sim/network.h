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

#include "main.h"
#include "matrix.h"
#include "rnn_unfold.h"
#include "set.h"
#include "vector.h"

#define MAX_GROUPS 5
#define MAX_PROJS 2

/* network types */
#define TYPE_FFN 0
#define TYPE_SRN 1
#define TYPE_RNN 2

/* training orders */
#define TRAIN_ORDERED 0
#define TRAIN_PERMUTED 1
#define TRAIN_RANDOMIZED 2

/*
 * ########################################################################
 * ## Network                                                            ##
 * ########################################################################
 */

struct network
{
        /* ## General ################################################## */

        char *name;                 /* name of the network */
        int type;                   /* type of the network */
        struct group_array *groups; /* array of groups in the network */

        struct group *input;        /* input group for this network */
        struct group *output;       /* output group for this network */

        bool use_act_lookup;        /* use activation lookup vectors */

        /* ## Random numbers ########################################### */

        int random_seed;            /* seed for the random number
                                       generator */
        double random_mu;           /* mu for normally distributed random
                                       matrices */
        double random_sigma;        /* sigma for normally distributed random
                                       matrices */

        /* ## Learning parameters ###################################### */

        struct status *status;      /* network status */

        double learning_rate;       /* learning rate coefficient */
        double lr_scale_factor;     /* scale factor for the learning rate */
        double lr_scale_after;      /* fraction of maximum number of epochs
                                       after which to scale the learning rate */

        double momentum;            /* momentum coefficient */
        double mn_scale_factor;     /* scale factor for momentum */
        double mn_scale_after;      /* fraction of maximum number of epochs
                                       after which to scale the momentum */
        
        double weight_decay;        /* weight decay coefficient */

        double error_threshold;     /* error threshold */

        int max_epochs;             /* maximum number of training epochs */
        int report_after;           /* number of training epochs after
                                       which to report status */

        void (*learning_algorithm)  /* learning algorithm */
                (struct network *n);
        void (*update_algorithm)    /* weight update algorithm */
                (struct network *n);

        int history_length;         /* number of history timesteps for BPTT */

        struct set *training_set;   /* set of training items */
        struct set *test_set;       /* set of test items */

        int batch_size;             /* size of batch after which to update
                                       weights */
        
        int training_order;         /* order in which training items are
                                       presented */        

        /* ## Rprop parameters ######################################### */

        double rp_init_update;      /* initial update values for Rprop */
        double rp_eta_plus;         /* rate with which update values are
                                       increased */
        double rp_eta_minus;        /* rate with which update values are
                                       decreased */
        int rp_type;                /* type of Rprop */

        /* ## Delta-Bar-Delta parameters ############################### */

        double dbd_rate_increment;  /* learning rate increment factor for DBD */
        double dbd_rate_decrement;  /* learning rate decrement factor for DBD */

        /* ## Loading and saving weight matrices ####################### */

        char *save_weights_file;    /* file to which weights should be saved */
        char *load_weights_file;    /* file from which weights should be loaded */

        /* ## Unfolded recurrent network ############################### */

        struct rnn_unfolded_network /* unfolded recurrent network */
                *unfolded_net;

        /* ## ERP correlates ########################################### */

        bool compute_erps;          /* flags whether ERP correlates should
                                       be computed */
};

/*
 * ########################################################################
 * ## Groups                                                             ##
 * ########################################################################
 */

struct group_array
{
        int num_elements;          /* number of groups */
        int max_elements;          /* maximum number of groups */
        struct group **elements;   /* the actual groups */
};

struct group
{
        char *name;                 /* name of the group */
        struct vector *vector;      /* the "neurons" of this group */
        struct vector *error;       /* error vector for this group */
        struct act_fun *act_fun;    /* activation function and its derivative
                                       for this group */
        struct err_fun *err_fun;    /* error function and its derivative
                                       for this group */
        struct projs_array          /* array of incoming projections */
                *inc_projs;
        struct projs_array          /* array of outgoing projections */
                *out_projs; 
        struct group                /* context group for Elman-type topologies */
                *context_group;
        bool bias;                  /* flags whether this is a bias group */
        bool recurrent;             /* flags whether this is a recurrent group */
};

/*
 * ########################################################################
 * ## Projections                                                        ##
 * ########################################################################
 */

struct projs_array
{
        int num_elements;           /* number of projections */
        int max_elements;           /* maximum number of projections */
        struct projection           /* the actual projections */
                **elements;
};

struct projection
{
        struct group *to;           /* the group towards which is projected */
        struct matrix *weights;     /* projection weights */
        bool frozen;                /* flags whether weights for this projection
                                       are frozen */
        struct matrix *gradients;   /* projection gradients for BP */
        struct matrix               /* previous projection gradients for BP */
                *prev_gradients;
        struct matrix               /* previous weight deltas */
                *prev_weight_deltas;
        struct matrix               /* update values for Rprop or 
                                       learning rates for DBD */
                *dyn_learning_pars;
        bool recurrent;             /* flags whether this is a recurrent
                                       projection for BPTT */
};

/*
 * ########################################################################
 * ## Activation function and derivative                                 ##
 * ########################################################################
 */

struct act_fun 
{
        double (*fun)               /* activation function  */
                (struct vector *, int);
        double (*deriv)             /* activation function derivative */
                (struct vector *, int);
        struct vector *lookup;      /* activation lookup vector */
};

/*
 * ########################################################################
 * ## Error function and derivative                                      ##
 * ########################################################################
 */

struct err_fun
{
        double (*fun)               /* error function */
                (struct group *g, struct vector *t);
        void(*deriv)                /* error function derivative */
                (struct group *g, struct vector *t);
};

/*
 * ########################################################################
 * ## Network status                                                     ##
 * ########################################################################
 */

struct status
{
        int epoch;                  /* current training epoch */
        double error;               /* network error */
        double prev_error;          /* previous network error */
        double weight_cost;         /* weight cost */
        double gradient_linearity;  /* gradient linearity */
        double                      /* length of last weight changes vector */ 
                last_weight_deltas_length;
        double gradients_length;    /* length of weight gradients vector */
};

/*
 * ########################################################################
 * ## Function prototypes                                                ##
 * ########################################################################
 */

struct network *create_network(char *name, int type);
void initialize_network(struct network *n);
void dispose_network(struct network *n);

struct group_array *create_group_array(int max_elements);
void add_to_group_array(struct group_array *gs, struct group *g);
void increase_group_array_size(struct group_array *gs);
void dispose_group_array(struct group_array *gs);

struct group *create_group(
                char *name,
                struct act_fun *act_fun,
                struct err_fun *err_fun,
                int size,
                bool bias, 
                bool recurrent);
void attach_bias_group(struct network *n, struct group *g);
void dispose_group(struct group *g);
void dispose_groups(struct group_array *groups);

void shift_context_group_chain(struct network *n, struct group *g,
                struct vector *v);
void reset_error_signals(struct network *n);
void reset_context_groups(struct network *n);
void reset_recurrent_groups(struct network *n);

struct projs_array *create_projs_array(int max_elements);
void add_to_projs_array(struct projs_array *ps, struct projection *p);
void increase_projs_array_size(struct projs_array *ps);
void dispose_projs_array(struct projs_array *ps);

struct projection *create_projection(
                struct group *to,
                struct matrix *weights,
                struct matrix *gradients,
                struct matrix *prev_gradients,
                struct matrix *prev_weight_deltas,
                struct matrix *dyn_learning_pars,
                bool recurrent);
void dispose_projection(struct projection *p);

void randomize_weight_matrices(struct group *g, struct network *n);
void initialize_dyn_learning_pars(struct group *g, struct network *n);
void initialize_act_lookup_vectors(struct network *n);

/*
 * Stuff below will move to cmd.c ...
 */

struct act_fun *load_activation_function(char *act_fun);
struct err_fun *load_error_function(char *err_fun);

struct group *find_group_by_name(struct network *n, char *name);

void load_weights(struct network *n);
void save_weights(struct network *n);
void save_weight_matrices(struct group *g, FILE *fd);

#endif /* NETWORK_H */
