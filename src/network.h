/*
 * network.h
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

#ifndef NETWORK_H
#define NETWORK_H

#include "array.h"
#include "main.h"
#include "matrix.h"
#include "rnn_unfold.h"
#include "set.h"
#include "vector.h"

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
        struct array *groups;       /* array of groups in the network */

        struct group *input;        /* input group for this network */
        struct group *output;       /* output group for this network */

        bool act_lookup;            /* use activation lookup vectors */

        bool initialized;           /* flags whether this network is
                                       intialized */

        /* ## Random numbers ########################################### */

        void (*random_algorithm)    /* randomization algorithm */
                (struct matrix *m, struct network *n);

        int random_seed;            /* seed for the random number
                                       generator */
        double random_mu;           /* mu for normally distributed random
                                       matrices */
        double random_sigma;        /* sigma for normally distributed random
                                       matrices */
        double random_min;          /* minimum value for range randomized
                                       matrices */
        double random_max;          /* maximum value for range randomized
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

        double target_radius;       /* target radius */
        double zero_error_radius;   /* zero error radius */

        int max_epochs;             /* maximum number of training epochs */
        int report_after;           /* number of training epochs after
                                       which to report status */

        void (*learning_algorithm)  /* learning algorithm */
                (struct network *n);
        void (*update_algorithm)    /* weight update algorithm */
                (struct network *n);

        int history_length;         /* number of history timesteps for BPTT */

        int batch_size;             /* size of batch after which to update
                                       weights */
        
        int training_order;         /* order in which training items are
                                       presented */        

        /* ## Steepest descent parameters ############################## */

        int sd_type;                /* type of steepest descent */
        double sd_scale_factor;     /* scaling factor */

        /* ## Rprop parameters ######################################### */

        double rp_init_update;      /* initial update value for Rprop */
        double rp_eta_plus;         /* rate with which update values are
                                       increased */
        double rp_eta_minus;        /* rate with which update values are
                                       decreased */
        int rp_type;                /* type of Rprop */

        /* ## Delta-Bar-Delta parameters ############################### */

        double dbd_rate_increment;  /* learning rate increment factor for DBD */
        double dbd_rate_decrement;  /* learning rate decrement factor for DBD */

        /* ## Sets ##################################################### */

        struct array *sets;         /* sets */
        struct set *asp;            /* active set pointer */

        /* ## Unfolded recurrent network ############################### */

        struct rnn_unfolded_network /* unfolded recurrent network */
                *unfolded_net;
};

/*
 * ########################################################################
 * ## Groups                                                             ##
 * ########################################################################
 */

struct group
{
        char *name;                 /* name of the group */
        struct vector *vector;      /* the "neurons" of this group */
        struct vector *error;       /* error vector for this group */
        struct act_fun *act_fun;    /* activation function and its derivative
                                       for this group */
        struct err_fun *err_fun;    /* error function and its derivative
                                       for this group */
        struct array                /* array of incoming projections */
                *inc_projs;
        struct array                /* array of outgoing projections */
                *out_projs;

        struct array                /* array of context groups */
                *ctx_groups;
        
        bool bias;                  /* flags whether this is a bias group */
        bool recurrent;             /* flags whether this is a recurrent group */
};

/*
 * ########################################################################
 * ## Projections                                                        ##
 * ########################################################################
 */

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
                (struct group *g, struct vector *t, double tr, double zr);
        void(*deriv)                /* error function derivative */
                (struct group *g, struct vector *t, double tr, double zr);
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
void init_network(struct network *n);
void reset_network(struct network *n);
void dispose_network(struct network *n);

struct group *create_group(char *name, int size, bool bias, bool recurrent);
struct group *attach_bias_group(struct network *n, struct group *g);
void dispose_group(struct group *g);
void dispose_groups(struct array *gs);

void shift_context_groups(struct network *n);
void shift_context_group_chain(struct group *g, struct vector *v);
void reset_context_groups(struct network *n);
void reset_context_group_chain(struct group *g);
void reset_recurrent_groups(struct network *n);
void reset_error_signals(struct network *n);

struct projection *create_projection(
                struct group *to,
                struct matrix *weights,
                struct matrix *gradients,
                struct matrix *prev_gradients,
                struct matrix *prev_weight_deltas,
                struct matrix *dyn_learning_pars,
                bool recurrent);
void dispose_projection(struct projection *p);

void dispose_sets(struct array *ss);

void randomize_weight_matrices(struct group *g, struct network *n);
void initialize_dyn_learning_pars(struct group *g, struct network *n);
void initialize_act_lookup_vectors(struct network *n);

bool save_weight_matrices(struct network *n, char *fn);
void save_weight_matrix(struct group *g, FILE *fd);
bool load_weight_matrices(struct network *n, char *fn);

#endif /* NETWORK_H */
