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

#include <stdint.h>

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
#define TRAIN_ORDERED    0
#define TRAIN_PERMUTED   1
#define TRAIN_RANDOMIZED 2

/**************************************************************************
 *************************************************************************/
struct network
{
        char *name;                 /* network name */
        uint32_t type;              /* network type */
        
        struct array *groups;       /* groups in the network */
        struct group *input;        /* input group */
        struct group *output;       /* output group */

        bool act_lookup;            /* use activation lookup */
        bool initialized;           /* flags initialization status */

        void (*random_algorithm)    /* randomization algorithm */
                (struct matrix *m, struct network *n);
        uint32_t random_seed;       /* random number generator seed */
        double random_mu;           /* mu for Gaussian random numbers */
        double random_sigma;        /* sigma for Gaussian random numbers */
        double random_min;          /* minimum for random ranges */
        double random_max;          /* maximum for random ranges */

        struct status *status;      /* network status */
        
        double learning_rate;       /* learning rate coefficient */
        double lr_scale_factor;     /* LR scale factor */
        double lr_scale_after;      /* LR scale after %epochs */

        double momentum;            /* momentum coefficient */
        double mn_scale_factor;     /* MN scale factor */
        double mn_scale_after;      /* MN scale after %epochs */
        
        double weight_decay;        /* weight decay coefficient */
        double wd_scale_factor;     /* WD scale factor */
        double wd_scale_after;      /* WD scale after %epochs */

        double target_radius;       /* target radius */
        double zero_error_radius;   /* zero error radius */

        double error_threshold;     /* error threshold */
        uint32_t max_epochs;        /* maximum number of training epochs */
        uint32_t report_after;      /* report status after #epochs
                                       number of training epochs after
                                       which to report status */

        void (*learning_algorithm)  /* learning algorithm */
                (struct network *n);
        void (*update_algorithm)    /* weight update algorithm */
                (struct network *n);

        uint32_t back_ticks;        /* number of back ticks for BPTT */

        uint32_t batch_size;        /* update after #items */
        uint32_t training_order;    /* order in which training items are
                                       presented */        

        uint32_t sd_type;           /* type of steepest descent */
        double sd_scale_factor;     /* scaling factor */

        double rp_init_update;      /* initial update value for Rprop */
        double rp_eta_plus;         /* update value increase rate */
        double rp_eta_minus;        /* update value decrease rate */
        uint32_t rp_type;                /* type of Rprop */

        double dbd_rate_increment;  /* LR increment factor for DBD */
        double dbd_rate_decrement;  /* LR decrement factor for DBD */

        struct array *sets;         /* sets in this network */
        struct set *asp;            /* active set pointer */

        struct rnn_unfolded_network /* unfolded recurrent network */
                *unfolded_net;
};

/**************************************************************************
 *************************************************************************/
struct group
{
        char *name;                 /* name of the group */
        
        struct vector *vector;      /* the "neurons" of this group */
        struct vector *error;       /* error vector for this group */

        struct act_fun *act_fun;    /* activation functions */
        struct err_fun *err_fun;    /* error functions */
                                       
        struct array *inc_projs;    /* array of incoming projections */
        struct array *out_projs;    /* array of outgoing projections */

        struct array *ctx_groups;   /* array of context groups */
        
        bool bias;                  /* flags bias groups */
        bool recurrent;             /* flags recurrent groups */
};

/**************************************************************************
 *************************************************************************/
struct projection
{
        struct group *to;           /* group projected to */
        struct matrix *weights;     /* projection weights */
        bool frozen;                /* flags frozen weights */
        
        struct matrix *gradients;   /* gradients */
        struct matrix               /* previous gradients */
                *prev_gradients;
        struct matrix               /* previous weight deltas */
                *prev_deltas;
        struct matrix               /* update values (Rprop) or LRs (DBD) */
                *dynamic_pars;

        bool recurrent;             /* flags recurrent projections (BPTT) */
};

/**************************************************************************
 *************************************************************************/
struct act_fun 
{
        double (*fun)               /* activation function  */
                (struct vector *, uint32_t);
        double (*deriv)             /* activation function derivative */
                (struct vector *, uint32_t);
        struct vector *lookup;      /* activation lookup vector */
};

/**************************************************************************
 *************************************************************************/
struct err_fun
{
        double (*fun)               /* error function */
                (struct group *g, struct vector *t, double tr, double zr);
        void (*deriv)               /* error function derivative */
                (struct group *g, struct vector *t, double tr, double zr);
};

/**************************************************************************
 *************************************************************************/
struct status
{
        uint32_t epoch;             /* current training epoch */
        double error;               /* network error */
        double prev_error;          /* previous network error */
        double weight_cost;         /* weight cost */
        double gradient_linearity;  /* gradient linearity */
        double last_deltas_length;  /* length of last weight changes vector */ 
        double gradients_length;    /* length of weight gradients vector */
};

/**************************************************************************
 *************************************************************************/
struct network *create_network(char *name, uint32_t type);
void init_network(struct network *n);
void reset_network(struct network *n);
void dispose_network(struct network *n);

/**************************************************************************
 *************************************************************************/
struct group *create_group(char *name, uint32_t size, bool bias,
                bool recurrent);
struct group *attach_bias_group(struct network *n, struct group *g);
void dispose_group(struct group *g);
void dispose_groups(struct array *gs);

/**************************************************************************
 *************************************************************************/
void shift_context_groups(struct network *n);
void shift_context_group_chain(struct group *g, struct vector *v);

/**************************************************************************
 *************************************************************************/
void reset_context_groups(struct network *n);
void reset_context_group_chain(struct group *g);
void reset_recurrent_groups(struct network *n);
void reset_ffn_error_signals(struct network *n);
void reset_rnn_error_signals(struct network *n);

/**************************************************************************
 *************************************************************************/
struct projection *create_projection(
                struct group *to,
                struct matrix *weights,
                struct matrix *gradients,
                struct matrix *prev_gradients,
                struct matrix *prev_deltas,
                struct matrix *dynamic_pars,
                bool recurrent);
void dispose_projection(struct projection *p);

/**************************************************************************
 *************************************************************************/
void dispose_sets(struct array *ss);

/**************************************************************************
 *************************************************************************/
void randomize_weight_matrices(struct group *g, struct network *n);
void initialize_dynamic_pars(struct group *g, struct network *n);
void initialize_act_lookup_vectors(struct network *n);

/**************************************************************************
 *************************************************************************/
bool save_weight_matrices(struct network *n, char *fn);
void save_weight_matrix(struct group *g, FILE *fd);
bool load_weight_matrices(struct network *n, char *fn);

#endif /* NETWORK_H */
