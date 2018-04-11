/*
 * Copyright 2012-2018 Harm Brouwer <me@hbrouwer.eu>
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

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#include "array.h"
#include "matrix.h"
#include "rnn_unfold.h"
#include "set.h"
#include "vector.h"

/* network type */
enum network_type
{
        ntype_ffn,
        ntype_srn,
        ntype_rnn
};

/* training order */
enum training_order
{
        train_ordered,
        train_permuted,
        train_randomized
};

                /*****************
                 **** network ****
                 *****************/

struct network
{
        char *name;                 /* network name */
        enum network_type type;     /* network type */
        struct array *groups;       /* array if groups in the network */
        struct group *input;        /* input group */
        struct group *output;       /* output group */
        bool initialized;           /* flags initialization status */
                                    /* randomization algorithm */
        bool reset_contexts;        /* flags context group resetting */
        double init_context_units;  /* initial value of context units */
        void (*random_algorithm)(struct matrix *m, struct network *n);
        uint32_t random_seed;       /* random number generator seed */
        double random_mu;           /* mu for Gaussian random numbers */
        double random_sigma;        /* sigma for Gaussian random numbers */
        double random_min;          /* minimum for random ranges */
        double random_max;          /* maximum for random ranges */
        struct status *status;      /* network status */
        double learning_rate;       /* learning rate (LR) coefficient */
        double lr_scale_factor;     /* LR scale factor */
        double lr_scale_after;      /* LR scale after %epochs */
        double momentum;            /* momentum (MN) coefficient */
        double mn_scale_factor;     /* MN scale factor */
        double mn_scale_after;      /* MN scale after %epochs */
        double weight_decay;        /* weight decay (WD) coefficient */
        double wd_scale_factor;     /* WD scale factor */
        double wd_scale_after;      /* WD scale after %epochs */
        double target_radius;       /* target radius */
        double zero_error_radius;   /* zero error radius */
        double error_threshold;     /* error threshold */
        uint32_t max_epochs;        /* maximum number of training epochs */
        uint32_t report_after;      /* report status after #epochs */
                                    /* learning algorithm */
        void (*learning_algorithm)(struct network *n);
                                    /* weight update algorithm */
        void (*update_algorithm)(struct network *n);
        uint32_t back_ticks;        /* number of back ticks for BPTT */
        uint32_t batch_size;        /* update after #items */
        uint32_t training_order;    /* order of which training items */
        struct group *ms_input;     /* multi-stage input group */
        struct set *ms_set;         /* multi-stage set */
        uint32_t sd_type;           /* type of steepest descent */
        double sd_scale_factor;     /* scaling factor */
        double rp_init_update;      /* initial update value for Rprop */
        double rp_eta_plus;         /* update value increase rate */
        double rp_eta_minus;        /* update value decrease rate */
        uint32_t rp_type;           /* type of Rprop */
        double dbd_rate_increment;  /* LR increment factor for DBD */
        double dbd_rate_decrement;  /* LR decrement factor for DBD */
        struct array *sets;         /* sets in this network */
        struct set *asp;            /* active set pointer */
                                    /* vector similarity metric */
        double (*similarity_metric)(struct vector *v1, struct vector *v2);
                                    /* unfolded recurrent network */
        struct rnn_unfolded_network *unfolded_net;
};

                /***************
                 **** group ****
                 ***************/

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
        bool bias;                  /* flags bias group */
        bool recurrent;             /* flags recurrent group */
};

                /********************
                 **** projection ****
                 ********************/

struct projection
{
        struct group *to;           /* group projected to */
        struct matrix *weights;     /* projection weights */
        bool frozen;                /* flags frozen weights */
        struct matrix *gradients;   /* gradients */
                                    /* previous gradients */
        struct matrix *prev_gradients;
                                    /* previous weight deltas */
        struct matrix *prev_deltas;
                                    /* update values (Rprop) or LRs (DBD) */
        struct matrix *dynamic_params;
        bool recurrent;             /* flags recurrent projections (BPTT) */
};

                /*****************************
                 **** activation function ****
                 *****************************/

struct act_fun 
{
                                    /* activation function  */
        double (*fun)(struct vector *, uint32_t);
                                    /* activation function derivative */
        double (*deriv)(struct vector *, uint32_t);
        struct vector *lookup;      /* activation lookup vector */
};

                /************************
                 **** error function ****
                 ************************/

struct err_fun
{
                                    /* error function */
        double (*fun)(struct group *g, struct vector *t,
                double tr, double zr);
                                    /* error function derivative */
        void (*deriv)(struct group *g, struct vector *t,
                double tr, double zr);
};

                /************************
                 **** network status ****
                 ************************/

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

struct network *create_network(char *name, enum network_type type);
void set_network_defaults(struct network *n);
void init_network(struct network *n);
void reset_network(struct network *n);
void free_network(struct network *n);

struct group *create_group(char *name, uint32_t size, bool bias,
        bool recurrent);
struct group *attach_bias_group(struct network *n, struct group *g);
void free_group(struct group *g);
void free_groups(struct array *gs);

void shift_context_groups(struct network *n);
void shift_context_group_chain(struct group *g, struct vector *v);
void shift_pointer_or_stack(struct network *n);

void reset_context_groups(struct network *n);
void reset_context_group_chain(struct network *n, struct group *g);
void reset_recurrent_groups(struct network *n);
void reset_ffn_error_signals(struct network *n);
void reset_rnn_error_signals(struct network *n);

struct projection *create_projection(
        struct group *to,
        struct matrix *weights,
        struct matrix *gradients,
        struct matrix *prev_gradients,
        struct matrix *prev_deltas,
        struct matrix *dynamic_params);
void free_projection(struct projection *p);

void free_sets(struct array *sets);

void randomize_weight_matrices(struct group *g, struct network *n);
void initialize_dynamic_params(struct group *g, struct network *n);

bool save_weight_matrices(struct network *n, char *filename);
void save_weight_matrix(struct group *g, FILE *fd);
bool load_weight_matrices(struct network *n, char *filename);

#endif /* NETWORK_H */
