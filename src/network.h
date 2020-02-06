/*
 * Copyright 2012-2020 Harm Brouwer <me@hbrouwer.eu>
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
        /* network name */
        char *name;
        /* array of groups in the network */
        struct array *groups;
        /* input group */
        struct group *input;
        /* output group */
        struct group *output;
        /* randomization algorithm */
        void (*random_algorithm)(struct matrix *m, struct network *n);
        /* network status */
        struct status *status;
        /* learning algorithm */
        void (*learning_algorithm)(struct network *n);
        /* weight update algorithm */
        void (*update_algorithm)(struct network *n);
        /* two-stage forward group */
        struct group *ts_fw_group;
        /* two-stage forward set */
        struct set *ts_fw_set;
        /* two-stage backward group */
        struct group *ts_bw_group;
        /* two-stage backward set */
        struct set *ts_bw_set;            
        /* sets in this network */
        struct array *sets;
        /* active set pointer */
        struct set *asp;
        /* vector similarity metric */
        double (*similarity_metric)(struct vector *v1, struct vector *v2);
        /* network flags */
        struct network_flags *flags;
        /* network paramaters */
        struct network_params *pars;
        /* unfolded recurrent network */
        struct rnn_unfolded_network *unfolded_net;
};

struct network_flags
{
        enum network_type type;     /* network type */
        bool initialized;           /* flags initialization status */
        bool reset_contexts;        /* flags context group resetting */
        uint32_t sd_type;           /* type of steepest descent */
        uint32_t rp_type;           /* type of Rprop */
        uint32_t training_order;    /* order of training items */
        bool dcs;                   /* flags whether DCS is enabled */
#ifdef _OPENMP
        bool omp_mthreaded;         /* flags if multi-threading is enabled */
#endif /* _OPENMP */   
};

struct network_params
{
        uint32_t random_seed;       /* random number generator seed */
        double random_mu;           /* mu for Gaussian random numbers */
        double random_sigma;        /* sigma for Gaussian random numbers */
        double random_min;          /* minimum for random ranges */
        double random_max;          /* maximum for random ranges */
        double init_context_units;  /* initial value of context units */
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
        uint32_t back_ticks;        /* number of back ticks for BPTT */
        uint32_t batch_size;        /* update after #items */
        double sd_scale_factor;     /* scaling factor */
        double rp_init_update;      /* initial update value for Rprop */        
        double rp_eta_plus;         /* update value increase rate */
        double rp_eta_minus;        /* update value decrease rate */        
        double dbd_rate_increment;  /* LR increment factor for DBD */
        double dbd_rate_decrement;  /* LR decrement factor for DBD */
};

struct rnn_unfolded_network
{
        struct array *rcr_groups;   /* recurrent groups */
        struct array *trm_groups;   /* "terminal" groups */
        uint32_t stack_size;        /* stack size */
        struct network **stack;     /* network stack */
        uint32_t sp;                /* stack pointer */
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
        struct group_flags *flags;  /* flags */
        struct group_params *pars;  /* paramaters */
};

struct group_flags
{
        bool bias;                  /* flags bias group */
};

struct group_params
{
        double relu_alpha;          /* alpha parameter for ReLUs */
        double relu_max;            /* maximum value for ReLUs */
        double logistic_fsc;        /* flat spot correction */
        double logistic_gain;       /* gain coefficient */
        struct set *dcs_set;        /* DSS context event set */
};

                /********************
                 **** projection ****
                 ********************/

struct projection
{
        /* group projected to */
        struct group *to;
        /* projection weights */
        struct matrix *weights;
         /* gradients */
        struct matrix *gradients;
        /* previous gradients */
        struct matrix *prev_gradients;
        /* previous weight deltas */
        struct matrix *prev_deltas;
        /* update values (Rprop) or LRs (DBD) */
        struct matrix *dynamic_params;
        /* flags */
        struct projection_flags *flags;
};

struct projection_flags
{
        bool frozen;                /* flags frozen weights */
        bool recurrent;             /* flags recurrent projections (BPTT) */
};

                /*****************************
                 **** activation function ****
                 *****************************/

struct act_fun 
{
        /* activation function  */
        double (*fun)(struct group *g, uint32_t i);
        /* activation function derivative */
        double (*deriv)(struct group *g, uint32_t i);
};

                /************************
                 **** error function ****
                 ************************/

struct err_fun
{
        /* error function */
        double (*fun)(struct network *, struct group *g, struct vector *t);
        /* error function derivative */
        void (*deriv)(struct network *n, struct group *g, struct vector *t);
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
void inspect_network(struct network *n);

struct group *create_group(char *name, uint32_t size, bool bias,
        bool recurrent);
struct group *create_bias_group(char *name);
struct group *attach_bias_group(struct network *n, struct group *g);
void free_group(struct group *g);
void free_groups(struct array *gs);
void add_group(struct network *n, struct group *g);
void remove_group(struct network *n, struct group *g);
void print_groups(struct network *n);
void reset_groups(struct network *n);

void shift_context_groups(struct network *n);
void shift_context_group_chain(struct group *g, struct vector *v);
void shift_pointer_or_stack(struct network *n);

void reset_stack_pointer(struct network *n);
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
        struct matrix *dynamic_params,
        struct projection_flags *flags);
void free_projection(struct projection *p);
void add_projection(struct array *projs, struct projection *p);
void add_bidirectional_projection(struct group *fg, struct group *tg);
void remove_projection(struct array *projs, struct projection *p);
void remove_bidirectional_projection(
        struct group *fg,
        struct projection *fg_to_tg,
        struct group *tg,
        struct projection *tg_to_fg);
struct projection *find_projection(struct array *projs, struct group *g);
void add_elman_projection(struct group *fg, struct group *tg);
void remove_elman_projection(struct group *fg, struct group *tg);
bool find_elman_projection(struct group *fg, struct group *tg);
void print_projections(struct network *n);
void freeze_projection(struct projection *p);
void unfreeze_projection(struct projection *p);

void free_sets(struct array *sets);
void add_set(struct network *n, struct set *set);
void remove_set(struct network *n, struct set *set);
void print_sets(struct network *n);

void reset_projection_matrices(struct group *g, struct network *n);
void randomize_weight_matrices(struct group *g, struct network *n);
void initialize_dynamic_params(struct group *g, struct network *n);

bool save_weight_matrices(struct network *n, char *filename);
void save_weight_matrix(struct group *g, FILE *fd);
bool load_weight_matrices(struct network *n, char *filename);
bool load_weight_matrix(FILE *fd, struct matrix *weights);

#endif /* NETWORK_H */
