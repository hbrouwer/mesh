/*
 * rnn_unfold.h
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

#ifndef RNN_UNFOLD_H
#define RNN_UNFOLD_H

#include "network.h"
#include "vector.h"

struct rnn_unfolded_network
{
        struct group_array        /* recurrent groups in the network */
                *recur_groups;    
        struct matrix             /* weights for recurrent connections */
                **recur_weights;
        struct matrix             /* previous weight changes for recurrent connections */
                **recur_prev_weight_deltas;
        struct matrix             /* previous Rprop update values or for DBD learning rates recurrent connection */
                **recur_dyn_learning_pars;
        int stack_size;           /* size of the network 'state' stack */
        struct network **stack;   /* stack for different 'states' of the
                                     network */
};

struct rnn_unfolded_network *rnn_init_unfolded_network(struct network *n);
void rnn_dispose_unfolded_network(struct rnn_unfolded_network *un);

struct network *rnn_duplicate_network(struct network *n);
void rnn_dispose_duplicate_network(struct network *n);

struct group *rnn_duplicate_group(struct group *g);
struct group *rnn_duplicate_groups(struct network *n, struct network *dn,
                struct group *g);
void rnn_dispose_duplicate_groups(struct group *dg);

struct projection *rnn_duplicate_projection(
                struct projection *p,
                struct vector *error,
                struct matrix *gradients,
                struct matrix *prev_gradients);
void rnn_dispose_duplicate_projection(struct projection *dp);

struct group_array *rnn_recurrent_groups(struct network *n);
void rnn_collect_recurrent_groups(struct group *g, struct group_array *gs);

void rnn_attach_recurrent_groups(struct rnn_unfolded_network *un,
                struct network *n);
void rnn_detach_recurrent_groups(struct rnn_unfolded_network *un,
                struct network *n);

void rnn_connect_duplicate_networks(struct rnn_unfolded_network *un,
                struct network *n1, struct network *n2);
void rnn_disconnect_duplicate_networks(struct rnn_unfolded_network *un,
                struct network *n1, struct network *n2);

void rnn_sum_gradients(struct rnn_unfolded_network *un);
void rnn_add_gradients(struct group *g1, struct group *g2);

void rnn_cycle_stack(struct rnn_unfolded_network *un);

#endif /* RNN_UNFOLD_H */