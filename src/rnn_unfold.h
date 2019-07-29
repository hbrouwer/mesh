/*
 * Copyright 2012-2019 Harm Brouwer <me@hbrouwer.eu>
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

#include <stdint.h>

#include "array.h"
#include "matrix.h"
#include "network.h"

struct rnn_unfolded_network *rnn_unfold_network(struct network *n);
void rnn_find_recurrent_groups(struct group *g, struct array *rcr_groups);
void rnn_free_unfolded_network(struct rnn_unfolded_network *un);

struct group *rnn_duplicate_group(struct group *g);
struct group *rnn_duplicate_groups(struct network *n, struct network *dn,
        struct group *g);
void rnn_free_duplicate_group(struct group *g);
void rnn_free_duplicate_groups(struct array *dgs);

struct projection *rnn_duplicate_projection(
        struct group *to,
        struct projection *p,
        struct matrix *gradients,
        struct matrix *prev_gradients);
void rnn_free_duplicate_projection(struct projection *dp);

struct network *rnn_duplicate_network(struct network *n);
void rnn_free_duplicate_network(struct network *n);

void rnn_attach_terminal_groups(struct rnn_unfolded_network *un,
        struct network *n);
void rnn_detach_terminal_groups(struct rnn_unfolded_network *un,
        struct network *n);

void rnn_connect_duplicate_networks(struct rnn_unfolded_network *un,
        struct network *n, struct network *nn);
void rnn_disconnect_duplicate_networks(struct rnn_unfolded_network *un,
        struct network *n, struct network *nn);

void rnn_sum_and_reset_gradients(struct rnn_unfolded_network *un);
void rnn_add_and_reset_gradients(struct group *g, struct group *dg);

void rnn_shift_stack(struct rnn_unfolded_network *un);

#endif /* RNN_UNFOLD_H */
