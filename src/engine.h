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

#ifndef ENGINE_H
#define ENGINE_H

#include "network.h"
#include "set.h"
#include "vector.h"

void clamp_input_vector(struct network *n, struct vector *input);
void reset_ticks(struct network *n);
void next_tick(struct network *n);
void forward_sweep(struct network *n);

void inject_error(struct network *n, struct vector *target);
double output_error(struct network *n, struct vector *target);

struct vector *output_vector(struct network *n);
struct group *find_network_group_by_name(struct network *n, char *name);

void reset_error_signals(struct network *n);
void backward_sweep(struct network *n);
void update_weights(struct network *n);

void two_stage_forward_sweep(struct network *n, struct item *item,
        uint32_t event);
void two_stage_backward_sweep(struct network *n, struct item *item,
        uint32_t event);

#endif /* ENGINE_H */
