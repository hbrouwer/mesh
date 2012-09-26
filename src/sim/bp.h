/*
 * bp.h
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

#ifndef BP_H
#define BP_H

#include "network.h"

void bp_backpropagate_error(struct network *n, struct group *g,
                struct vector *e);

void bp_projection_deltas_and_error(struct network *n, struct projection *p,
                struct vector *e);

struct vector *bp_sum_group_error(struct network *n, struct group *g);

void bp_adjust_weights(struct network *n, struct group *g);
void bp_adjust_projection_weights(struct network *n, struct group *g,
                struct projection *p);

#endif /* BP_H */