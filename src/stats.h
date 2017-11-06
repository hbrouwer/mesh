/*
 * Copyright 2012-2017 Harm Brouwer <me@hbrouwer.eu>
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

#ifndef STATS_H
#define STATS_H

#include <stdint.h>

#include "network.h"

struct weight_stats
{
        uint32_t num_weights;       /* number of weights */
        double cost;                /* cost */
        double mean;                /* mean */
        double mean_abs;            /* absolute mean */
        double mean_dist;           /* mean distance */
        double variance;            /* variance */
        double minimum;             /* minimum */
        double maximum;             /* maximum */
};

struct weight_stats *create_weight_statistics(struct network *n);
void dispose_weight_statistics(struct weight_stats *ws);

void collect_weight_statistics(struct weight_stats *ws, struct group *g);
void collect_mean_dependent_ws(struct weight_stats *ws, struct group *g);

void print_weight_statistics(struct network *n, struct weight_stats *ws);

#endif /* STATS_H */
