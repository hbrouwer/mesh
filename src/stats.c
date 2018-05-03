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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "main.h"
#include "math.h"
#include "matrix.h"
#include "stats.h"

struct weight_stats *create_weight_statistics(struct network *n)
{
        struct weight_stats *ws;
        if (!(ws = malloc(sizeof(struct weight_stats))))
                goto error_out;
        memset(ws, 0, sizeof(struct weight_stats));

        collect_weight_statistics(ws, n->output);
        ws->mean /= ws->num_weights;
        ws->mean_abs /= ws->num_weights;

        collect_mean_dependent_ws(ws, n->output);
        ws->mean_dist /= ws->num_weights;
        ws->variance /= (ws->num_weights - 1);

        return ws;

error_out:
        perror("[weight_statistics()]");
        return NULL;
}

void free_weight_statistics(struct weight_stats *ws)
{
        free(ws);
}

/*
 * Recursively collect weight statistics for all groups that project to the
 * current group.
 */
void collect_weight_statistics(struct weight_stats *ws, struct group *g)
{
 
        for (uint32_t i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *p = g->inc_projs->elements[i];
                struct matrix *w = p->weights;
                for (uint32_t r = 0; r < w->rows; r++) {
                        for (uint32_t c = 0; c < w->cols; c++) {
                                ws->num_weights++;
                                ws->cost += pow(w->elements[r][c], 2.0);
                                ws->mean += w->elements[r][c];
                                ws->mean_abs += fabs(w->elements[r][c]);
                                if (w->elements[r][c] < ws->minimum)
                                        ws->minimum = w->elements[r][c];
                                if (w->elements[r][c] > ws->maximum)
                                        ws->maximum = w->elements[r][c];
                        }
                }
                if (p->flags->recurrent)
                        continue;
                collect_weight_statistics(ws, p->to);
        }
}

/* 
 * Recursively collect mean dependent weight statistics for all groups that
 * project to the current group.
 */
void collect_mean_dependent_ws(struct weight_stats *ws, struct group *g)
{
        for (uint32_t i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *p = g->inc_projs->elements[i];
                struct matrix *w = p->weights;
                for (uint32_t r = 0; r < w->rows; r++) {
                        for (uint32_t c = 0; c < w->cols; c++) {
                                ws->mean_dist +=
                                        fabs(w->elements[r][c] - ws->mean);
                                ws->variance +=
                                        pow(w->elements[r][c] - ws->mean, 2.0);
                        }
                }
                if (p->flags->recurrent)
                        continue;
                collect_mean_dependent_ws(ws, p->to);
        }
}

void print_weight_statistics(struct network *n)
{
        struct weight_stats *ws = create_weight_statistics(n);
        cprintf("\n");
        cprintf("Weight statistics for network '%s'\n", n->name);
        cprintf("\n");
        cprintf("Number of weights: \t %d\n", ws->num_weights);
        cprintf("Cost: \t\t\t %f\n", ws->cost);
        cprintf("Mean: \t\t\t %f\n", ws->mean);
        cprintf("Absolute mean: \t\t %f\n", ws->mean_abs);
        cprintf("Mean dist.: \t\t %f\n", ws->mean_dist);
        cprintf("Variance: \t\t %f\n", ws->variance);
        cprintf("Minimum: \t\t %f\n", ws->minimum);
        cprintf("Maximum: \t\t %f\n", ws->maximum);
        cprintf("\n");
        free_weight_statistics(ws);
}
