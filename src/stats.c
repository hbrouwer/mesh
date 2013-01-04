/*
 * stats.c
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

#include <math.h>

#include "math.h"
#include "matrix.h"
#include "stats.h"

/*
 * ########################################################################
 * ## Weight statistics                                                  ##
 * ########################################################################
 */

/*
 * Compute weight statistics.
 */ 

struct weight_stats *create_weight_statistics(struct network *n)
{
        struct weight_stats *ws;
        if (!(ws = malloc(sizeof(struct weight_stats))))
                goto error_out;
        memset(ws, 0, sizeof(struct weight_stats));

        /* collect weight statistics */
        collect_weight_statistics(ws, n->output);

        /* compute means */
        ws->mean /= ws->num_weights;
        ws->mean_abs /= ws->num_weights;

        /* collect mean dependent statistics */
        collect_mean_dependent_ws(ws, n->output);

        /* compute mean dependent measures */
        ws->mean_dist /= ws->num_weights;
        ws->variance /= (ws->num_weights - 1);

        return ws;

error_out:
        perror("[weight_statistics()]");
        return NULL;
}

/*
 * Dispose weight statistics.
 */

void dispose_weight_statistics(struct weight_stats *ws)
{
        free(ws);
}

/*
 * Collect weight statistics.
 */

void collect_weight_statistics(struct weight_stats *ws, struct group *g)
{
        /*
         * Recursively collect weight statistics for all groups that
         * project to the current group.
         */ 
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *p = g->inc_projs->elements[i];
                struct matrix *w = p->weights;

                for (int r = 0; r < w->rows; r++) {
                        for (int c = 0; c < w->cols; c++) {
                                /* number of weights */
                                ws->num_weights++;

                                /* means */
                                ws->mean += w->elements[r][c];
                                ws->mean_abs += fabs(w->elements[r][c]);

                                /* minimum */
                                if (w->elements[r][c] < ws->minimum)
                                        ws->minimum = w->elements[r][c];

                                /* maximum */
                                if (w->elements[r][c] > ws->maximum)
                                        ws->maximum = w->elements[r][c];
                        }
                }

                /*
                 * Collect weight statistics for all groups
                 * that project to this group.
                 */
                collect_weight_statistics(ws, p->to);
        }
}

/*
 * Collect mean dependent weight statistics.
 */

void collect_mean_dependent_ws(struct weight_stats *ws, struct group *g)
{
        /* 
         * Recursively collect mean dependent weight statistics for all
         * groups that project to the current group.
         */
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *p = g->inc_projs->elements[i];
                struct matrix *w = p->weights;

                for (int r = 0; r < w->rows; r++) {
                        for (int c = 0; c < w->cols; c++) {
                                /* mean distance */
                                ws->mean_dist += fabs(w->elements[r][c] - ws->mean);

                                /* variance */
                                ws->variance += square(w->elements[r][c] - ws->mean);
                        }
                }

                /*
                 * Collect mean dependent weight statistics for all
                 * groups that project to this group.
                 */
                collect_mean_dependent_ws(ws, p->to);
        }
}