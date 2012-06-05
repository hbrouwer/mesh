/*
 * stats.c
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

#include <math.h>

#include "matrix.h"
#include "stats.h"

struct weight_stats *gather_weight_stats(struct network *n)
{
        struct weight_stats *ws;

        if (!(ws = malloc(sizeof(struct weight_stats))))
                goto error_out;
        memset(ws, 0, sizeof(struct weight_stats));

        gather_proj_weight_stats(ws, n->output);

        ws->mean = ws->mean / ws->num_weights;
        ws->mean_abs = ws->mean_abs / ws->num_weights;

        gather_proj_weight_md_stats(ws, n->output);

        ws->mean_dist = ws->mean_dist / ws->num_weights;
        ws->variance = ws->variance / (ws->num_weights - 1); /* <- sample variance */

        /*
        if (ws->num_weights > 1) {
                ws->mean_dist =
                        ws->mean_dist / ws->num_weights;
                ws->variance =
                        ws->variance / (ws->num_weights - 1);
        } else {
                ws->variance = 0.0;
        }
        */

        return ws;

error_out:
        perror("[gather_weight_stats()]");
        return NULL;
}

void gather_proj_weight_stats(struct weight_stats *ws, struct group *g)
{
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *p = g->inc_projs->elements[i];
                struct matrix *m = p->weights;
                for (int r = 0; r < m->rows; r++) {
                        for (int c = 0; c < m->cols; c++) {
                                ws->num_weights++;
                                
                                ws->mean += m->elements[r][c];
                                ws->mean_abs += fabs(m->elements[r][c]);
                                
                                if (m->elements[r][c] < ws->minimum)
                                        ws->minimum = m->elements[r][c];
                                if (m->elements[r][c] > ws->maximum)
                                        ws->maximum = m->elements[r][c];

                        }
                }

                gather_proj_weight_stats(ws, g->inc_projs->elements[i]->to);
        }
}

void gather_proj_weight_md_stats(struct weight_stats *ws, struct group *g)
{
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *p = g->inc_projs->elements[i];
                struct matrix *m = p->weights;
                for (int r = 0; r < m->rows; r++) {
                        for (int c = 0; c < m->cols; c++) {
                                ws->mean_dist +=
                                        fabs(m->elements[r][c] - ws->mean);
                                ws->variance +=
                                        pow(m->elements[r][c] - ws->mean, 2.0);
                        }
                }

                gather_proj_weight_md_stats(ws, g->inc_projs->elements[i]->to);
        }
}
