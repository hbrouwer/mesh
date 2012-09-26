/*
 * bp.c
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

#include "bp.h"

/*
 * This provides an implementation of the backpropagation (BP) algorithm.
 *
 *
 */

void bp_backpropagate_error(struct network *n, struct group *g,
                struct vector *e)
{
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct projection *p = g->inc_projs->elements[i];
                zero_out_vector(p->error);
                bp_projection_deltas_and_error(n, p, e);
        }

        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                struct group *ng = g->inc_projs->elements[i]->to;
                struct vector *ge = bp_sum_group_error(n, ng);
                bp_backpropagate_error(n, ng, ge);
                dispose_vector(ge);
        }
}

void bp_projection_deltas_and_error(struct network *n, struct projection *p,
                struct vector *e)
{
        for (int i = 0; i < p->to->vector->size; i++) {
                for (int j = 0; j < e->size; j++) {
                        p->error->elements[i] += p->weights->elements[i][j]
                                * e->elements[j];
                        p->deltas->elements[i][j] += p->to->vector->elements[i]
                                * e->elements[j];
                }
        }        
}

struct vector *bp_sum_group_error(struct network *n, struct group *g)
{
        struct vector *e = create_vector(g->vector->size);

        for (int i = 0; i < g->vector->size; i++) {
                for (int j = 0; j < g->out_projs->num_elements; j++) {
                        struct projection *p = g->out_projs->elements[j];
                        e->elements[i] += p->error->elements[i];
                }

                double act_deriv;
                if (g != n->input)
                        act_deriv = g->act->deriv(g->vector, i);
                else 
                        act_deriv = g->vector->elements[i];

                e->elements[i] *= act_deriv;
        }

        return e;
}

void bp_adjust_weights(struct network *n, struct group *g)
{
        for (int i = 0; i < g->inc_projs->num_elements; i++) {
                bp_adjust_projection_weights(n, g, g->inc_projs->elements[i]);
                if (!g->inc_projs->elements[i]->recurrent)
                        bp_adjust_weights(n, g->inc_projs->elements[i]->to);
        }        
}

void bp_adjust_projection_weights(struct network *n, struct group *g,
                struct projection *p)
{
        for (int i = 0; i < p->to->vector->size; i++)
                for (int j = 0; j < g->vector->size; j++)
                        p->weights->elements[i][j] += 
                                n->learning_rate
                                * p->deltas->elements[i][j]
                                - n->weight_decay
                                * p->prev_deltas->elements[i][j]
                                + n->momentum
                                * p->prev_deltas->elements[i][j];
        
        copy_matrix(p->prev_deltas, p->deltas);
        zero_out_matrix(p->deltas);        
}
