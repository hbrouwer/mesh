/*
 * erps.c
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

#include "erps.h"

#include "../act.h"
#include "../math.h"

#include <math.h>

void erp_generate_table(struct network *n, char *fn)
{
        FILE *fd = fopen(fn, "w");

        fprintf(fd,"item_id,item_name,item_meta,word_pos,n400_amp,p600_amp,mtg_bold,ifg_bold\n");

        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                struct item *item = n->asp->items->elements[i];

                /* ERP amplitudes */
                struct vector *n4av = erp_amplitudes_for_item(n,
                                find_array_element_by_name(n->groups, "lpMTG_hidden"), item);
                struct vector *p6av = erp_amplitudes_for_item(n,
                                find_array_element_by_name(n->groups, "lIFG_hidden"), item);

                /* BOLD responses */
                struct vector *mtg_bv = bold_responses_for_item(n,
                                find_array_element_by_name(n->groups, "lpMTG_hidden"), item);
                struct vector *ifg_bv = bold_responses_for_item(n,
                                find_array_element_by_name(n->groups, "lIFG_hidden"), item);

                for (uint32_t j = 0; j < item->num_events; j++) 
                        fprintf(fd,"%d,\"%s\",\"%s\",%d,%f,%f,%f,%f\n",
                                        i, item->name, item->meta, j,
                                        n4av->elements[j],
                                        p6av->elements[j],
                                        mtg_bv->elements[j],
                                        ifg_bv->elements[j]);

                dispose_vector(n4av);
                dispose_vector(p6av);
        }

        fclose(fd);
}

struct vector *erp_amplitudes_for_item(struct network *n, struct group *g,
                struct item *item)
{
        struct vector *av = create_vector(item->num_events);
        struct vector *pv = create_vector(g->vector->size);
        fill_vector_with_value(pv, 0.5);

        if (n->type == TYPE_SRN)
                reset_context_groups(n);

        for (uint32_t i = 0; i < item->num_events; i++) {
                /*
                 * Shift context group chain, in case of 
                 * "Elman-towers".
                 */
                if (i > 0 && n->type == TYPE_SRN)
                        shift_context_groups(n);

                copy_vector(n->input->vector, item->inputs[i]);
                feed_forward(n, n->input);

                /* compute vector dissimilarity */
                av->elements[i] = 1.0 - n->similarity_metric(g->vector, pv);

                copy_vector(pv, g->vector);
        }

        dispose_vector(pv);

        return av;
}

struct vector *bold_responses_for_item(struct network *n, struct group *g,
                struct item *item)
{
        struct vector *bv = create_vector(item->num_events);
        //struct vector *pv = create_vector(g->vector->size);
        //fill_vector_with_value(pv, 0.5);

        if (n->type == TYPE_SRN)
                reset_context_groups(n);

        for (uint32_t i = 0; i < item->num_events; i++) {
                /*
                 * Shift context group chain, in case of 
                 * "Elman-towers".
                 */
                if (i > 0 && n->type == TYPE_SRN)
                        shift_context_groups(n);

                copy_vector(n->input->vector, item->inputs[i]);
                feed_forward(n, n->input);

                double lfp = 0.0;
                for (uint32_t j = 0; j < g->vector->size; j++) {
                        for (uint32_t x = 0; x < g->inc_projs->num_elements; x++) {
                                struct projection *ip = g->inc_projs->elements[x];
                                struct group *pg = ip->to;
                                struct matrix *w = ip->weights;
                                for (uint32_t z = 0; z < pg->vector->size; z++)
                                        lfp += pg->vector->elements[z] * w->elements[z][j];
                        }
                }
                lfp /= g->vector->size;

                /*
                double bold = 0.0;
                for (int x = 0; x < g->vector->size; x++)
                        bold += g->vector->elements[x];
                        */
                //bold /= g->vector->size;
                
                bv->elements[i] = lfp; 

                //copy_vector(pv, g->vector);
        }

        //dispose_vector(pv);

        return bv;
}
