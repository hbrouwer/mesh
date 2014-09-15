/*
 * erp.c
 *
 * Copyright 2012-2014 Harm Brouwer <me@hbrouwer.eu>
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

#include "erp.h"

#include "../act.h"
#include "../math.h"
#include "../vector.h"

/**************************************************************************
 *************************************************************************/
void erp_generate_table(struct network *n, struct group *n400_gen,
                struct group *p600_gen, char *filename)
{
        FILE *fd;
        if (!(fd = fopen(filename, "w")))
                goto error_out;

        fprintf(fd,"item_id,item_name,item_meta,word_pos,n400_amp,p600_amp\n");

        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                struct item *item = n->asp->items->elements[i];

                /* ERP amplitudes */
                struct vector *n4av = erp_amplitudes_for_item(n, n400_gen, item);
                struct vector *p6av = erp_amplitudes_for_item(n, p600_gen, item);

                for (uint32_t j = 0; j < item->num_events; j++) 
                        fprintf(fd,"%d,\"%s\",\"%s\",%d,%f,%f\n",
                                        i, item->name, item->meta, j,
                                        n4av->elements[j], p6av->elements[j]);

                dispose_vector(n4av);
                dispose_vector(p6av);
        }

        fclose(fd);

        return;

error_out:
        perror("[erp_generate_table()]");
        return;
}

/**************************************************************************
 *************************************************************************/
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
