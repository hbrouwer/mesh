/*
 * Copyright 2012-2022 Harm Brouwer <me@hbrouwer.eu>
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

#include <signal.h>
#include <stdlib.h>
#include <string.h>

#include "../engine.h"
#include "../main.h"
#include "../math.h"
#include "../matrix.h"
#include "../vector.h"

static bool keep_running = true;

                /**********************************
                 **** event-related potentials ****
                 **********************************/

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
This implements the estimation of ERP correlates, as described in:
 
Brouwer, H. (2014). The Electrophysiology of Language Comprehension: A
        Neurocomputational Model. PhD thesis, University of Groningen.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

void erp_contrast(struct network *n, struct group *gen,
        struct item *ctl, struct item *tgt)
{
        struct vector *cv = erp_values_for_item(n, gen, ctl);
        struct vector *tv = erp_values_for_item(n, gen, tgt);

        struct matrix *effects = create_matrix(cv->size, tv->size);
        for (uint32_t r = 0; r < effects->rows; r++)
                for (uint32_t c = 0; c < effects->cols; c++)
                        effects->elements[r][c] = tv->elements[c] - cv->elements[r];

        cprintf("\n");
        cprintf("Control: %s\n\n", ctl->name);
        print_vector(cv);
        cprintf("\n");
        cprintf("Target:  %s\n\n", tgt->name);
        print_vector(tv);
        cprintf("\n");
        cprintf("Effect matrix (control x target)\n");
        cprintf("(positive values indicate: target > control)\n\n");
        print_matrix(effects);
        cprintf("\n");

        free_matrix(effects);
        free_vector(cv);
        free_vector(tv);
}

void erp_write_values(struct network *n, struct group *N400_gen,
        struct group *P600_gen, char *filename)
{
        struct sigaction sa;
        sa.sa_handler = erp_signal_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        sigaction(SIGINT, &sa, NULL);

        FILE *fd;
        if (!(fd = fopen(filename, "w")))
                goto error_out;

        cprintf("\n");
        fprintf(fd,"\"ItemId\",\"ItemName\",\"ItemMeta\",\"WordPos\",\"N400\",\"P600\"\n");
        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                if (!keep_running) {
                        keep_running = true;
                        goto out;
                }
                struct item *item   = n->asp->items->elements[i];
                struct vector *N400 = erp_values_for_item(n, N400_gen, item);
                struct vector *P600 = erp_values_for_item(n, P600_gen, item);
                for (uint32_t j = 0; j < item->num_events; j++) 
                        fprintf(fd,"%d,\"%s\",\"%s\",%d,%f,%f\n",
                                i + 1, item->name, item->meta, j + 1,
                                N400->elements[j], P600->elements[j]);
                pprintf("%d: %s\n", i + 1, item->name);
                free_vector(N400);
                free_vector(P600);
        }
        cprintf("\n");

out:
        fclose(fd);

        sa.sa_handler = SIG_DFL;
        sigaction(SIGINT, &sa, NULL);

        return;

error_out:
        perror("[erp_amplitudes()]");
        return;
}

struct vector *erp_values_for_item(struct network *n, struct group *g,
        struct item *item)
{
        struct vector *ev = create_vector(item->num_events);

        /*
         * Previous activation vector for the specified group. At time-step
         * t=0, we bootstrap this using the unit vector v(1) / |v(1)|.
         */
        struct vector *pv = create_vector(g->vector->size);
        fill_vector_with_value(pv, 1.0);
        fill_vector_with_value(pv, 1.0 / euclidean_norm(pv));

        reset_ticks(n);
        for (uint32_t i = 0; i < item->num_events; i++) {
                if (i > 0)
                        next_tick(n);
                clamp_input_vector(n, item->inputs[i]);
                forward_sweep(n);
                /*
                 * amplitude = 1.0 - sim(g_t, g_{t-1})
                 */
                struct group *ng = find_network_group_by_name(n, g->name);
                ev->elements[i] =
                        1.0 - n->similarity_metric(ng->vector, pv);
                copy_vector(ng->vector, pv);
        }

        free_vector(pv);

        return ev;
}

void erp_signal_handler(int32_t signal)
{
        cprintf("(interrupted): Abort [y/n]? ");
        int32_t c = getc(stdin);
        getc(stdin); /* get newline */
        if (c == 'y' || c == 'Y')
                keep_running = false;
}
