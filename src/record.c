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

#include <signal.h>
#include <stdbool.h>
#include <stdio.h>

#include "act.h"
#include "main.h"
#include "record.h"

static bool keep_running = true;

                /**********************
                 **** record units ****
                 **********************/

void record_units(struct network *n, struct group *g, char *filename)
{
        struct sigaction sa;
        sa.sa_handler = recording_signal_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        sigaction(SIGINT, &sa, NULL);

        keep_running = true;

        FILE *fd;
        if (!(fd = fopen(filename, "w")))
                goto error_out;

        fprintf(fd, "\"ItemId\",\"ItemName\",\"ItemMeta\",\"EventNum\",\"Group\"");
        for (uint32_t u = 0; u < g->vector->size; u++)
                fprintf(fd, ",\"Unit%d\"", u + 1);
        fprintf(fd, "\n");

        switch (n->flags->type) {
        case ntype_ffn: /* fall through */
        case ntype_srn:
                record_ffn_units(n, g, fd);
                break;
        case ntype_rnn:
                record_rnn_units(n, g, fd);
                break;
        }

        fclose(fd);
        sa.sa_handler = SIG_DFL;
        sigaction(SIGINT, &sa, NULL);

        return;

error_out:
        perror("[record_units()]");
        return;
}

void record_ffn_units(struct network *n, struct group *g, FILE *fd)
{
        /* record units for all events of all items in the current set */
        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                if (!keep_running) return;
                struct item *item = n->asp->items->elements[i];

                if (n->flags->type == ntype_srn)
                        reset_context_groups(n);
                for (uint32_t j = 0; j < item->num_events; j++) {
                        if (j > 0 && n->flags->type == ntype_srn)
                                shift_context_groups(n);
                        copy_vector(n->input->vector, item->inputs[j]);
                        feed_forward(n, n->input);
                        
                        fprintf(fd, "%d,\"%s\",\"%s\",%d,%s",
                                i + 1, item->name, item->meta, j + 1, g->name);
                        for (uint32_t u = 0; u < g->vector->size; u++)
                                fprintf(fd, ",%f", g->vector->elements[u]);
                        fprintf(fd, "\n");
                }
                pprintf("%d: %s\n", i + 1, item->name);
        } 
}

/*
 * TODO: Test this!
 */
void record_rnn_units(struct network *n, struct group *g, FILE *fd)
{
        struct rnn_unfolded_network *un = n->unfolded_net;

        /* record units for all events of all items in the current set */
        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                if (!keep_running) return;
                struct item *item = n->asp->items->elements[i];

                reset_recurrent_groups(un->stack[un->sp]);
                for (uint32_t j = 0; j < item->num_events; j++) {
                        copy_vector(
                                un->stack[un->sp]->input->vector,
                                item->inputs[j]);
                        feed_forward(
                                un->stack[un->sp],
                                un->stack[un->sp]->input);
                        
                        /* XXX: Is this correct? */
                        struct group *rg = find_array_element_by_name(
                                un->stack[un->sp]->groups, g->name);
                        fprintf(fd, "%d,\"%s\",\"%s\",%d,%s",
                                i + 1, item->name, item->meta, j + 1, rg->name);
                        for (uint32_t u = 0; u < rg->vector->size; u++)
                                fprintf(fd, ",%f", rg->vector->elements[u]);
                        fprintf(fd, "\n");

                        shift_pointer_or_stack(n);
                }
        }
}

void recording_signal_handler(int32_t signal)
{
        cprintf("Recording interrupted. Abort [y/n]");
        int32_t c = getc(stdin);
        getc(stdin); /* get newline */
        if (c == 'y' || c == 'Y')
                keep_running = false;
}
