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

#include <signal.h>
#include <stdbool.h>
#include <stdio.h>

#include "engine.h"
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

        FILE *fd;
        if (!(fd = fopen(filename, "w")))
                goto error_out;
        fprintf(fd, "\"ItemId\",\"ItemName\",\"ItemMeta\",\"EventNum\",\"Group\"");
        for (uint32_t u = 0; u < g->vector->size; u++)
                fprintf(fd, ",\"Unit%d\"", u + 1);
        fprintf(fd, "\n");
        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                if (!keep_running)
                        return;
                struct item *item = n->asp->items->elements[i];
                reset_ticks(n);
                for (uint32_t j = 0; j < item->num_events; j++) {
                        if (j > 0)
                                next_tick(n);
                        clamp_input_vector(n, item->inputs[j]);
                        forward_sweep(n);
                        fprintf(fd, "%d,\"%s\",\"%s\",%d,%s",
                                i + 1, item->name, item->meta, j + 1, g->name);
                        for (uint32_t u = 0; u < g->vector->size; u++)
                                fprintf(fd, ",%f", g->vector->elements[u]);
                        fprintf(fd, "\n");                        
                }
                pprintf("%d: %s\n", i + 1, item->name);
        }

        fclose(fd);
        sa.sa_handler = SIG_DFL;
        sigaction(SIGINT, &sa, NULL);

        return;

error_out:
        perror("[record_units()]");
        return;
}

void recording_signal_handler(int32_t signal)
{
        cprintf("Recording interrupted. Abort [y/n]");
        int32_t c = getc(stdin);
        getc(stdin); /* get newline */
        if (c == 'y' || c == 'Y')
                keep_running = false;
}
