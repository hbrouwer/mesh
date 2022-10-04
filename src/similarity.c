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

#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>

#include "engine.h"
#include "main.h"
#include "similarity.h"

static bool keep_running = true;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Construct a similarity matrix. The rows of this matrix represent the output
vectors for each item the active set, the columns the target vector for each
item, and the cells the similarity between each output and target vector:

                           target:                        
                   |   A   |   B   |   C
                ----------------------------
                A  |  .99  |  .26  |  .30  |
                ----------------------------
        output: B  |  .31  |  .97  |  .12  |
                ----------------------------
                C  |  .44  |  .15  |  .98  |
                ----------------------------
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

struct matrix *similarity_matrix(struct network *n)
{
        struct sigaction sa;
        sa.sa_handler = sm_signal_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        sigaction(SIGINT, &sa, NULL);

        keep_running = true;   
        
        uint32_t d = n->asp->items->num_elements;
        struct matrix *sm = create_matrix(d, d);
        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                if (!keep_running) {
                        keep_running = true;
                        goto out;
                }
                struct item *item = n->asp->items->elements[i];
                reset_ticks(n);
                for (uint32_t j = 0; j < item->num_events; j++) {
                        if (j > 0)
                                next_tick(n);
                        clamp_input_vector(n, item->inputs[j]);
                        forward_sweep(n);
                        if (!(item->targets[j] && j == item->num_events - 1))
                                continue;
                        struct vector *ov = output_vector(n);
                        for (uint32_t x = 0; x < n->asp->items->num_elements; x++) {
                                struct item *ci   = n->asp->items->elements[x];
                                struct vector *tv = ci->targets[ci->num_events - 1];
                                if (!tv)
                                        continue;
                                sm->elements[i][x] = n->similarity_metric(ov, tv);
                        }
                }
        }

out:
        sa.sa_handler = SIG_DFL;
        sigaction(SIGINT, &sa, NULL);

        return sm;
}

void print_sm_summary(struct network *n, bool print_sm, bool pprint,
        enum color_scheme scheme)
{
        struct matrix *sm = similarity_matrix(n);
        
        if (print_sm) {
                cprintf("\nOutput-target similarity matrix:\n\n");
                pprint ? pprint_matrix(sm, scheme) : print_matrix(sm);
        }

        /*
         * Compute mean similarity, and its standard deviation. Also,
         * determine how many items reached threshold.
         */
        uint32_t tr = n->asp->items->num_elements;
        double sim_mean = 0.0, sim_sd = 0.0;
        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                double s = sm->elements[i][i];
                for (uint32_t x = 0; x < n->asp->items->num_elements; x++) {
                        if (!keep_running)
                                return;
                        if (sm->elements[i][x] > s) {
                                tr--; /* note: we count down */
                                break;
                        }
                }
                sim_mean += s;
        }
        sim_mean /= n->asp->items->num_elements;
        for (uint32_t i = 0; i < sm->rows; i++)
                sim_sd += pow(sm->elements[i][i] - sim_mean, 2.0);
        sim_sd = sqrt(sim_sd / n->asp->items->num_elements);

        cprintf("\n");
        cprintf("Similarity statistics:\n");
        cprintf("\n");
        cprintf("Number of items: \t\t %d\n",
                        n->asp->items->num_elements);
        cprintf("Mean similarity: \t\t %lf\n",
                        sim_mean);
        cprintf("SD of similarity:\t\t %lf\n",
                        sim_sd);
        cprintf("# Items reached threshold: \t %d (%.2lf%%)\n",
                        tr, ((double)tr / n->asp->items->num_elements) * 100.0);
        cprintf("\n");
        
        free_matrix(sm);
}

void sm_signal_handler(int32_t signal)
{
        cprintf("(interrupted): Abort [y/n]? ");
        int32_t c = getc(stdin);
        getc(stdin); /* get newline */
        if (c == 'y' || c == 'Y')
                keep_running = false;
}
