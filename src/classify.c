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
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>

#include "classify.h"
#include "engine.h"
#include "main.h"

static bool keep_running = true;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Construct a confusion matrix for classification tasks. The rows of this
matrix will be the 'actual' classes, and the columns the 'predicted' ones:

                           predicted:
                   |   A   |   B   |   C
                ----------------------------
                A  |  18   |   2   |   3   | 23
                ----------------------------
        actual: B  |   9   |  22   |   0   | 31
                ----------------------------
                C  |   0   |   1   |  10   | 11
                ----------------------------
                      27      25      13     65
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

struct matrix *confusion_matrix(struct network *n)
{
        struct sigaction sa;
        sa.sa_handler = cm_signal_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        sigaction(SIGINT, &sa, NULL);

        keep_running = true;

        uint32_t d = n->output->vector->size;
        struct matrix *cm = create_matrix(d, d);
        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                if (!keep_running)
                        goto out;
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
                        struct vector *tv = item->targets[j];
                        uint32_t t = 0, o = 0;
                        for (uint32_t x = 0; x < ov->size; x++) {
                                if (tv->elements[x] > tv->elements[t]) t = x;
                                if (ov->elements[x] > ov->elements[o]) o = x;
                        }
                        cm->elements[t][o]++;
                }
        }

        sa.sa_handler = SIG_DFL;
        sigaction(SIGINT, &sa, NULL);

out:
        return cm;        
}

void print_cm_summary(struct network *n, bool print_cm, bool pprint,
        enum color_scheme scheme)
{       
        struct matrix *cm = confusion_matrix(n);

        if (print_cm) {
                cprintf("\nConfusion matrix (actual x predicted):\n\n");
                pprint ? pprint_matrix(cm, scheme) : print_matrix(cm);
        }

        /* compute row and column totals */
        struct vector *rows = create_vector(cm->rows);
        struct vector *cols = create_vector(cm->cols);
        for (uint32_t r = 0; r < cm->rows; r++) {
                for (uint32_t c = 0; c < cm->cols; c++) {
                        rows->elements[r] += cm->elements[r][c];
                        cols->elements[c] += cm->elements[r][c];
                }
        }

        double num_correct   = 0.0;
        double num_incorrect = 0.0;
        double precision     = 0.0;
        double recall        = 0.0;

        /*
         * precision = #correct / column total
         *              
         * recall = #correct / row total
         */
        for (uint32_t r = 0; r < cm->rows; r++) {
                for (uint32_t c = 0; c < cm->cols; c++) {
                        /* incorrectly classified */
                        if (r != c) {
                                num_incorrect += cm->elements[r][c];
                                continue;
                        }
                        /* correctly classified */
                        num_correct += cm->elements[r][c];
                        if (cols->elements[c] > 0) 
                                precision += cm->elements[r][c]
                                        / cols->elements[c];
                        if (rows->elements[r] > 0)
                                recall += cm->elements[r][c]
                                        / rows->elements[r];
                }
        }
        precision /= cols->size;
        recall    /= rows->size;

        /*
         *            precision * recall
         * F(1) = 2 * ------------------
         *            precision + recall
         */
        double fscore = 2.0 * ((precision * recall) / (precision + recall));

        /*     
         *                   #correct
         * accuracy = ---------------------
         *            #correct + #incorrect
         */
        double accuracy = num_correct / (num_correct + num_incorrect);
        
        /*
         *                   #incorrect
         * error rate = -------------------
         *              #correct + #incorrect
         */
        double error_rate = num_incorrect / (num_correct + num_incorrect);

        cprintf("\n");
        cprintf("Classification statistics:\n");
        cprintf("\n");
        cprintf("Accurracy: \t %f\n",   accuracy);
        cprintf("Error rate: \t %f\n",  error_rate);
        cprintf("Precision: \t %f\n",   precision);
        cprintf("Recall: \t %f\n",      recall);
        cprintf("F(1)-score: \t %f\n",  fscore);
        cprintf("\n");
        
        free_vector(rows);
        free_vector(cols);
        free_matrix(cm);
}

void cm_signal_handler(int32_t signal)
{
        cprintf("Confusion matrix computation interrupted. Abort [y/n]");
        int32_t c = getc(stdin);
        getc(stdin); /* get newline */
        if (c == 'y' || c == 'Y')
                keep_running = false;
}
