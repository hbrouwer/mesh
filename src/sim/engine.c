/*
 * engine.c
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

#include "act.h"
#include "bp.h"
#include "engine.h"
#include "pprint.h"

/*
 * ########################################################################
 * ## Network training                                                   ##
 * ########################################################################
 */

void train_network(struct network *n)
{
        mprintf("starting training of network: [%s]", n->name);

        n->learning_algorithm(n);
}

/*
 * This function implements backpropagation (BP) training.
 */

void train_network_bp(struct network *n)
{
        for (int epoch = 1; epoch <= n->max_epochs; epoch++) {
                double me = 0.0;
                
                /* determine order of training items */
                struct set *training_set;
                if (n->training_order == TRAIN_ORDERED)
                        training_set = n->training_set;
                if (n->training_order == TRAIN_PERMUTED)
                        training_set = permute_set(n->training_set);
                if (n->training_order == TRAIN_RANDOMIZED)
                        training_set = randomize_set(n->training_set);

                /* present all training items */
                for (int i = 0; i < training_set->num_elements; i++) {
                        struct element *e = training_set->elements[i];

                        /* reset context groups */
                        if (n->srn)
                                reset_context_groups(n);

                        /* present all events for this item */
                        for (int j = 0; j < e->num_events; j++) {
                                copy_vector(n->input->vector, e->inputs[j]);
                                feed_forward(n, n->input);

                                /* inject error if a target is specified */
                                if (e->targets[j]) {
                                        copy_vector(n->target, e->targets[j]);

                                        /* backpropagate error */
                                        struct vector *error = bp_output_error(n);
                                        bp_backpropagate_error(n, n->output, error);
                                        dispose_vector(error);
                                        
                                        /* compute error */
                                        me += n->error->fun(n);
                                }
                        }
                }

                /* compute and report mean error */
                me /= training_set->num_elements;
                if (epoch == 1 || epoch % n->report_after == 0)
                        pprintf("epoch: [%d] | error: [%lf]", epoch, me);

                /* stop training if threshold is reached */
                if (me < n->error_threshold)
                        break;

                /* adjust weights */
                bp_adjust_weights(n, n->output);

                /* scale LR and Momentum */
                scale_learning_rate(epoch,n);
                scale_momentum(epoch,n);
        }
}

/*
 * This function implements backpropagation through time (BPTT) training.
 */

void train_network_bptt(struct network *n)
{
        struct ffn_unfolded_network *un = n->unfolded_net;
        struct network *nsp = un->stack[0];

        for (int epoch = 1; epoch <= n->max_epochs; epoch++) {
                double me = 0.0;
                int his = 0;

                /* determine order of training items */
                struct set *training_set;
                if (n->training_order == TRAIN_ORDERED)
                        training_set = n->training_set;
                if (n->training_order == TRAIN_PERMUTED)
                        training_set = permute_set(n->training_set);
                if (n->training_order == TRAIN_RANDOMIZED)
                        training_set = randomize_set(n->training_set);

                /* present all training items */
                for (int i = 0; i < training_set->num_elements; i++) {
                        struct element *e = training_set->elements[i];

                        /*
                         * reset recurrent groups
                         *
                         * XXX: Does this make sense when history length is
                         *   larger then the number of events in an item?
                         */
                        reset_recurrent_groups(nsp);

                        /* present all events for this item */
                        for (int j = 0; j < e->num_events; j++) {
                                /* cycle network stack if necessary */
                                if (his == un->stack_size) {
                                        ffn_cycle_stack(un);
                                        nsp = un->stack[--his];
                                } else {
                                        nsp = un->stack[his];
                                }

                                copy_vector(nsp->input->vector, e->inputs[j]);
                                if (e->targets[j])
                                        copy_vector(nsp->target, e->targets[j]);

                                feed_forward(nsp, nsp->input);

                                his++;
                        }

                        if (his == un->stack_size) {
                                /* backpropagate error */
                                struct vector *error = bp_output_error(nsp);
                                bp_backpropagate_error(nsp, nsp->output, error);
                                dispose_vector(error);

                                /* compute error */
                                me += n->error->fun(nsp);

                                /* sum deltas over unfolded network */
                                ffn_sum_deltas(un);

                                /* adjust weights */
                                bp_adjust_weights(un->stack[0], un->stack[0]->output);
                        }
                }

                /* compute and report mean error */
                me /= training_set->num_elements;
                if (epoch == 1 || epoch % n->report_after == 0)
                        pprintf("epoch: [%d] | error: [%lf]", epoch, me);

                /* stop training if threshold is reached */
                if (me < n->error_threshold)
                        break;

                /* scale LR and Momentum */
                scale_learning_rate(epoch,n);
                scale_momentum(epoch,n);
        }
}



/*
 * ########################################################################
 * ## Learning rate and Momentum scaling                                 ##
 * ########################################################################
 */

void scale_learning_rate(int epoch, struct network *n)
{
        int scale_after = n->lr_scale_after * n->max_epochs;
        if (scale_after > 0 && epoch % scale_after == 0) {
                double lr = n->learning_rate;
                n->learning_rate = n->lr_scale_factor * n->learning_rate;
                mprintf("scaled learning rate: [%lf --> %lf]",
                                lr, n->learning_rate);
        }
}

void scale_momentum(int epoch, struct network *n)
{
        int scale_after = n->mn_scale_after * n->max_epochs;
        if (scale_after > 0 && epoch % scale_after == 0) {
                double mn = n->momentum;
                n->momentum = n->mn_scale_factor * n->momentum;
                mprintf("scaled momentum: [%lf --> %lf]",
                                mn, n->momentum);
        }
}

/*
 * ########################################################################
 * ## Network testing                                                    ##
 * ########################################################################
 */

void test_network(struct network *n)
{
        mprintf("starting testing of network: [%s]", n->name);
        
        double me = 0.0;

        /* present all test items */
        for (int i = 0; i < n->test_set->num_elements; i++) {
                struct element *e = n->test_set->elements[i];
                
                /* reset context groups */
                if (n->srn)
                        reset_context_groups(n);

                /* present all events for this item */
                rprintf("\nI: \"%s\"", e->name);
                for (int j = 0; j < e->num_events; j++) {
                        copy_vector(n->input->vector, e->inputs[j]);
                        feed_forward(n, n->input);

                        if (e->targets[j] != NULL) {
                                copy_vector(n->target, e->targets[j]);
                                
                                /* compute error */
                                me += n->error->fun(n);

                                printf("T: ");
                                pprint_vector(n->target);
                                printf("O: ");
                                pprint_vector(n->output->vector);
                        }
                }
        }

        /* report error */
        me /= n->test_set->num_elements;
        pprintf("error: [%lf]", me);
}

void test_unfolded_network(struct network *n)
{
        mprintf("starting testing of network: [%s]", n->name);

        struct ffn_unfolded_network *un = n->unfolded_net;
        struct network *nsp = un->stack[0];

        double me = 0.0;
        int his = 0;

        /* present all test items */
        for (int i = 0; i < n->test_set->num_elements; i++) {
                struct element *e = n->test_set->elements[i];

                /*
                 * reset recurrent groups
                 *
                 * XXX: Does this make sense when history length is
                 *   larger then the number of events in an item?
                 */
                reset_recurrent_groups(nsp);

                /* present all events for this item */
                rprintf("\nI: \"%s\"", e->name);
                for (int j = 0; j < e->num_events; j++) {
                        /* cycle network stack if necessary */
                        if (his == un->stack_size) {
                                ffn_cycle_stack(un);
                                nsp = un->stack[--his];
                        } else {
                                nsp = un->stack[his];
                        }

                        copy_vector(nsp->input->vector, e->inputs[j]);
                        if (e->targets[j])
                                copy_vector(nsp->target, e->targets[j]);

                        feed_forward(nsp, nsp->input);

                        if (e->targets[j]) {
                                copy_vector(nsp->target, e->targets[j]);

                                /* compute error */
                                me += n->error->fun(nsp);

                                printf("T: ");
                                pprint_vector(nsp->target);
                                printf("O: ");
                                pprint_vector(nsp->output->vector);
                                
                        }
                        his++;
                }
        }

        /* report error */
        me /= n->test_set->num_elements;
        pprintf("error: [%lf]", me);
}
