/*
 * cmd.h
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

#ifndef CMD_H
#define CMD_H

#include <stdint.h>

#include "network.h"
#include "session.h"

/**************************************************************************
 *************************************************************************/
void process_command(char *cmd, struct session *s);

/**************************************************************************
 *************************************************************************/
void cmd_quit(char *cmd, char *fmt, struct session *s, char *msg);

/**************************************************************************
 *************************************************************************/
bool cmd_load_file(char *cmd, char *fmt, struct session *s, char *msg);

/**************************************************************************
 *************************************************************************/
bool cmd_create_network(char *cmd, char *fmt, struct session *s, char *msg);
bool cmd_dispose_network(char *cmd, char *fmt, struct session *s, char *msg);
bool cmd_list_networks(char *cmd, char *fmt, struct session *s, char *msg);
bool cmd_change_network(char *cmd, char *fmt, struct session *s, char *msg);

/**************************************************************************
 *************************************************************************/
bool cmd_create_group(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_dispose_group(char *cmd, char *fmt, struct network *n, char *msg); 
bool cmd_list_groups(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_attach_bias(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_set_input_group(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_set_output_group(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_set_act_func(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_set_err_func(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_toggle_act_lookup(char *cmd, char *fmt, struct network *n, char *msg);

/**************************************************************************
 *************************************************************************/
bool cmd_create_projection(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_dispose_projection(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_create_elman_projection(char *cmd, char *fmt, struct network *n,
                char *msg);
bool cmd_dispose_elman_projection(char *cmd, char *fmt, struct network *n,
                char *msg);
bool cmd_list_projections(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_freeze_projection(char *cmd, char *fmt, struct network *n, char *msg);

/**************************************************************************
 *************************************************************************/
bool cmd_set_double_parameter(char *cmd, char *fmt, double *par, char *msg);
bool cmd_set_int_parameter(char *cmd, char *fmt, uint32_t *par, char *msg);

/**************************************************************************
 *************************************************************************/
bool cmd_load_set(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_dispose_set(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_list_sets(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_change_set(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_list_items(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_set_training_order(char *cmd, char *fmt, uint32_t *training_order,
                char *msg);

/**************************************************************************
 *************************************************************************/
bool cmd_set_rand_algorithm(char *cmd, char *fmt, struct network *n,
                char *msg);
bool cmd_set_learning_algorithm(char *cmd, char *fmt, struct network *n,
                char *msg);
bool cmd_set_update_algorithm(char *cmd, char *fmt, struct network *n,
                char *msg);

/**************************************************************************
 *************************************************************************/
bool cmd_set_similarity_metric(char *cmd, char *fmt, struct network *n,
                char *msg);

/**************************************************************************
 *************************************************************************/
bool cmd_init(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_reset(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_train(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_test(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_test_item(char *cmd, char *fmt, struct session *s, char *msg);

/**************************************************************************
 *************************************************************************/
bool cmd_similarity_matrix(char *cmd, char *fmt, struct network *n,
                char *msg);
bool cmd_confusion_matrix(char *cmd, char *fmt, struct network *n,
                char *msg);

/**************************************************************************
 *************************************************************************/
bool cmd_weight_stats(char *cmd, char *fmt, struct network *n, char *msg);

/**************************************************************************
 *************************************************************************/
bool cmd_show_vector(char *cmd, char *fmt, struct session *s, char *msg,
                uint32_t type);
bool cmd_show_matrix(char *cmd, char *fmt, struct session *s, char *msg,
                uint32_t type);

/**************************************************************************
 *************************************************************************/
bool cmd_load_weights(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_save_weights(char *cmd, char *fmt, struct network *n, char *msg);

/**************************************************************************
 *************************************************************************/
bool cmd_toggle_pretty_printing(char *cmd, char *fmt, struct session *s, 
                char *msg);
bool cmd_set_colorscheme(char *cmd, char *fmt, struct session *s, char *msg);

/**************************************************************************
 *************************************************************************/
bool cmd_erp_generate_table(char *cmd, char *fmt, struct session *s, char *msg);

#endif /* CMD_H */
