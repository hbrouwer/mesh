/*
 * cmd.h
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

#ifndef CMD_H
#define CMD_H

#include "network.h"
#include "session.h"

void process_command(char *cmd, struct session *s);

void cmd_quit(char *cmd, char *fmt, struct session *s, char *msg);

bool cmd_create_network(char *cmd, char *fmt, struct session *s, char *msg);
bool cmd_load_network(char *cmd, char *fmt, struct session *s, char *msg);
bool cmd_dispose_network(char *cmd, char *fmt, struct session *s, char *msg);

bool cmd_create_group(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_dispose_group(char *cmd, char *fmt, struct network *n, char *msg); 
bool cmd_attach_bias(char *cmd, char *fmt, struct network *n, char *msg);

bool cmd_set_input_group(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_set_output_group(char *cmd, char *fmt, struct network *n, char *msg);

bool cmd_create_projection(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_create_elman_projection(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_dispose_projection(char *cmd, char *fmt, struct network *n, char *msg);
bool cmd_freeze_projection(char *cmd, char *fmt, struct network *n, char *msg);

bool cmd_set_double_parameter(char *cmd, char *fmt, double *par, char *msg);
bool cmd_set_int_parameter(char *cmd, char *fmt, int *par, char *msg);

bool cmd_load_item_set(char *cmd, char *fmt, struct network *n, bool train,
                char *msg);

bool cmd_set_training_order(char *cmd, char *fmt, int *training_order,
                char *msg);

bool cmd_set_learning_algorithm(char *cmd, char *fmt, struct network *n,
                char *msg);
bool cmd_set_update_algorithm(char *cmd, char *fmt, struct network *n,
                char *msg);

bool cmd_train(char *cmd, char *fmt, struct network *n);
bool cmd_test(char *cmd, char *fmt, struct network *n);
bool cmd_erps(char *cmd, char *fmt, struct network *n);

#endif /* CMD_H */