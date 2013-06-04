/*
 * bp.h
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

#ifndef BP_H
#define BP_H

#define SD_DEFAULT  0
#define SD_BOUNDED  1

#define RPROP_PLUS    0
#define RPROP_MINUS   1
#define IRPROP_PLUS   2
#define IRPROP_MINUS  3

#include "network.h"

void bp_output_error(struct group *g, struct vector *t, double tr, double zr);

void bp_backpropagate_error(struct network *n, struct group *g);

void bp_update_sd(struct network *n);
void bp_recursively_update_sd(struct network *n, struct group *g);
void bp_update_projection_sd(struct network *n, struct group *g,
                struct projection *p);

void bp_determine_sd_sf(struct network *n);
void bp_recursively_determine_sd_sf(struct network *n, struct group *g);

void bp_update_rprop(struct network *n);
void bp_recursively_update_rprop(struct network *n, struct group *g);
void bp_update_projection_rprop(struct network *n, struct group *g,
                struct projection *p);

void bp_update_qprop(struct network *n);
void bp_recursively_update_qprop(struct network *n, struct group *g);
void bp_update_projection_qprop(struct network *n, struct group *g,
                struct projection *p);

void bp_update_dbd(struct network *n);
void bp_recursively_update_dbd(struct network *n, struct group *g);
void bp_update_projection_dbd(struct network *n, struct group *g,
                struct projection *p);

#endif /* BP_H */
