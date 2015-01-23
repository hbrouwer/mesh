/*
 * sanity.h
 *
 * Copyright 2012-2015 Harm Brouwer <me@hbrouwer.eu>
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

#ifndef SANITY_H
#define SANITY_H

#include <stdbool.h>

#include "network.h"

/**************************************************************************
 *************************************************************************/
bool verify_network_sanity(struct network *n);
bool verify_input_to_output(struct network *n, struct group *g);

#endif /* SANITY_H */
