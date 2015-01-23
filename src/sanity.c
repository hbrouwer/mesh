/*
 * sanity.c
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

#include "main.h"
#include "sanity.h"

/**************************************************************************
 *************************************************************************/
bool verify_network_sanity(struct network *n)
{
        if (!n->input) {
                eprintf("Network has no input group");
                return false;
        }
        if (!n->output) {
                eprintf("Network has no output group");
                return false;
        }

        if(!verify_input_to_output(n, n->input)) {
                eprintf("No pathway from input group to output group");
                return false;
        }

        return true;
}

/**************************************************************************
 *************************************************************************/
bool verify_input_to_output(struct network *n, struct group *g)
{
        bool reachable = false;

        for (uint32_t i = 0; i < g->out_projs->num_elements; i++) {
                struct projection *p = g->out_projs->elements[i];

                if (p->to == n->output)
                        return true;

                reachable = verify_input_to_output(n, p->to);
        }

        return reachable;
}
