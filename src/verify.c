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

#include "main.h"
#include "verify.h"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
This implements various check to verify if a network architecture is sane.

TODO: Add verification check.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

bool verify_network(struct network *n)
{
        if (!n->input) {
                eprintf("Network has no input group\n");
                return false;
        }
        if (!n->output) {
                eprintf("Network has no output group\n");
                return false;
        }
        if(!verify_input_to_output_path(n, n->input)) {
                eprintf("No pathway from input group to output group\n");
                return false;
        }
        return true;
}

bool verify_input_to_output_path(struct network *n, struct group *g)
{
        bool reachable = false;
        for (uint32_t i = 0; i < g->out_projs->num_elements; i++) {
                struct projection *p = g->out_projs->elements[i];
                if (p->to == n->output)
                        return true;
                if (p->flags->recurrent)
                        continue;
                reachable = verify_input_to_output_path(n, p->to);
        }
        return reachable;
}
