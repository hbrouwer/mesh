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

#include "main.h"
#include "verify.h"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
This implements various check to verify if a network architecture is sane.

TODO: Add more verification checks.
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
        if (!verify_projection_path(n->input, n->output)) {
                eprintf("No projection path from input group to output group\n");
                return false;
        }
        verify_group_connectivity(n);
        return true;
}

bool verify_projection_path(struct group *fg, struct group *tg)
{
        if (fg == tg)
                return true;
        bool reachable = false;
        for (uint32_t i = 0; i < fg->out_projs->num_elements; i++) {
                struct projection *p = fg->out_projs->elements[i];
                if (p->to == tg)
                        return true;
                if (p->flags->recurrent)
                        continue;
                reachable = verify_projection_path(p->to, tg);
                if (reachable)
                        break;
        }
        return reachable;
}

/*
 * Warn if there is a group that is not connected to the network. 
 */
bool verify_group_connectivity(struct network *n)
{
        for (uint32_t i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];
                if (!(verify_projection_path(g, n->output)))
                        eprintf("WARNING: %s is not connected to the network\n",
                                g->name);
        }
        return true;
}
