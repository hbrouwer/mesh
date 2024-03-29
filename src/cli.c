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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cli.h"
#include "cmd.h"
#include "main.h"

/* command line interface loop */
void cli_loop(struct session *s)
{
        char *line = NULL;
        size_t linecap = 0;

        do {
                /* prompt */
                cprintf("  [");
                if (s->anp) {
                        cprintf("%s:", s->anp->name);
                        if (s->anp->asp)
                                cprintf("%s", s->anp->asp->name);
                } else {
                        cprintf(":");
                }
                cprintf("> ");
                
                /* get a line */
                ssize_t num_chars;
                if ((num_chars = getline(&line, &linecap, stdin)) == -1)
                        goto error_out;
                line[num_chars - 1] = '\0';

                process_command(line, s);
                
                memset(line, 0, num_chars);
        } while (!feof(stdin));

error_out:
        perror("[cli_loop()]");
        return;
}
