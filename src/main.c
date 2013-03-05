/*
 * main.c
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

#include <stdarg.h>

#include "cli.h"
#include "cmd.h"
#include "engine.h"
#include "main.h"
#include "network.h"
#include "rnn_unfold.h"
#include "session.h"

#define VERSION "[Mon Jan 28 11:52:14 CET 2013]"

int main(int argc, char **argv)
{
        struct session *s;

        cprintf("MESH version %s", VERSION);
        cprintf("Copyright (c) 2012, 2013 Harm Brouwer <me@hbrouwer.eu>");
        cprintf("Center for Language and Cognition, University of Groningen");
        cprintf("& Netherlands Organisation for Scientific Research (NWO)");

        s = create_session();

        for (int i = 1; i < argc; i++) {
                /* last argument is a network file */
                if (argv[argc - 1] != NULL) {
                        char *cmd;
                        asprintf(&cmd, "loadFile %s", argv[i]);
                        process_command(cmd, s);
                        free(cmd);
                }
        }

        cli_loop(s);
        dispose_session(s);

        exit(EXIT_SUCCESS);
}

void print_help(char *exec_name)
{
        cprintf(
                        "usage: %s [options]\n\n"

                        "  running network simulations:\n"
                        "    --network <file>\t\tload and test the network specified in <file>\n"
                        "    --save_weights <file>\tsave weight matrices to <file> after training\n"
                        "    --load_weights <file>\tload weight matrices from <file>\n"

                        "\n"
                        "  basic information for users:\n"
                        "    --help\t\t\tshows this help message\n"
                        "    --version\t\t\tshows version\n",
                        exec_name);
}

void print_version()
{
        cprintf("%s\n", VERSION);
}

/* print console message */
void cprintf(const char *fmt, ...)
{
        va_list args;
        // fprintf(stderr, "\x1b[38;05;14m");
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);
        fprintf(stderr, "\n");
        // fprintf(stderr, "\x1b[0m");
}

/* print program message */
void mprintf(const char *fmt, ...)
{
        va_list args;

        fprintf(stderr, "--- ");
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);
        fprintf(stderr, "\n");
}

/* print error message */
void eprintf(const char *fmt, ...)
{
        va_list args;

        fprintf(stderr, "\x1b[38;05;1m");
        fprintf(stderr, "!!! ERROR: ");
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);
        fprintf(stderr, "\n");
        fprintf(stderr, "\x1b[0m");
}

/* print progress message */
void pprintf(const char *fmt, ...)
{
        va_list args;

        fprintf(stderr, "=== ");
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);
        fprintf(stderr, "\n");
}

/* print report message */
void rprintf(const char *fmt, ...)
{
        va_list args;

        va_start(args, fmt);
        vfprintf(stdout, fmt, args);
        va_end(args);
        fprintf(stdout, "\n");
}
