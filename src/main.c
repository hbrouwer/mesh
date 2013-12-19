/*
 * main.c
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

#ifdef _OPENMP
#include <omp.h>
#endif /* OPENMP */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cli.h"
#include "cmd.h"
#include "main.h"
#include "session.h"

#define VERSION "0.99a"

/**************************************************************************
 *************************************************************************/
int main(int argc, char **argv)
{
        struct session *s;

        mprintf("MESH, version %s: http://hbrouwer.github.io/mesh/", VERSION);
#ifdef _OPENMP
        print_openmp_status();
#endif /* _OPENMP */

        s = create_session();

        for (uint32_t i = 1; i < argc; i++) {
                if (strcmp(argv[i],"--help") == 0) {
                        print_help(argv[0]);
                        goto leave_session;
                }
                else if (strcmp(argv[i],"--version") == 0) {
                        print_version();
                        goto leave_session;
                }
                else if (argv[i] != NULL) {
                        char *cmd;
                        asprintf(&cmd, "loadFile %s", argv[i]);
                        process_command(cmd, s);
                        free(cmd);
                }
        }

        cli_loop(s);

leave_session:
        dispose_session(s);

        exit(EXIT_SUCCESS);
}

/**************************************************************************
 *************************************************************************/
#ifdef _OPENMP
void print_openmp_status()
{
        mprintf("[+openmp: %d processors available]", omp_get_num_procs());

        omp_sched_t k;
        int m;
        omp_get_schedule(&k, &m);
        switch(k) {
                case 1:
                        mprintf("[+openmp: static schedule (chunk size: %d)]", m);
                        break;
                case 2:
                        mprintf("[+openmp: dynamic schedule (chunk size: %d)]", m);
                        break;
                case 3:
                        mprintf("[+openmp: guided schedule (chunk size: %d)]", m);
                        break;
                case 4:
                        mprintf("[+openmp: auto schedule]");
                        break;
        }
}
#endif /* _OPENMP */

/**************************************************************************
 *************************************************************************/
void print_help(char *exec_name)
{
        mprintf(
                        "Usage: %s [options-and-file]\n\n"

                        "  Basic information for users:\n"
                        "    --help\t\t\tShows this help message\n"
                        "    --version\t\t\tShows version\n",

                        exec_name);
}

void print_version()
{
        mprintf(
                        "\n"
                        "(c) 2013 Harm Brouwer <me@hbrouwer.eu>\n"
                        "Center for Language and Cognition Groningen (CLCG), University of Groningen\n"
                        "& Netherlands Organisation for Scientific Research (NWO)\n",
                        
                        VERSION);
        
}

/**************************************************************************
 * Print console message
 *************************************************************************/
void cprintf(const char *fmt, ...)
{
        va_list args;
        va_start(args, fmt);
        vfprintf(stdout, fmt, args);
        va_end(args);
}

/**************************************************************************
 * Print program message
 *************************************************************************/
void mprintf(const char *fmt, ...)
{
        va_list args;

        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);
        fprintf(stderr, "\n");
}

/**************************************************************************
 * Print error message
 *************************************************************************/
void eprintf(const char *fmt, ...)
{
        va_list args;

        fprintf(stderr, "\x1b[31m");
        fprintf(stderr, "ERROR: ");
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);
        fprintf(stderr, "\n");
        fprintf(stderr, "\x1b[0m");
}

/**************************************************************************
 * Print progress/report message
 *************************************************************************/
void pprintf(const char *fmt, ...)
{
        va_list args;

        fprintf(stdout, "**** ");
        va_start(args, fmt);
        vfprintf(stdout, fmt, args);
        va_end(args);
}
