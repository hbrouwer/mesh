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

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cli.h"
#include "cmd.h"
#include "help.h"
#include "main.h"
#include "math.h"
#include "session.h"

int main(int argc, char **argv)
{
        cprintf("Mesh, version %s: https://github.com/hbrouwer/mesh (`?` for help)\n", VERSION);
#ifdef FAST_EXP
        print_fast_exp_status();
#endif /* FAST_EXP */
#ifdef _OPENMP
        print_openmp_status();
#endif /* _OPENMP */

        struct session *s = create_session();
        for (uint32_t i = 1; i < argc; i++) {
                if (strcmp(argv[i],"--help") == 0) {
                        help("usage");
                        goto exit_session;
                }
                else if (strcmp(argv[i],"--version") == 0) {
                        goto exit_session;
                }
                else if (argv[i] != NULL) {
                        char cmd_base[] = "loadFile";
                        char *cmd;
                        size_t block_size = (strlen(cmd_base) + 1 + strlen(argv[1]) + 1) * sizeof(char);
                        if (!(cmd = malloc(block_size)))
                                goto error_out;
                        memset(cmd, 0, block_size);
                        snprintf(cmd, block_size, "%s %s", cmd_base, argv[1]);
                        process_command(cmd, s);
                        free(cmd);
                }
        }
        cli_loop(s);

exit_session:
        free_session(s);
        exit(EXIT_SUCCESS);

error_out:
        perror("[main()]");
        exit(EXIT_FAILURE);
}

#ifdef FAST_EXP
void print_fast_exp_status()
{
        cprintf("+ [ FastExp ]: Using Schraudolph's exp() approximation (c: %d)\n",
                EXP_C);
}
#endif /* FAST_EXP */

#ifdef _OPENMP
void print_openmp_status()
{
        cprintf("+ [ OpenMP ]: %d processor(s) available (%d thread(s) max)\n",
                        omp_get_num_procs(),
                        omp_get_max_threads());
        omp_sched_t k;
        int m;
        omp_get_schedule(&k, &m);
        // switch (k & ~omp_sched_monotonic) {
        switch (k) {
        case omp_sched_static:
                cprintf("+ [ OpenMP ]: Static schedule (chunk size: %d)\n", m);
                break;
        case omp_sched_dynamic:
                cprintf("+ [ OpenMP ]: Dynamic schedule (chunk size: %d)\n", m);
                break;
        case omp_sched_guided:
                cprintf("+ [ OpenMP ]: Guided schedule (chunk size: %d)\n", m);
                break;
        case omp_sched_auto:
                cprintf("+ [ OpenMP ]: Auto schedule\n");
                break;
        default:
                /* to handle new schedules */
                break;
        }
        /*
        if (omp_get_num_devices() > 0)
                cprintf("+ [ OpenMP ]: %d accelerator(s) available (default device: %d)\n",
                        omp_get_num_devices(),
                        omp_get_default_device()); */
}
#endif /* _OPENMP */

/* console message */
void cprintf(const char *fmt, ...)
{
        va_list args;
        va_start(args, fmt);
        vfprintf(stdout, fmt, args);
        va_end(args);
}

/* program message */
void mprintf(const char *fmt, ...)
{
        va_list args;
        fprintf(stderr,"> ");
        va_start(args, fmt);
        vfprintf(stdout, fmt, args);
        va_end(args);
}

/* error message */
void eprintf(const char *fmt, ...)
{
        va_list args;
        fprintf(stderr, "\x1b[31m");
        fprintf(stderr, "! ");
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);
        fprintf(stderr, "\x1b[0m");
}

/* progress message */
void pprintf(const char *fmt, ...)
{
        va_list args;
        fprintf(stdout, "%% ");
        va_start(args, fmt);
        vfprintf(stdout, fmt, args);
        va_end(args);
}
