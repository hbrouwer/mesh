/*
 * session.c
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "session.h"

/**************************************************************************
 *************************************************************************/
struct session *create_session()
{
        struct session *s;
        if (!(s = malloc(sizeof(struct session))))
                goto error_out;
        memset(s, 0, sizeof(struct session));

        s->networks = create_array(TYPE_NETWORKS);

        return s;

error_out:
        perror("[create_session()]");
        return NULL;
}

/**************************************************************************
 *************************************************************************/
void dispose_session(struct session *s)
{
        dispose_array(s->networks);
        free(s);
}
