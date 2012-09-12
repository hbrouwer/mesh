%%
%% dcg_gen.pl
%%
%% Copyright 2012 Harm Brouwer <me@hbrouwer.eu>
%%
%% Licensed under the Apache License, Version 2.0 (the "License");
%% you may not use this file except in compliance with the License.
%% You may obtain a copy of the License at
%%
%%     http://www.apache.org/licenses/LICENSE-2.0
%%
%% Unless required by applicable law or agreed to in writing, software
%% distributed under the License is distributed on an "AS IS" BASIS,
%% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
%% See the License for the specific language governing permissions and
%% limitations under the License.
%%

:- use_module(library(lists)).
:- use_module(library(random)).

:- dynamic(word/1).
:- dynamic(word_vector/2).
:- dynamic(word_vector_size/1).

:- dynamic(situation_vector/2).
:- dynamic(situation_vector_size/2).

term_expansion((H --> [B]),(H --> [B])) :-
        assert(word(B)).

assert_word_vectors :-
        retractall(word_vector(_,_)),
        findall(W,word(W),Ws),
        assert_word_vectors(Ws).

assert_word_vectors([]).
assert_word_vectors([W|Ws]) :-
        word_vector_size(S),
        random_bit_vector(S,BV),
        (   word_vector(_,BV)
        ->  !, assert_word_vectors([W|Ws])
        ;   !, assert(word_vector(W,BV)),
            assert_word_vectors(Ws)
        ).

random_bit_vector(0,[]).
random_bit_vector(N,[B|Bs]) :-
        N > 0,
        N0 is N - 1,
        random(0,2,B),
        random_bit_vector(N0,Bs).

required_word_vector_size(S) :-
        findall(W,word(W),Ws),
        length(Ws,N),
        S is ceiling(log(N) / log(2)).

assert_situation_vectors :-
        retractall(situation_vector(_,_)),
        situation_vector_size(S),
        findall(R,s(R,_,[]),Rs),
        list_to_set(Rs,Rs0),
        assert_situation_vectors(Rs0,S).

assert_situation_vectors(Rs,S) :-
        assert_situation_vectors(Rs,1,S).

assert_situation_vectors([],_,_).
assert_situation_vectors([R|Rs],N,S) :-
        create_situation_vector(N,S,V),
        assert(situation_vector(R,V)),
        N0 is N + 1,
        assert_situation_vectors(Rs,N0,S).

required_situation_vector_size(S) :-
        findall(T,s(T,_,[]),T),
        list_to_set(T,T0),
        length(T0,S).

create_situation_vector(N,S,V) :-
        create_situation_vector(1,N,S,V).

create_situation_vector(I,_,S,[]) :-
        I > S, !.
create_situation_vector(I,N,S,[0|V]) :-
        I \= N, !,
        I0 is I + 1,
        create_situation_vector(I0,N,S,V).
create_situation_vector(I,N,S,[1|V]) :-
        I == N, !,
        I0 is I + 1,
        create_situation_vector(I0,N,S,V).

word_list_to_vector([],[]).
word_list_to_vector([W|Ws],[V|Vs]) :-
        word_vector(W,V),
        word_list_to_vector(Ws,Vs).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generation                                                           %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

generate_theta_pairs(F) :-
        initialize_theta_pairs,
        findall(p(T,S),s(T,S,[]),Ps),
        open(F,write,Fs),
        write_theta_pairs(Ps,Fs), !,
        close(Fs).

initialize_theta_pairs :-
        retractall(word_vector_size(_)),
        required_word_vector_size(S),
        assert(word_vector_size(S)),
        assert_word_vectors,
        listing(word_vector_size),
        listing(word_vector).

write_theta_pairs([],_).
write_theta_pairs([p(T,S)|Ps],F) :-
        word_list_to_vector(T,T0),
        flatten(T0,T1),
        word_list_to_vector(S,S0),
        mesh_format_item(S,S0,T1,F),
        write_theta_pairs(Ps,F).

generate_situation_pairs(F) :-
        initialize_situation_pairs,
        findall(p(T,S),s(T,S,[]),Ps),
        open(F,write,Fs),
        write_situation_pairs(Ps,Fs), !,
        close(Fs).

initialize_situation_pairs :-
        initialize_theta_pairs,
        retractall(situation_vector_size(_)),
        required_situation_vector_size(S),
        assert(situation_vector_size(S)),
        assert_situation_vectors,
        listing(situation_vector_size),
        listing(situation_vector).

write_situation_pairs([],_).
write_situation_pairs([p(T,S)|Ps],F) :-
        situation_vector(T,T0),
        flatten(T0,T1),
        word_list_to_vector(S,S0),
        mesh_format_item(S,S0,T1,F),
        write_situation_pairs(Ps,F).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MESH input format                                                    %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mesh_format_item(S,Is,T,F) :-
        format(F,'Name \"',[]),
        mesh_format_name(S,F),
        length(Is,IL),
        format(F,'\" ~d\n',IL),
        mesh_format_inputs(Is,T,F),
        format(F,'\n',[]).

mesh_format_name([W],F) :-
        format(F,'~s',W).
mesh_format_name([W|Ws],F) :-
        format(F,'~s ',W),
        mesh_format_name(Ws,F).

mesh_format_inputs([I],T,F) :-
        format(F,'Input ',[]),
        mesh_format_vector(I,F),
        format(F,' Target ',[]),
        mesh_format_vector(T,F),
        format(F,'\n',[]).
mesh_format_inputs([I|Is],T,F) :-
        format(F,'Input ',[]),
        mesh_format_vector(I,F),
        format(F,'\n',[]),
        mesh_format_inputs(Is,T,F).

mesh_format_vector([I],F) :-
        format(F,'~d',I).
mesh_format_vector([I|Is],F) :-
        format(F,'~d ',I),
        mesh_format_vector(Is,F).
