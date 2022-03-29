# Distributed Situation-state Space (DSS)


## Comprehension scores


`dssTest`                        Show comprehension(target,output) for
each sentence in the active set

`dssScores <set> <sen>`          Show comprehension(event,output) at each
word, for each event in set

`dssInferences <set> <sen> <th>` Show each event in set that yields
comprehension(event,output) > |th|


## Information theory


`dssWordInfo <set> <sen>`        Show information-theoretic metrics for
each word, given a sentence set

`dssWriteWordInfo <set> <fn>`    Write information-theoretic metrics for
each word of each sentence to a file
