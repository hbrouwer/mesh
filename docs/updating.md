# Weight updating


`set UpdateAlgorithm <name>`     Set weight update algorithm


## Update algorithms and parameters


* `steepest`                     Steepest (gradient) descent

`set LearningRate <value>`       Set learning rate (LR) coefficient

`set LRScaleFactor <value>`      Set LR scaling factor

`set LRScaleAfter <value>`       Scale LR after \%epochs

`set Momentum <value>`           Set momentenum (MN) coefficient

`set MNScaleFactor <value>`      Set MN scaling factor

`set MNScaleAfter <value>`       Scale MN after \%epochs

`set WeightDecay <value>`        Set weight decay (WD) coefficient

`set WDScaleFactor <value>`      Set WD scaling factor

`set WDScaleAfter <value>`       Scale WD after \%epochs

* `bounded`                      Bounded steepest descent

(see `steepest`)

* `rprop+|irprop+`               (modified) Rprop (+ weight backtracking)

`set RpropInitUpdate <value>`    Set initial update value for Rprop

`set RpropEtaMinus <value>`      Set Eta- for Rprop

`set RpropEtaPlus <value>`       Set Eta+ for Rprop

* `rprop-|irprop-`               (modified) Rprop (- weight backtracking)

(see `rprop+|irprop+` and `steepest`)

* `qprop`                        Quick propagation

(see `steepest`)

* `dbd`                          Delta-Bar-Delta

`set DBDRateIncrement <value>`   Set Kappa for Delta-Bar-Delta

`set DBDRateDecrement <value>`   Set Phi for Delta-Bar-Delta

(see `steepest`)


## Other relevant topics


* [learning](learning.md)                     Learning algorithms, parameters
