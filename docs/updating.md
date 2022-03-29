# Weight updating


`set UpdateAlgorithm <name>`     Set weight update algorithm


## Update algorithms and parameters


* `steepest`                     Steepest (gradient) descent

`set LearningRate <val>`         Set learning rate (LR) coefficient

`set LRScaleFactor <val>`        Set LR scaling factor

`set LRScaleAfter <val>`         Scale LR after \%epochs

`set Momentum <val>`             Set momentenum (MN) coefficient

`set MNScaleFactor <val>`        Set MN scaling factor

`set MNScaleAfter <val>`         Scale MN after \%epochs

`set WeightDecay <val>`          Set weight decay (WD) coefficient

`set WDScaleFactor <val>`        Set WD scaling factor

`set WDScaleAfter <val>`         Scale WD after \%epochs

* `bounded`                      Bounded steepest descent

(see `steepest`)

* `rprop+|irprop+`               (modified) Rprop (+ weight backtracking)

`set RpropInitUpdate <val>`      Set initial update value for Rprop

`set RpropEtaMinus <val>`        Set Eta- for Rprop

`set RpropEtaPlus <val>`         Set Eta+ for Rprop

* `rprop-|irprop-`               (modified) Rprop (- weight backtracking)

(see `rprop+|irprop+` and also `steepest`)

* `qprop`                        Quick propagation

(see `steepest`)

* `dbd`                          Delta-Bar-Delta

`set DBDRateIncrement <val>`     Set Kappa for Delta-Bar-Delta

`set DBDRateDecrement <val>`     Set Phi for Delta-Bar-Delta

(also see `steepest`)


## Other relevant topics


* [learning](learning.md)                     Learning algorithms, parameters
