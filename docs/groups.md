# Groups


`createGroup <name> <size>`      Create group of specified size

`createBiasGroup <name>`         Create a bias group

`removeGroup <name>`             Remove group from network

`groups`                         List all groups of the active network

`attachBias <name>`              Attach a bias unit to a group

`set InputGroup <name>`          Set the input group of the network

`set OutputGroup <name>`         Set the output group of the network

`set ActFunc <name> <func>`      Set the activation function of a group

`set ErrFunc <name> <func>`      Set the error function of a group


`set ReLUAlpha <name> <value>`   Set alpha coeff. for Leaky ReLU and ELU

`set LogisicFSC <name> <value>`  Set logistic flat spot correction

`set LogisicGain <name> <value>` Set logistic gain coefficient


`showVector <type> <name>`       Show group vector
(type = `[units|error]`)


## Other relevant topics


* [projections](projections.md)                  Creating projections between group

* [activation](activation.md)                   Activation functions

* [error](error.md)                        Error functions
