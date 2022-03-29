# Networks


`createNetwork <name> <type>`    Create network of specified type
(type = `[ffn|srn|rnn]`)

`removeNetwork <name>`           Remove network from session

`networks`                       List all active networks

`changeNetwork <name>`           Change active network

`inspect`                        Show properties of active network


`init`                           Initialize network

`reset`                          Reset network

`train`                          Train network

`test`                           Test network on all items

`testVerbose`                    Show error for each item

`testItem <id>`                  Test network on specified item
(id = `[<name>|<number>]`)


`toggleResetContexts`            Toggle context resetting

`set InitContextUnits <val>`     Activation of initial context units


## Other relevant topics


* [groups](groups.md)                       Creating groups

* [projections](projections.md)                  Creating projections

* [training](training.md)                     Training networks

* [testing](testing.md)                      Testing networks
