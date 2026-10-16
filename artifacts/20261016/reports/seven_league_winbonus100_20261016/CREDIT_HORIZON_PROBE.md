# Direct GAE trace probe

Synthetic zero-value single-reward pulse in a real SB3 buffer reconstructed from saved learner configuration,not a saved historical rollout. Tests direct GAE recurrence only. Actual learned critic values and bootstrapping can propagate later rewards;this is not proof of inability to learn long-term behavior or improvement from changing lambda. Current training settings were not changed.

|Gamma|Lambda|Delay (s)|Analytical trace weight|Actual buffer advantage|
|---|---|---:|---:|---:|
|0.999|0.99|5|0.33118319|0.33118242|
|0.999|0.99|10|0.10968230|0.10968176|
|0.999|0.99|20|0.01203021|0.01203009|
|0.999|0.99|30|0.00131950|0.00131948|
|0.9995|0.99|5|0.34817638|0.34817615|
|0.9995|0.99|10|0.12122679|0.12122663|
|0.9995|0.99|20|0.01469593|0.01469589|
|0.9995|0.99|30|0.00178154|0.00178153|
|0.9995|0.999|5|0.86065415|0.86065596|
|0.9995|0.999|10|0.74072557|0.74072903|
|0.9995|0.999|20|0.54867437|0.54867977|
|0.9995|0.999|30|0.40641713|0.40642309|
