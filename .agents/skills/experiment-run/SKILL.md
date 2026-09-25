---
name: experiment-run
description: "Prepare and execute a reproducible E3-SSE experiment with frozen configuration, smoke test, provenance record, output isolation, and failure logging."
---

# Experiment Run

Before a large run:

1. confirm Git status and commit;
2. select the configuration;
3. confirm input dataset/split;
4. confirm model/checkpoint;
5. confirm seed;
6. run a small smoke test;
7. create a unique output directory;
8. record provenance using:

   python .agents/scripts/record_provenance.py \
       --output OUTPUT_DIR \
       --config CONFIG \
       --command "COMMAND"

9. run the production command;
10. preserve logs and failures;
11. never write outputs into raw-data directories.
