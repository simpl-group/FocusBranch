"""Message tags exchanged between the analyzer (main process) and its GPU / MILP workers.

Every message is either a bare tag or a tuple whose first element is a tag."""

# per image
TERMINATE = 'terminate'           # main -> worker: no more images
SKIP = 'skip'                     # main -> gpu: planning decided not to run the image
STOP = 'stop'                     # main -> worker: abandon the current image
STOPPED = 'stopped'               # worker -> main: acknowledged STOP

# planning (request / reply)
BOUND_CONCRETE = 'CC'             # main -> gpu: (BOUND_CONCRETE, combs) bound concrete nodes; replies depend on the phase
PROBE_DONE = 'finished probing'   # main -> gpu: (PROBE_DONE, first_all_safe_level)
SAMPLE_SUPER = 'sample-all-k'     # main -> gpu: (SAMPLE_SUPER, level) sample super nodes of every focus-set size

# verification
START_CONCRETE = 'CCSaP'          # main -> gpu: (START_CONCRETE, level) bound this worker's share of all level-l concrete nodes
START_SUPER = 'SCSaP'             # main -> gpu: (START_SUPER, geometry, chosen_indices, refinement_strategy)
SPLIT_PRUNE_CHECK = 'SaP&CC'      # main -> gpu: (SPLIT_PRUNE_CHECK, unknown_combs) branch, prune and bound the next level

# super branching (SCSaP) iteration
STATS = 'stats'                   # gpu -> main: (STATS, size, checked, verified, seconds)
UNKNOWN = 'unknown-combs'         # gpu -> main: (UNKNOWN, combs) unknown concrete nodes below level eps
SUSPECTS = 'adversarial-example-suspects'  # gpu -> main: (SUSPECTS, combs) unknown level-eps nodes, for MILP
DONE = 'done'                     # gpu -> main: (DONE,) this worker's share of the covering is verified
