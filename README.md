# FLAN
Code for multiple network alignment method FLAN and progNatalie.

## Citation

Malmi, E., Chawla, S., Gionis, A. "Lagrangian relaxations for multiple network alignment". Data Mining and Knowledge Discovery (2017) 31: 1331. ([PDF](http://users.ics.aalto.fi/gionis/network_alignment.pdf))

## Testing

To test your setup, run the following

    python align_multiple_networks.py toy_problem/edgefile toy_problem/similarityfile -m prognatalie++ -o toy_problem/output.txt

and compare the output to toy_problem/output_EXPECTED.txt.

## Usage (command line)

```
$ python align_multiple_networks.py --help
usage: align_multiple_networks.py [-h]
                                  [-m {cflan,flan,flan0,icm,isorankn,natalie,prognatalie,prognatalie++}]
                                  [-f F] [-g G] [-e ENTITIES] [-i ITERATIONS]
                                  [-o OUTPUT]
                                  edgefile similarityfile

positional arguments:
  edgefile              Path to file containing the edges of the input graphs.
                        One row contains one edge with the format: "graphID
                        edgeID1 edgeID2 (weight)", where edge weight is
                        optional (1 by default). NOTE: currently edge weights
                        are NOT supported.
  similarityfile        Path to file containing the candidate matches for
                        vertices and the associated similarity values. One row
                        contains one candidate match of a node: "graphID1
                        nodeID1 graphID2 nodeID2 similarity". Note that these
                        matches are directed (add nodeID2 -> nodeID1
                        separately) if you want. Also note that a node can be
                        matched with only the nodes specified in this file so
                        typically you want to add at least the option to map
                        to itself ("graphID1 nodeID1 graphID1 nodeID1 1").

optional arguments:
  -h, --help            show this help message and exit
  -m {cflan,flan,flan0,icm,isorankn,natalie,prognatalie,prognatalie++}, --method {cflan,flan,flan0,icm,isorankn,natalie,prognatalie,prognatalie++}
                        Alignment method. Default: flan
  -f F                  Cost of opening an entity. A larger value results in
                        fewer clusters. Default: 1
  -g G                  Discount for mapping neighbors to neighbors. Default:
                        0.5
  -e ENTITIES, --entities ENTITIES
                        Number of entities (must be specified when cFLAN is
                        used
  -i ITERATIONS, --iterations ITERATIONS
                        Number of iteration the Lagrange multipliers are
                        solved. Default: 300
  -o OUTPUT, --output OUTPUT
                        Output filename. Each line of the output contains a
                        list of graphID_nodeID pairs aligned to the same
                        cluster. Default: output_clusters.txt
```

## Notes

* Make sure that all nodes of an input graph can be aligned to distinct candidate nodes. You can add each node as its own candidate match to ensure that a solution exists.
* If you have two symmetric similarity tuples, e.g., tup1=(G2, v1, G1, u1, 1) and tup2=(G1, u1, G2, v1, 1), the one where the graph labels are strictly in an alphabetical order (in this case tup2 since 'G1' < 'G2') will be ignored. So you can either always add both tuples or only add the one where the graph labels are not in an alphabetical order. (This should be made more intuitive in the future.)

## Selected files

* align_multiple_networks.py: The main file to look at when aligning networks.
* lagrangian_relaxation/flan.py: Implementation of FLAN.
* lagrangian_relaxation/natalie.py: Implementation of Natalie.
* variables.py: Class Problem (contains the parameters of the problem) and class W (implementation of the linearized qudratic term w).
* toy_experiment.py: Synthetic graph alignment experiments.
* multiplex_experiment.py: Social network alignment experiments.
* genealogy_experiment.py: Family tree alignment experiments (data will be added later).

## Contact

eric dot malmi at aalto dot fi
