# ScienceBeam Trainer DeLFT

Work in-progress..

A thin(ish) wrapper around [DeLFT](https://github.com/kermitt2/delft) to enable training in the cloud.

Some of the main features:

- resources (model, data etc.) can be loaded from remote sources, currently:
  - HTTP (`https://`, `http://`)
  - Google Storage (`gs://`)
- resources can be saved to remote buckets, currently:
  - Google Storage (`gs://`)
- on-demand embedding download
- Docker container(s)

## Example Notebooks

- [train-header.ipynb](notebooks/train-header.ipynb) ([open in colab](https://colab.research.google.com/github/elifesciences/sciencebeam-trainer-delft/blob/develop/notebooks/train-header.ipynb))

## GROBID Trainer CLI

The GROBID Trainer CLI is the equivallent to [DeLFT's grobidTagger](https://github.com/kermitt2/delft/blob/master/grobidTagger.py). That is the main interface to interact with this project.

To get a list of all of the available parameters:

```bash
python -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer --help
```

### Using Docker Image

```bash
docker run --rm elifesciences/sciencebeam-trainer-delft_unstable \
python -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer --help
```

### Tag Command

The `tag` sub command supports multiple output formats:

- `json`: more detailed tagging output
- `data`: data output with features but label being replaced by predicted label
- `text`: not really a tag output as it just outputs the input text
- `xml`: uses predicted labels as XML elements
- `xml_diff`: same as `xml` but it is showing a diff between expected and predicted results

#### XML Output Example

```bash
python -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
    header tag \
    --batch-size="10" \
    --input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.test.gz \
    --model-path="https://github.com/kermitt2/grobid/raw/0.5.6/grobid-home/models/header/" \
    --limit="1" \
    --tag-output-format="xml" \
    2> /tmp/tag.log
```

With the result:

```xml
<xml>
  <p>
    <title>Markov Chain Algorithms for Planar Lattice Structures</title>
    <author>Michael Luby y Dana Randall z Alistair Sinclair</author>
    <abstract>Abstract Consider the following Markov chain , whose states are all domino tilings of a 2n &#x6EF59; 2n chessboard : starting from some arbitrary tiling , pick a 2 &#x6EF59; 2 window uniformly at random . If the four squares appearing in this window are covered by two parallel dominoes , rotate the dominoes in place . Repeat many times . This process is used in practice to generate a tiling , and is a tool in the study of the combinatorics of tilings and the behavior of dimer systems in statistical physics . Analogous Markov chains are used to randomly generate other structures on various two - dimensional lattices . This paper presents techniques which prove for the &#x6EF59;rst time that , in many interesting cases , a small number of random moves suuce to obtain a uniform distribution .</abstract>
  </p>
</xml>
```

#### XML Diff Output Example

```bash
python -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
    header tag \
    --batch-size="10" \
    --input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.test.gz \
    --model-path="https://github.com/kermitt2/grobid/raw/0.5.6/grobid-home/models/header/" \
    --limit="2" \
    --tag-output-format="xml_diff" \
    2> /tmp/tag.log
```

With the result (the second document contains differences):

```xml
  <xml>
    <p>
      <title>Markov Chain Algorithms for Planar Lattice Structures</title>
      <author>Michael Luby y Dana Randall z Alistair Sinclair</author>
      <abstract>Abstract Consider the following Markov chain , whose states are all domino tilings of a 2n 񮽙 2n chessboard : starting from some arbitrary tiling , pick a 2 񮽙 2 window uniformly at random . If the four squares appearing in this window are covered by two parallel dominoes , rotate the dominoes in place . Repeat many times . This process is used in practice to generate a tiling , and is a tool in the study of the combinatorics of tilings and the behavior of dimer systems in statistical physics . Analogous Markov chains are used to randomly generate other structures on various two - dimensional lattices . This paper presents techniques which prove for the 񮽙rst time that , in many interesting cases , a small number of random moves suuce to obtain a uniform distribution .</abstract>
    </p>
  
  
    <p>
      <title>Translucent Sums : A Foundation for Higher - Order Module Systems</title>
      <author>Mark Lillibridge</author>
      <date>May , 1997</date>
-     <pubnum>- - 95 -</pubnum>
+     <pubnum>- - 95 - of</pubnum>
?                     +++
-     <affiliation>of</affiliation>
    </p>
  </xml>
```

#### DATA Output Example

```bash
python -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
    header tag \
    --batch-size="10" \
    --input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.test.gz \
    --model-path="https://github.com/kermitt2/grobid/raw/0.5.6/grobid-home/models/header/" \
    --limit="1" \
    --tag-output-format="data" \
    2> /tmp/tag.log \
    | head -5
```

With the result:

```text
Markov markov M Ma Mar Mark v ov kov rkov BLOCKSTART LINESTART NEWFONT HIGHERFONT 0 0 0 INITCAP NODIGIT 0 0 0 0 0 0 0 0 0 0 NOPUNCT 0 0 B-<title>
Chain chain C Ch Cha Chai n in ain hain BLOCKIN LINEIN SAMEFONT SAMEFONTSIZE 0 0 0 INITCAP NODIGIT 0 0 1 0 0 0 0 0 0 0 NOPUNCT 0 0 I-<title>
Algorithms algorithms A Al Alg Algo s ms hms thms BLOCKIN LINEIN SAMEFONT SAMEFONTSIZE 0 0 0 INITCAP NODIGIT 0 0 1 0 0 0 0 0 0 0 NOPUNCT 0 0 I-<title>
for for f fo for for r or for for BLOCKIN LINEIN SAMEFONT SAMEFONTSIZE 0 0 0 NOCAPS NODIGIT 0 0 1 0 0 0 0 0 0 0 NOPUNCT 0 0 I-<title>
Planar planar P Pl Pla Plan r ar nar anar BLOCKIN LINEIN SAMEFONT SAMEFONTSIZE 0 0 0 INITCAP NODIGIT 0 0 0 0 0 0 0 0 0 0 NOPUNCT 0 0 I-<title>
```

#### Text Output Example

```bash
python -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
    header tag \
    --batch-size="10" \
    --input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.test.gz \
    --model-path="https://github.com/kermitt2/grobid/raw/0.5.6/grobid-home/models/header/" \
    --limit="1" \
    --tag-output-format="text" \
    2> /tmp/tag.log
```

With the result:

```text
Markov Chain Algorithms for Planar Lattice Structures Michael Luby y Dana Randall z Alistair Sinclair Abstract Consider the following Markov chain , whose states are all domino tilings of a 2n 񮽙 2n chessboard : starting from some arbitrary tiling , pick a 2 񮽙 2 window uniformly at random . If the four squares appearing in this window are covered by two parallel dominoes , rotate the dominoes in place . Repeat many times . This process is used in practice to generate a tiling , and is a tool in the study of the combinatorics of tilings and the behavior of dimer systems in statistical physics . Analogous Markov chains are used to randomly generate other structures on various two - dimensional lattices . This paper presents techniques which prove for the 񮽙rst time that , in many interesting cases , a small number of random moves suuce to obtain a uniform distribution .
```

## Checkpoints CLI

The checkpoints CLI tool is there to give you a summary of the saved checkpoints. Checkpoints are optionally saved during training, they allow you to resume model training or further evaluate performance at the individual checkpoints. Usually training will stop after the f1 score hasn't improved for a number of epochs. The last checkpoint may not be the best.

The checkpoints tool will sort by the f1 score and show the *n* (`limit`) top checkpoints.

```bash
python -m sciencebeam_trainer_delft.sequence_labelling.tools.checkpoints --help
```

### Checkpoints Text Output

```bash
python -m sciencebeam_trainer_delft.sequence_labelling.tools.checkpoints \
    --checkpoint="/path/to/checkpoints" \
    --limit=3 \
    --output-format=text
```

```text
best checkpoints:
00039: 0.5877923107411811 (/path/to/checkpoints/epoch-00039) (last)

00036: 0.5899450117831894 (/path/to/checkpoints/epoch-00036)

00034: 0.591387179996031 (/path/to/checkpoints/epoch-00034) (best)
```

### Checkpoints JSON Output

```bash
python -m sciencebeam_trainer_delft.sequence_labelling.tools.checkpoints \
    --checkpoint="/path/to/checkpoints" \
    --limit=3 \
    --output-format=json
```

```json
[
  {
    "loss": 40.520591011530236,
    "f1": 0.5877923107411811,
    "optimizer": {
      "type": "keras.optimizers.Adam",
      "lr": 0.0010000000474974513
    },
    "epoch": 39,
    "path": "/path/to/checkpoints/epoch-00039",
    "is_last": true,
    "is_best": false
  },
  {
    "loss": 44.48661111276361,
    "f1": 0.5899450117831894,
    "optimizer": {
      "type": "keras.optimizers.Adam",
      "lr": 0.0010000000474974513
    },
    "epoch": 36,
    "path": "/path/to/checkpoints/epoch-00036",
    "is_last": false,
    "is_best": false
  },
  {
    "loss": 47.80826501711393,
    "f1": 0.591387179996031,
    "optimizer": {
      "type": "keras.optimizers.Adam",
      "lr": 0.0010000000474974513
    },
    "epoch": 34,
    "path": "/path/to/checkpoints/epoch-00034",
    "is_last": false,
    "is_best": true
  }
]
```
