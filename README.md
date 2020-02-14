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

#### XML Output Example

```bash
python -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
    header tag \
    --batch-size="10" \
    --input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.test.gz \
    --model-path="https://github.com/kermitt2/grobid/raw/0.5.6/grobid-home/models/header/" \
    --limit="1" \
    --tag-output-format="xml" \
    2> /tmp/tag.log \
    | xmllint --format /dev/stdin
```

With the result:

```xml
<?xml version="1.0"?>
<xml>
  <p>
    <title>Markov Chain Algorithms for Planar Lattice Structures</title>
    <author>Michael Luby y Dana Randall z Alistair Sinclair</author>
    <abstract>Abstract Consider the following Markov chain , whose states are all domino tilings of a 2n &#x6EF59; 2n chessboard : starting from some arbitrary tiling , pick a 2 &#x6EF59; 2 window uniformly at random . If the four squares appearing in this window are covered by two parallel dominoes , rotate the dominoes in place . Repeat many times . This process is used in practice to generate a tiling , and is a tool in the study of the combinatorics of tilings and the behavior of dimer systems in statistical physics . Analogous Markov chains are used to randomly generate other structures on various two - dimensional lattices . This paper presents techniques which prove for the &#x6EF59;rst time that , in many interesting cases , a small number of random moves suuce to obtain a uniform distribution .</abstract>
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
