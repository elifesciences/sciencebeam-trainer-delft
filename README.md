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
- Support for Wapiti models

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

### Train Sub Command

Training a model comes with many parameters. The following is an example to run the training without recommending parameters.

```bash
python -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
    header train \
    --batch-size="10" \
    --embedding="https://github.com/elifesciences/sciencebeam-models/releases/download/v0.0.1/glove.6B.50d.txt.xz" \
    --max-sequence-length="100" \
    --input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.train.gz \
    --limit="100" \
    --early-stopping-patience="3" \
    --max-epoch="50"
```

### Train Eval Sub Command

The `train_eval` sub command is combining the `train` and `eval` command. It is reserving a slice of the input for the evaluation.

```bash
python -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
    header train_eval \
    --batch-size="10" \
    --embedding="https://github.com/elifesciences/sciencebeam-models/releases/download/v0.0.1/glove.6B.50d.txt.xz" \
    --max-sequence-length="100" \
    --input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.train.gz \
    --limit="100" \
    --early-stopping-patience="3" \
    --max-epoch="50"
```

If you rather want to provide separate evaluation data:

```bash
python -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
    header train_eval \
    --batch-size="10" \
    --embedding="https://github.com/elifesciences/sciencebeam-models/releases/download/v0.0.1/glove.6B.50d.txt.xz" \
    --max-sequence-length="100" \
    --input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.train.gz \
    --limit="100" \
    --eval-input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.test.gz \
    --eval-limit="100" \
    --early-stopping-patience="3" \
    --max-epoch="50"
```

You can also train without using word embedding:

```bash
python -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
    header train_eval \
    --batch-size="10" \
    --no-embedding \
    --max-sequence-length="100" \
    --input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.train.gz \
    --limit="100" \
    --early-stopping-patience="3" \
    --max-epoch="50"
```

Train with additional token features:

```bash
python -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
    header train_eval \
    --batch-size="10" \
    --embedding="https://github.com/elifesciences/sciencebeam-models/releases/download/v0.0.1/glove.6B.50d.txt.xz" \
    --additional-token-feature-indices="0" \
    --max-char-length="60" \
    --max-sequence-length="100" \
    --input="https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-segmentation.train.gz" \
    --limit="100" \
    --early-stopping-patience="3" \
    --max-epoch="50"
```

### Training very long sequences

Some training sequences can be very long and may exceed the available memory. This is in particular an issue when training the sequences.

Some approches to deal with the issue.

#### Truncate the sequences to a maximum length

By passing in the `--max-sequence-length`, sequences are being truncated.
In that case the model will not be trained on any data beyond the max sequence length.

```bash
python -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
    header train_eval \
    --batch-size="10" \
    --embedding="https://github.com/elifesciences/sciencebeam-models/releases/download/v0.0.1/glove.6B.50d.txt.xz" \
    --max-sequence-length="100" \
    --input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.train.gz \
    --limit="100" \
    --early-stopping-patience="3" \
    --max-epoch="50"
```

#### Training using [truncated BPTT](https://en.wikipedia.org/wiki/Backpropagation_through_time#Pseudocode) (Backpropagation through time)

This requires the LSTMs to be *stateful* (the state from the previous batch is passed on to the next). The `--stateful` flag should be passed in, and the `--input-window-stride` should be the same as `--max-sequence-length`

```bash
python -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
    header train_eval \
    --batch-size="10" \
    --embedding="https://github.com/elifesciences/sciencebeam-models/releases/download/v0.0.1/glove.6B.50d.txt.xz" \
    --max-sequence-length="100" \
    --input-window-stride="100" \
    --stateful \
    --input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.train.gz \
    --limit="100" \
    --early-stopping-patience="3" \
    --max-epoch="50"
```

Unfortunately the current implementation is very slow and training time might increase significantly.

#### Training using window slices

The alternative to the above is to not use *stateful* LSTMs but still pass in the input data using sliding windows.
To do that, do not pass `--stateful`. But use `--input-window-stride` which is equal or less to `--max-sequence-length`.

```bash
python -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
    header train_eval \
    --batch-size="10" \
    --embedding="https://github.com/elifesciences/sciencebeam-models/releases/download/v0.0.1/glove.6B.50d.txt.xz" \
    --max-sequence-length="100" \
    --input-window-stride="50" \
    --input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.train.gz \
    --limit="100" \
    --early-stopping-patience="3" \
    --max-epoch="50"
```

This will not allow the LSTM to capture long term dependencies beyond the max sequence length but it will allow it to have seen all of the data, in chunks. Therefore max sequence length should be large enough, which depends on the model.

### Eval Sub Command

```bash
python -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
    eval \
    --batch-size="10" \
    --input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.test.gz \
    --model-path="https://github.com/kermitt2/grobid/raw/0.5.6/grobid-home/models/header/" \
    --limit="10" \
    --quiet
```

### Tag Sub Command

The `tag` sub command supports multiple output formats:

- `json`: more detailed tagging output
- `data`: data output with features but label being replaced by predicted label
- `text`: not really a tag output as it just outputs the input text
- `xml`: uses predicted labels as XML elements
- `xml_diff`: same as `xml` but it is showing a diff between expected and predicted results

#### XML Output Example

```bash
python -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
    tag \
    --batch-size="10" \
    --input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.test.gz \
    --model-path="https://github.com/kermitt2/grobid/raw/0.5.6/grobid-home/models/header/" \
    --limit="1" \
    --tag-output-format="xml" \
    --quiet
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
    tag \
    --batch-size="10" \
    --input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.test.gz \
    --model-path="https://github.com/kermitt2/grobid/raw/0.5.6/grobid-home/models/header/" \
    --limit="2" \
    --tag-output-format="xml_diff" \
    --quiet
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
    tag \
    --batch-size="10" \
    --input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.test.gz \
    --model-path="https://github.com/kermitt2/grobid/raw/0.5.6/grobid-home/models/header/" \
    --limit="1" \
    --tag-output-format="data" \
    --quiet \
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
    tag \
    --batch-size="10" \
    --input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.test.gz \
    --model-path="https://github.com/kermitt2/grobid/raw/0.5.6/grobid-home/models/header/" \
    --limit="1" \
    --tag-output-format="text" \
    --quiet
```

With the result:

```text
Markov Chain Algorithms for Planar Lattice Structures Michael Luby y Dana Randall z Alistair Sinclair Abstract Consider the following Markov chain , whose states are all domino tilings of a 2n 񮽙 2n chessboard : starting from some arbitrary tiling , pick a 2 񮽙 2 window uniformly at random . If the four squares appearing in this window are covered by two parallel dominoes , rotate the dominoes in place . Repeat many times . This process is used in practice to generate a tiling , and is a tool in the study of the combinatorics of tilings and the behavior of dimer systems in statistical physics . Analogous Markov chains are used to randomly generate other structures on various two - dimensional lattices . This paper presents techniques which prove for the 񮽙rst time that , in many interesting cases , a small number of random moves suuce to obtain a uniform distribution .
```

### Wapiti Sub Commands

The Wapiti sub commands allow to use a similar process for training, evaluating and tagging Wapiti models, as the sub commands for the other DL model(s) above.

Currently you would need to either install [Wapiti](https://wapiti.limsi.fr/) and make the `wapiti` command available in the path, or use the `--wapiti-install-source` switch to download and install a version from source.

#### Wapiti Train Sub Command

```bash
python -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
    header wapiti_train \
    --input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.train.gz \
    --wapiti-template=https://raw.githubusercontent.com/kermitt2/grobid/0.5.6/grobid-trainer/resources/dataset/header/crfpp-templates/header.template \
    --wapiti-install-source=https://github.com/kermitt2/Wapiti/archive/5f9a52351fddf21916008daa4becd41d56e7f608.tar.gz \
    --output="data/models" \
    --limit="100" \
    --max-epoch="10"
```

#### Wapiti Train Eval Sub Command

```bash
python -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
    header wapiti_train_eval \
    --input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.train.gz \
    --eval-input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.test.gz \
    --wapiti-template=https://raw.githubusercontent.com/kermitt2/grobid/0.5.6/grobid-trainer/resources/dataset/header/crfpp-templates/header.template \
    --output="data/models" \
    --limit="100" \
    --max-epoch="10"
```

#### Wapiti Eval Sub Command

```bash
python -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
    wapiti_eval \
    --input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.test.gz \
    --model-path="https://github.com/kermitt2/grobid/raw/0.5.6/grobid-home/models/header" \
    --quiet
```

#### Wapiti Tag Sub Command

```bash
python -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
    wapiti_tag \
    --input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.test.gz \
    --model-path="https://github.com/kermitt2/grobid/raw/0.5.6/grobid-home/models/header" \
    --limit="1" \
    --tag-output-format="xml_diff" \
    --quiet
```

### Input Info Sub Command

```bash
python -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
    input_info \
    --quiet \
    --input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.train.gz
```

Result:

```text
number of input sequences: 2538
sequence lengths: {'q.00': 1, 'q.25': 61.0, 'q.50': 178.0, 'q.75': 300.75, 'q1.0': 6606}
token lengths: {'q.00': 1, 'q.25': 1.0, 'q.50': 3.0, 'q.75': 7.0, 'q1.0': 142}
number of features: 31
inconsistent feature length counts: {31: 536893, 30: 12855}
examples with feature length=31:
die D Di Die Die e ie Die Die BLOCKSTART LINESTART NEWFONT HIGHERFONT 0 0 0 INITCAP NODIGIT 0 0 1 0 0 0 0 0 0 0 NOPUNCT 0 0
abscheidung A Ab Abs Absc g ng ung dung BLOCKIN LINEIN SAMEFONT SAMEFONTSIZE 0 0 0 INITCAP NODIGIT 0 0 0 0 0 0 0 0 0 0 NOPUNCT 0 0
strömender s st str strö r er der nder BLOCKIN LINEIN SAMEFONT SAMEFONTSIZE 0 0 0 NOCAPS NODIGIT 0 0 0 0 0 0 0 0 0 0 NOPUNCT 0 0
examples with feature length=30:
gudina G Gu Gud Gudi a na ina dina BLOCKSTART LINESTART LINEINDENT NEWFONT HIGHERFONT 0 0 0 INITCAP NODIGIT 0 0 0 0 0 0 0 0 NOPUNCT 0 0
et e et et et t et et et BLOCKIN LINEIN LINEINDENT NEWFONT SAMEFONTSIZE 0 0 0 NOCAPS NODIGIT 0 0 0 0 0 0 0 0 NOPUNCT 0 0
al a al al al l al al al BLOCKIN LINEIN LINEINDENT SAMEFONT SAMEFONTSIZE 0 0 0 NOCAPS NODIGIT 0 1 0 0 0 0 0 0 NOPUNCT 0 0
feature value lengths: {0: {'q.00': 1, 'q.25': 1.0, 'q.50': 3.0, 'q.75': 7.0, 'q1.0': 142}, 1: {'q.00': 1, 'q.25': 1.0, 'q.50': 1.0, 'q.75': 1.0, 'q1.0': 1}, 2: {'q.00': 1, 'q.25': 1.0, 'q.50': 2.0, 'q.75': 2.0, 'q1.0': 2}, 3: {'q.00': 1, 'q.25': 1.0, 'q.50': 3.0, 'q.75': 3.0, 'q1.0': 3}, 4: {'q.00': 1, 'q.25': 1.0, 'q.50': 3.0, 'q.75': 4.0, 'q1.0': 4}, 5: {'q.00': 1, 'q.25': 1.0, 'q.50': 1.0, 'q.75': 1.0, 'q1.0': 1}, 6: {'q.00': 1, 'q.25': 1.0, 'q.50': 2.0, 'q.75': 2.0, 'q1.0': 2}, 7: {'q.00': 1, 'q.25': 1.0, 'q.50': 3.0, 'q.75': 3.0, 'q1.0': 3}, 8: {'q.00': 1, 'q.25': 1.0, 'q.50': 3.0, 'q.75': 4.0, 'q1.0': 4}, 9: {'q.00': 7, 'q.25': 7.0, 'q.50': 7.0, 'q.75': 7.0, 'q1.0': 10}, 10: {'q.00': 6, 'q.25': 6.0, 'q.50': 6.0, 'q.75': 6.0, 'q1.0': 9}, 11: {'q.00': 7, 'q.25': 8.0, 'q.50': 8.0, 'q.75': 8.0, 'q1.0': 8}, 12: {'q.00': 9, 'q.25': 12.0, 'q.50': 12.0, 'q.75': 12.0, 'q1.0': 12}, 13: {'q.00': 1, 'q.25': 1.0, 'q.50': 1.0, 'q.75': 1.0, 'q1.0': 1}, 14: {'q.00': 1, 'q.25': 1.0, 'q.50': 1.0, 'q.75': 1.0, 'q1.0': 1}, 15: {'q.00': 1, 'q.25': 1.0, 'q.50': 1.0, 'q.75': 1.0, 'q1.0': 1}, 16: {'q.00': 6, 'q.25': 6.0, 'q.50': 6.0, 'q.75': 6.0, 'q1.0': 7}, 17: {'q.00': 7, 'q.25': 7.0, 'q.50': 7.0, 'q.75': 7.0, 'q1.0': 14}, 18: {'q.00': 1, 'q.25': 1.0, 'q.50': 1.0, 'q.75': 1.0, 'q1.0': 1}, 19: {'q.00': 1, 'q.25': 1.0, 'q.50': 1.0, 'q.75': 1.0, 'q1.0': 1}, 20: {'q.00': 1, 'q.25': 1.0, 'q.50': 1.0, 'q.75': 1.0, 'q1.0': 1}, 21: {'q.00': 1, 'q.25': 1.0, 'q.50': 1.0, 'q.75': 1.0, 'q1.0': 1}, 22: {'q.00': 1, 'q.25': 1.0, 'q.50': 1.0, 'q.75': 1.0, 'q1.0': 1}, 23: {'q.00': 1, 'q.25': 1.0, 'q.50': 1.0, 'q.75': 1.0, 'q1.0': 1}, 24: {'q.00': 1, 'q.25': 1.0, 'q.50': 1.0, 'q.75': 1.0, 'q1.0': 1}, 25: {'q.00': 1, 'q.25': 1.0, 'q.50': 1.0, 'q.75': 1.0, 'q1.0': 1}, 26: {'q.00': 1, 'q.25': 1.0, 'q.50': 1.0, 'q.75': 1.0, 'q1.0': 1}, 27: {'q.00': 1, 'q.25': 1.0, 'q.50': 1.0, 'q.75': 1.0, 'q1.0': 1}, 28: {'q.00': 3, 'q.25': 7.0, 'q.50': 7.0, 'q.75': 7.0, 'q1.0': 11}, 29: {'q.00': 1, 'q.25': 1.0, 'q.50': 1.0, 'q.75': 1.0, 'q1.0': 1}, 30: {'q.00': 1, 'q.25': 1.0, 'q.50': 1.0, 'q.75': 1.0, 'q1.0': 1}}
feature counts: {0: 1000, 1: 247, 2: 1000, 3: 1000, 4: 1000, 5: 265, 6: 1000, 7: 1000, 8: 1000, 9: 3, 10: 3, 11: 2, 12: 3, 13: 2, 14: 2, 15: 2, 16: 3, 17: 3, 18: 2, 19: 2, 20: 2, 21: 1, 22: 1, 23: 2, 24: 2, 25: 1, 26: 2, 27: 2, 28: 8, 29: 1, 30: 1}
suggested feature indices: 9-30
label counts: {'B-<title>': 2363, 'I-<title>': 24481, 'B-<author>': 2241, 'I-<author>': 25830, 'B-<reference>': 414, 'I-<reference>': 10121, 'B-<submission>': 409, 'I-<submission>': 3729, 'B-<abstract>': 1528, 'I-<abstract>': 269983, 'B-<affiliation>': 2782, 'I-<affiliation>': 23886, 'B-<address>': 2330, 'I-<address>': 13963, 'B-<date>': 658, 'I-<date>': 2204, 'B-<grant>': 105, 'I-<grant>': 4509, 'B-<email>': 891, 'I-<email>': 7796, 'B-<keyword>': 424, 'I-<keyword>': 7804, 'B-<entitle>': 24, 'I-<entitle>': 421, 'B-<pubnum>': 421, 'I-<pubnum>': 3755, 'B-<note>': 1823, 'I-<note>': 26033, 'B-<copyright>': 281, 'I-<copyright>': 5152, 'B-<date-submission>': 29, 'I-<date-submission>': 166, 'B-<intro>': 439, 'I-<intro>': 96944, 'B-<web>': 187, 'I-<web>': 3162, 'B-<phone>': 71, 'I-<phone>': 710, 'B-<dedication>': 22, 'I-<dedication>': 243, 'B-<degree>': 59, 'I-<degree>': 1355}
```

## Training in Google's AI Platform

You can train a model using Google's [AI Platform](https://cloud.google.com/ai-platform/). e.g.

```bash
gcloud beta ai-platform jobs submit training \
    --job-dir "gs://your-job-bucket/path" \
    --scale-tier=custom \
    --master-machine-type=n1-highmem-8 \
    --master-accelerator=count=1,type=NVIDIA_TESLA_K80 \
    --region=europe-west1 \
    --stream-logs \
    --module-name sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
    --package-path sciencebeam_trainer_delft \
    -- \
    header train_eval \
    --batch-size="10" \
    --embedding="https://github.com/elifesciences/sciencebeam-models/releases/download/v0.0.1/glove.6B.50d.txt.xz" \
    --max-sequence-length="500" \
    --input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.train.gz \
    --limit="10000" \
    --early-stopping-patience="10" \
    --max-epoch="50"
```

Or using the project's wrapper script which provides some default values:

```bash
./gcloud-ai-platform-submit.sh \
    --job-prefix "my_job_prefix" \
    --job-dir "gs://your-job-bucket/path" \
    --scale-tier=custom \
    --master-machine-type=n1-highmem-8 \
    --master-accelerator=count=1,type=NVIDIA_TESLA_K80 \
    --region=europe-west1 \
    -- \
    header train_eval \
    --batch-size="10" \
    --embedding="https://github.com/elifesciences/sciencebeam-models/releases/download/v0.0.1/glove.6B.50d.txt.xz" \
    --max-sequence-length="500" \
    --input=https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-0.5.6-header.train.gz \
    --limit="10000" \
    --early-stopping-patience="10" \
    --max-epoch="50"
```

(Alternatively you can train for free using Google Colab, see Example Notebooks above)

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
