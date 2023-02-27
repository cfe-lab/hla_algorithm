# pyEasyHLA

[![pipeline status](https://git-int.cfenet.ubc.ca/drickett/pyeasyhla/badges/main/pipeline.svg)](https://git-int.cfenet.ubc.ca/drickett/pyeasyhla/-/commits/main)
[![coverage report](https://git-int.cfenet.ubc.ca/drickett/pyeasyhla/badges/main/coverage.svg)](https://git-int.cfenet.ubc.ca/drickett/pyeasyhla/-/commits/main)
[![Latest Release](https://git-int.cfenet.ubc.ca/drickett/pyeasyhla/-/badges/release.svg)](https://git-int.cfenet.ubc.ca/drickett/pyeasyhla/-/releases)

[[_TOC_]]

## Objective

This is a python implementation of [BBLab's hla-easy.rb](https://github.com/cfe-lab/bblab-server/blob/main/alldata/hla_class/hla-easy.rb) file. The goal is to remove the ruby dependency from BBLab.

## Testing

The current validation method is to go to the [BBLab HLA Class tool page](https://hivresearchtools.bccfe.ca/django/tools/hla_class/), upload [test.fasta](https://github.com/cfe-lab/bblab-server/blob/main/tests/test.fasta) in `HLA Type C`, and assert that its output matches [the output kept in version control](https://github.com/cfe-lab/bblab-server/blob/main/tests/hla_class/HLA-C%20batch%20mode%20test%20data%20OUTPUT.csv).

## TODO

<https://waylonwalker.com/hatch-version/>
