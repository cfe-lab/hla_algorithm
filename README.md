# BC-CfE HLA algorithm

[![test](https://github.com/cfe-lab/hla_algorithm/actions/workflows/test.yml/badge.svg)](https://github.com/cfe-lab/hla_algorithm/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/cfe-lab/hla_algorithm/graph/badge.svg?token=ZJQKPBRTQB)](https://codecov.io/gh/cfe-lab/hla_algorithm)
![GitHub Release](https://img.shields.io/github/v/release/cfe-lab/hla_algorithm?sort=semver)

## Objective

This is a Python implementation of the BC-CfE HLA algorithm.  This algorithm
has several implementations, including [BBLab's hla-easy.rb][bblab_hla] file.
This project aims to consolidate all of our HLA algorithm versions into a
single version.

[bblab_hla]: https://github.com/cfe-lab/bblab-server/blob/main/alldata/hla_class/hla-easy.rb

The `hla_algorithm` Python package provided by this repository exposes some
executables as well.

* [`interpret_from_json`][interpret_from_json]: A bare-bones script that
performs interpretation on an HLA sequence specified in JSON.
  * This script is also available as `python -m hla_algorithm` (in the
  environment where the module is installed).
* [`update_alleles`][update_alleles]: Retrieves alleles from IMGT/HLA (or an
alternative source) and creates a YAML configuration file suitable for use with
this code.  This is used periodically to update our lists of alleles with
up-to-date information.
* [`update_frequency_file`][update_frequency_file]: Updates an "old-style"
HLA frequency file into the format used by this algorithm.
  * As of the time of writing, the frequencies file has unknown origin and has
  never been updated.  Under typical operation, you'll likely never use this.
* [`reformat_old_alleles`][reformat_old_alleles]: Updates "old-style" HLA
allele lists into a YAML configuration file suitable for use with this code.
These allele lists may be either "unreduced" (with duplicates remaining) or
"reduced" (with duplicates collapsed into a single representative).  This
script can be used to update old allele lists that you wish to continue using
with this code, e.g. if you already have an old implementation of this
algorithm running somewhere and wish to preserve its functionality.

[interpret_from_json]: src/hla_algorithm/interpret_from_json.py
[update_alleles]: src/hla_algorithm/update_alleles.py
[update_frequency_file]: src/hla_algorithm/update_frequency_file.py
[reformat_old_alleles]: src/hla_algorithm/reformat_old_alleles.py

### Ruby wrapper

In addition to the Python module, this repository also contains a Ruby wrapper
enabling this code to be used in Ruby (through system calls).  The Ruby wrapper
is intended as a drop-in replacement for the Ruby-based HLA algorithm that has
been used at the CfE for years; once installed, this Ruby module provides an
`HLAAlgorithm` class with an `analyze` method, as the old code did.
Additionally, this `HLAAlgorithm` class can optionally be provided with new
configuration files, so that the alleles (and frequencies, if that ever happens)
can be updated without having to update the code.

This Ruby module works by simply packing up its inputs into a JSON file and
invoking `interpret_from_json`.  Therefore, the Python module must be installed
on the system where you're using the Ruby module.

## Installation

This code may be installed from our git repo.  For example, using pip:

```
pip install git+https://github.com/cfe-lab/hla_algorithm.git@[tag]
```

Using uv:

```
uv add git+https://github.com/cfe-lab/hla_algorithm.git@[tag]
```

### Ruby wrapper

Once the Python module is installed, the Ruby module can be installed from our
GitHub package repository.  First, you need to authenticate with the GitHub
RubyGems registry.  In brief, a personal access token (classic) is required to
install from this registry (even though our repository is public).  Then you
must update your gem sources/bundler configuration with this token along with
the username of the token creator.  Full details may be found in
[the GitHub instructions][github_rubygem_auth].

From there, you can install using either `gem` or via a Gemfile (i.e. with
`bundler`).  Instructions may be found on
[the GitHub page for the package][github_package_page].

The Ruby module must be instructed on how to invoke `interpret_from_json`,
using the environment variable `HLA_INTERPRET_FROM_JSON`.  For example, if you
install Python using `uv` in your project directory (as recommended), then you
would set this to `uv run interpret_from_json`.  If your project is
Docker-based, then you can set this in your Dockerfile:

```
ENV HLA_INTERPRET_FROM_JSON="uv run interpret_from_json"
```

[github_rubygem_auth]: https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-rubygems-registry
[github_package_page]: https://github.com/cfe-lab/hla_algorithm/pkgs/rubygems/hla_algorithm
