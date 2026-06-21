[![Test](https://github.com/rjzak/bwmd/actions/workflows/test.yml/badge.svg)](https://github.com/rjzak/bwmd/actions/workflows/test.yml)
[![Lint](https://github.com/rjzak/bwmd/actions/workflows/lint.yml/badge.svg)](https://github.com/rjzak/bwmd/actions/workflows/lint.yml)

## Burrows Wheeler Markov Distance (BWMD)

This crate provides a simple Rust implementation of the Burrows-Wheeler Markov Distance (BWMD) algorithm by Edward Raff.
* Original Python code: <https://github.com/EdwardRaff/pyBWMD>
* Paper: <https://arxiv.org/pdf/1912.13046.pdf>


Please see the readme in the original code for more details. As for this crate, it has a few functions:
* `vectorize()`: creates a vector from binary data
* `distance()`: calculates the distance between arrays of binary data.

A sparse vector implementation is also provided since the output vector is large (65,536). The sparse vector can convert to/from a dense vector, encode/decode from base64, and with the `serde` feature is able to be directly serialized and deserialized. For a very large file, the sparse vector may end up being larger than the original. Measure and test with your data.
