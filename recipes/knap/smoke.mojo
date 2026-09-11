# =============================================================================
# Project     : Knap, a pure Mojo byte level BPE tokenizer
# File        : conda.recipe/smoke.mojo
# Purpose     : Package acceptance test. Imports Knap from the installed
#               conda package and decodes one token, with no source tree.
# Stage       : Distribution. See docs/PACKAGING.md
# Depends on  : knap.flat_vocab, resolved from $PREFIX/lib/mojo
# Invariants  : Imports nothing that needs a vocabulary file, so the test
#               passes or fails on the package rather than on a download.
# -----------------------------------------------------------------------------
# Author      : Olaf Yunus Laitinen Imanov <yunus.imanov@metropolia.fi>
# ORCID       : 0009-0006-5184-0810
# Affiliation : School of Information and Communication Technology,
#               Metropolia University of Applied Sciences
# -----------------------------------------------------------------------------
# SPDX-License-Identifier: EUPL-1.2
# Copyright 2026 Olaf Yunus Laitinen Imanov
# =============================================================================
"""Package acceptance test for the Knap conda package.

A package that builds but cannot be imported without the source tree is not
a package, and nothing in the build itself would notice. This program is run
by the recipe's test section against the installed artefact only, with
`-I $PREFIX/lib/mojo` and no path into the repository.

It is deliberately small. It builds a two byte vocabulary by hand and
decodes one token, which exercises the import path, the precompiled package,
and one real code path, and needs no vocabulary download to do it. A larger
test here would be testing Knap, which the suite already does, rather than
testing the package.

    mojo run -I "$PREFIX/lib/mojo" smoke.mojo
"""

from knap.flat_vocab import FlatVocab


def main() raises:
    """Decode one token from a hand built vocabulary.

    Raises:
        Error: if the package cannot be imported, or if the decode returns
            anything other than the two bytes that went in.
    """
    var data = List[UInt8]()
    data.append(UInt8(72))  # H
    data.append(UInt8(105))  # i

    var offsets: List[Int] = [0]
    var lengths: List[Int] = [2]
    var vocabulary = FlatVocab(data^, offsets^, lengths^)

    var decoded = vocabulary.decode([0])
    if decoded != String("Hi"):
        raise Error(
            String(
                t"knap package smoke test: decoded '{decoded}', expected 'Hi'"
            )
        )
    print("knap package smoke test: import and decode both work")


# =============================================================================
# End of file: conda.recipe/smoke.mojo
# =============================================================================
