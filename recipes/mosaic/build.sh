#
# build.sh
# mosaic
#
# Created by Christian Bator on 01/02/2025
#

set -euo pipefail

#
# Locations
#
mkdir -p $PREFIX/lib/mojo
mkdir -p $PREFIX/lib/mosaic

#
# Build libcodec
#
if [[ -z "${OSX_ARCH+x}" ]]; then
    clang -fPIC -shared -Wall -Werror -o $PREFIX/lib/mosaic/libcodec.dylib libcodec/libcodec.c
else
    clang -fPIC -shared -Wall -Werror -DSTBI_NEON -o $PREFIX/lib/mosaic/libcodec.dylib libcodec/libcodec.c
fi

#
# Build libvisualizer
#
if [[ -z "${OSX_ARCH+x}" ]]; then
    echo "> Unsupported platform for libvisualizer: $(uname)"
else
    swift_source_files=$(find libvisualizer/mac/MacVisualizer -name "*.swift")
    swiftc -emit-library -o $PREFIX/lib/mosaic/libvisualizer.dylib $swift_source_files
fi

#
# Build mosaic
#
mojo package mosaic -o $PREFIX/lib/mojo/mosaic.mojopkg
