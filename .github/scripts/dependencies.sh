#!/bin/bash
debug=0
while getopts do:m: name
do
    case $name in
    o)    os="$OPTARG";;
    m)    meson_version="$OPTARG";;
    d)    debug=1;;
    ?)   printf "Usage: %s: [-d] -o <os> -m <meson_version>\n" $0
          exit 2;;
    esac
done

if [[ ${debug} -eq 1 ]]
then
    PREFIX=echo

else
    PREFIX=
fi

echo $os

if [[ "$os" = "linux" ]]
then
    ${PREFIX} sudo apt-get update
    ${PREFIX} sudo apt-get install -y libfmt-dev catch2 librange-v3-dev
elif [[ "$os" = "osx" ]]
then
    ${PREFIX} brew install fmt catch2 range-v3
    echo "hi"
fi


${PREFIX} python -m pip install meson==${meson_version} ninja

