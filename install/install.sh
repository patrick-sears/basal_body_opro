#!/bin/bash

echo

cd ..

wd=`pwd -P`

name="basal_body_opro"
libname="prs_basal_body_opro_lib.py"
linkname="prs-$name"

libdes=`python3 -m site --user-site`
if ! [[ -e "$libdes" ]]; then
  echo "Error.  Couldn't find lib destination."
  echo "  Looking for:  [$libdes]"
  echo "  We may have to create it manually."
  echo
  exit 1
fi

cp -f "lib/$libname" "$libdes/$libname"


if ! [[ -e "$HOME/bin" ]]; then
  echo "Error."; echo "  Couldn't find $HOME/bin"; echo; exit 1
fi

cd "$HOME/bin"


if [[ -L "$linkname" ]]; then
  unlink "$linkname"
fi

ln -s "$wd/main.py" "$linkname"


echo "Install Done."

echo

