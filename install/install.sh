#!/bin/bash

echo

cd ..

wd=`pwd -P`

name="prs_basal_body_opro"
libname="prs_basal_body_opro_lib.py"

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


if [[ -L "$name" ]]; then
  unlink "$name"
fi

ln -s "$wd/main.py" "$name"


echo "Install Done."

echo

