#!/bin/bash
# A script to setup a Python 3.5 virtual environment
# So that project Python setup doesn't mess up main machine
# or other projects.

on_error() {
    echo "Error at $(caller), aborting"
    # don't exit, the trap will break, but set the return code
    RETURN=1
}

POSITIONAL=()
while [[ $# -gt 0 ]]
    do
    key="$1"

    case $key in
        --style-only)
        STYLE_ONLY=true
        shift # past value
        ;;
        *)    # unknown option
        POSITIONAL+=("$1") # save it in an array for later
        shift # past argument
        ;;
    esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# script will always be sourced - so break the loop on error
# set RETURN value
RETURN=0
while true; do
    trap 'on_error; break' ERR
    SCRIPT_DIR=`dirname $BASH_SOURCE`
    ROOT_DIR="${SCRIPT_DIR}/.."
    SOURCE_DIR="${ROOT_DIR}/src"
    VE_DIR="${ROOT_DIR}/venv"
    if [ ! -d $VE_DIR ]; then
        echo Initializing virtualenv at $VE_DIR
        python3.5 -m venv $VE_DIR
    fi

    echo Entering Python 3.5 virtual environment at $VE_DIR
    source $VE_DIR/bin/activate
    pip install --upgrade pip

    if [ "$STYLE_ONLY" = true ]; then
        pip install --upgrade -r ${SOURCE_DIR}/requirements_code_style.txt
        break
    fi

    # install testing requirements
    pushd ${SOURCE_DIR}
    pip install --upgrade -r requirements_test.txt
    popd

    break
done
trap - ERR
return $RETURN;