#!/bin/bash

if [ $TRAVIS_OS_NAME = 'osx' ]; then

    # Install some custom requirements on macOS
    # e.g. brew install pyenv-virtualenv

    case "${TOXENV}" in
        py36)
            # Install some custom Python 3.2 requirements on macOS
	    brew install python@3.6
            ;;
        py38)
            # Install some custom Python 3.3 requirements on macOS
	    brew install python@3.8
	    ;;
    esac
else
    # Install some custom requirements on Linux
fi
