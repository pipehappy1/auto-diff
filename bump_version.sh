#!/bin/bash

cargo_file="Cargo.toml"

function validate_semversion() {
    version=$1
    if [[ "${version}" =~ ^v.+$ ]]; then
	version="${version:1}"
    else
	echo "bad version: ${version}"
	exit 1
    fi
    
    if [[ "${version}" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-((0|[1-9][0-9]*|[0-9]*[a-zA-Z-][0-9a-zA-Z-]*)(\.(0|[1-9][0-9]*|[0-9]*[a-zA-Z-][0-9a-zA-Z-]*))*))?(\+([0-9a-zA-Z-]+(\.[0-9a-zA-Z-]+)*))?$ ]]; then
	echo "${version}"
    else
	echo "bad version: ${version}"
	exit 1
    fi
}

function bump_version_usage() {
    echo
    echo "Usage: bump_version.sh [OPTIONS] VERSION_OLD VERSION_NEW"
    echo
    echo "bump the cargo version through the package."
    echo
    echo "Options:"
    echo "  -v        Verbose output, prints errors and echos the raw version on success"
    echo "  -t        Run tests for this script"
    echo "  -h        Print usage"
    echo
    echo "Run like: bump_version.sh -v v0.5.5 v0.5.6"
    echo
    echo
}

function bump_version() {
    test=0
    verbose=0

    while getopts ":vt" opt; do
	case $opt in
	    t) test=1
	       ;;
	    v) verbose=1
	       ;;
	    \?) echo "Invalid option -$OPTARG" >&2; echo; bump_version_usage; exit 1
		;;
	esac
    done

    shift $(($OPTIND - 1))
    version_old=$1
    version_new=$2

    semver_old=$(validate_semversion "${version_old}")
    semver_new=$(validate_semversion "${version_new}")

    line_old="version = \"${semver_old}\""
    line_new="version = \"${semver_new}\""

    find . -type f  -name "${cargo_file}" -exec perl -pi \
	 -e "s|${line_old}|${line_new}|g;" \
	 {} +

    echo "bump version from ${semver_old} to ${semver_new} ..."

}

bump_version "$@"
