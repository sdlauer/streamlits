#!/usr/bin/env python3

import sys
import os
import pkg_resources


def get_pkgs(reqs_file="requirements_orig.txt"):
    if reqs_file and os.path.isfile(reqs_file):
        ret = dict()
        with open(reqs_file) as f:
            for item in f.readlines():
                name, ver = item.strip("\n").split("==")[:2]
                ret[name] = ver, ()
        return ret
    else:
        return {
                   item.project_name: (item.version, tuple([dep.name for dep in item.requires()])) for item in pkg_resources.working_set
               }


def print_pkg_data(text, pkg_info):
    print("{:s}\nSize: {:d}\n\n{:s}".format(text, len(pkg_info), "\n".join(["{:s}=={:s}".format(*item) for item in pkg_info])))


def main():
    pkgs = get_pkgs(reqs_file=None)
    full_pkg_info = [(name, data[0]) for name, data in sorted(pkgs.items())]
    print_pkg_data("----------FULL LIST----------", full_pkg_info)

    deps = set()
    for name in pkgs:
        deps = deps.union(pkgs[name][1])
    min_pkg_info = [(name, data[0]) for name, data in sorted(pkgs.items()) if name not in deps]
    print_pkg_data("\n----------MINIMAL LIST----------", min_pkg_info)


if __name__ == "__main__":
    print("Python {:s} on {:s}\n".format(sys.version, sys.platform))
    main()
