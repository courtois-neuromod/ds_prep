#!/bin/bash


remote=${1:-ria-sequoia}
for b in $(git branch -a | grep -E "remotes/$remote/p0._[a-z0-9]+[0-9]{2,3}") ; do
    git merge --no-edit -X ours ${b#remotes/}
    git rm .heudiconv
    GIT_EDITOR=/bin/true git commit --amend
done
