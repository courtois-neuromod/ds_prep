#!/bin/bash

remote=${1:-ria-sequoia}
for b in $(git branch -a | grep -E "remotes/$remote/p0._[a-z0-9]+[0-9]{2,3}") ; do
    echo $b
    if [ "$(git merge --no-edit -X ours ${b#remotes/})" = 'Already up to date.' ] ;then
	continue;
    fi
    git rm .heudiconv
    GIT_EDITOR=/bin/true git commit --amend
done
