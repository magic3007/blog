#!/usr/bin/env bash

echo $1
sed -i -E "s/\!\[([^]]*)\]\(([^\)]*)\)/\{\%\ include\ img.html\ src=\"\2\"\ alt=\"\1\"\ \%\} /" $1